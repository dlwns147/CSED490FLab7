"""Project 2: retrieval-augmented generation with Llama 3.2 1B on the Raspberry Pi (standalone, no web server).

    python3 rag_cli.py index                       # build the index from kb/*.txt (Wikipedia snapshots)
    python3 rag_cli.py add-url https://en.wikipedia.org/wiki/Raspberry_Pi    # add an external source
    python3 rag_cli.py docs                        # list the documents in the knowledge base
    python3 rag_cli.py search "Which team won UEFA Euro 2024?" -k 4
    python3 rag_cli.py ask "Which team won UEFA Euro 2024?" [-k 4] [--mode hybrid] [--no-rag]
    python3 rag_cli.py chat                        # interactive; /rag on|off, /k N, /quit

Pipeline:  question -> retrieve top-k chunks (dense + BM25, fused with RRF) -> put them in the prompt
           -> Llama 3.2 1B (Ollama, on the Pi) streams the answer -> print the answer and its sources.
"""
import argparse
import pathlib
import sys
import time

from ollama_client import Ollama
from rag import KnowledgeBase

HERE = pathlib.Path(__file__).resolve().parent
CHAT_MODEL, EMBED_MODEL = "llama3.2-1b", "nomic-embed-text"
SYSTEM_PROMPT = "You are a concise, helpful assistant running locally on a Raspberry Pi 5."
OPTIONS = {"num_ctx": 4096, "num_predict": 512}


def answer(ol, kb, question, rag=True, k=4, mode="hybrid", prompt=None, reverse=False, options=None,
           prefix="", on_token=None):
    """One RAG round trip.  Returns {reply, sources, stats}.  `prefix` (benchmarking only) is put in front
    of the user turn so that Ollama cannot reuse a cached prompt prefix."""
    t0 = time.perf_counter()
    hits, timing, content = [], {"embed_ms": 0.0, "search_ms": 0.0}, question
    if rag:
        hits, timing = kb.search(question, k=k, mode=mode)
        content = KnowledgeBase.augment(question, hits, prompt, reverse)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prefix + content}]
    reply, t_first, done = [], None, {}
    for chunk in ol.chat_stream(CHAT_MODEL, messages, {**OPTIONS, **(options or {})}):
        piece = chunk.get("message", {}).get("content", "")
        if piece:
            t_first = t_first or time.perf_counter()
            reply.append(piece)
            if on_token:
                on_token(piece)
        if chunk.get("done"):
            done = chunk
    t_end = time.perf_counter()
    gen, ns = done.get("eval_count", 0), 1e-9
    stats = {**timing,
             "ttft_ms": round((t_first - t0) * 1e3, 1) if t_first else None,
             "total_ms": round((t_end - t0) * 1e3, 1),
             "prompt_tokens": done.get("prompt_eval_count", 0),             # whole prompt
             "cached_tokens": done.get("prompt_eval_cached_count", 0),      # part reused from Ollama's KV cache
             "prefill_ms": round(done.get("prompt_eval_duration", 0) * ns * 1e3, 1),
             "gen_tokens": gen,
             "decode_tok_s": round(gen / (done["eval_duration"] * ns), 2) if done.get("eval_duration") else None}
    stats["computed_tokens"] = stats["prompt_tokens"] - stats["cached_tokens"]
    return {"reply": "".join(reply), "sources": hits, "stats": stats}


def show(ol, kb, q, rag, k, mode, prompt=None):
    print("Bot: ", end="", flush=True)
    r = answer(ol, kb, q, rag, k, mode, prompt, on_token=lambda t: print(t, end="", flush=True))
    s = r["stats"]
    print()
    for h in r["sources"]:
        print(f"   [{h['rank']}] {KnowledgeBase.label(h)}  (score {h['score']})")
    print(f"   -- retrieval {s['embed_ms'] + s['search_ms']:.0f} ms | first token {s['ttft_ms'] / 1000:.2f} s | "
          f"prompt {s['prompt_tokens']} tok | {s['gen_tokens']} tok at {s['decode_tok_s']} tok/s", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["index", "add-url", "remove", "docs", "search", "ask", "chat"])
    ap.add_argument("text", nargs="?", default="")
    ap.add_argument("-k", type=int, default=4, help="passages to retrieve (default 4)")
    ap.add_argument("--mode", default="hybrid", choices=["hybrid", "dense", "bm25"])
    ap.add_argument("--prompt", default=None, choices=list(KnowledgeBase.PROMPTS), help="instruction variant")
    ap.add_argument("--no-rag", action="store_true", help="ask the model without retrieval")
    ap.add_argument("--ollama", default="http://127.0.0.1:11434")
    ap.add_argument("--index", default=str(HERE / "index"))
    a = ap.parse_args()

    ol = Ollama(a.ollama)
    kb = KnowledgeBase(ol, EMBED_MODEL, a.index)
    if a.cmd == "index":
        t = time.perf_counter()
        n = sum(kb.add_file(p) for p in sorted((HERE / "kb").glob("*.txt")))
        print(f"indexed {n} chunks in {time.perf_counter() - t:.1f} s -> {a.index}")
    elif a.cmd == "add-url":
        t = time.perf_counter()
        title, n = kb.add_url(a.text)
        print(f"added '{title}': {n} chunks in {time.perf_counter() - t:.1f} s")
    elif a.cmd == "remove":
        print(f"removed {kb.remove(a.text)} chunks")
    elif a.cmd == "docs":
        for d, n in kb.docs().items():
            print(f"{n:4d}  {d}")
    elif a.cmd == "search":
        hits, tm = kb.search(a.text, k=a.k, mode=a.mode)
        for h in hits:
            print(f"[{h['rank']}] {KnowledgeBase.label(h)}  (score {h['score']})\n    {h['text'][:160]}...")
        print(f"-- {tm['embed_ms'] + tm['search_ms']:.0f} ms")
    elif a.cmd == "ask":
        show(ol, kb, a.text, not a.no_rag, a.k, a.mode, a.prompt)
    else:
        rag, k = not a.no_rag, a.k
        print(f"RAG chat with {CHAT_MODEL} ({len(kb.chunks)} chunks).  /rag on|off, /k N, /quit")
        while True:
            try:
                q = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if q in ("/quit", "/exit"):
                break
            if q.startswith("/rag"):
                rag = q.endswith("on"); print(f"   RAG {'on' if rag else 'off'}"); continue
            if q.startswith("/k "):
                k = int(q.split()[1]); print(f"   k = {k}"); continue
            if q:
                show(ol, kb, q, rag, k, a.mode, a.prompt)


if __name__ == "__main__":
    sys.exit(main())
