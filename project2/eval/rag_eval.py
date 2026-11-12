"""Evaluate the RAG pipeline on the Pi (calls rag_cli.answer / KnowledgeBase.search directly).

1. retrieval : recall@k for BM25 / dense / hybrid, k = 1..10 (a hit = a chunk of the right article that contains the answer)
2. answers   : accuracy of Llama 3.2 1B with no retrieval (closed book) and with each retriever / k
               (temperature 0).  An answer is correct when it contains every required key fact
               (any accepted spelling), case-insensitive.  6 extra questions have no answer in the
               knowledge base; for those we count how often the model says it does not know.
3. latency   : time to first token vs. k, with Ollama's prompt cache defeated (every passage is new)
               and for the same request repeated (cached).

    cd project2 && python3 rag_cli.py index && python3 eval/rag_eval.py
"""
import argparse
import json
import pathlib
import re
import statistics
import sys
import time
import uuid

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from ollama_client import Ollama          # noqa: E402
from rag import KnowledgeBase             # noqa: E402
from rag_cli import EMBED_MODEL, answer   # noqa: E402


def norm(s):
    return re.sub(r"\s+", " ", s.lower())


def correct(answer, keys):
    a = norm(answer)
    return all(any(norm(alt) in a for alt in group) for group in keys)


ABSTAIN = re.compile(r"(don't|do not|does not) know|not (explicitly )?(mentioned|stated|specified|provided|given)|"
                     r"(does not|doesn't|do not) (mention|contain|specify|say)|no (information|mention)|cannot find|"
                     r"can't find|unable to find|(don't|do not) have (any |specific )?information|not aware of")


def abstained(answer):
    return bool(ABSTAIN.search(norm(answer).replace("\u2019", "'")))


def gold_rank(hits, q):
    """Rank of the first retrieved chunk that comes from the question's article and contains the answer."""
    for h in hits:
        if h["doc"].removeprefix("Wikipedia: ") == q["gold"] and correct(h["text"], q["keys"]):
            return h["rank"]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ollama", default="http://127.0.0.1:11434")
    ap.add_argument("--index", default=str(pathlib.Path(__file__).resolve().parent.parent / "index"))
    ap.add_argument("--questions", default=str(pathlib.Path(__file__).with_name("questions.jsonl")))
    ap.add_argument("--out", default=str(pathlib.Path(__file__).with_name("results")))
    ap.add_argument("--configs", default="closed,bm25:4,dense:4,hybrid:4,hybrid:1,hybrid:2,hybrid:6",
                    help="mode[:k[:prompt[:r]]], r = best passage last")
    ap.add_argument("--skip_retrieval", action="store_true")
    ap.add_argument("--skip_latency", action="store_true")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    allq = [json.loads(l) for l in open(a.questions)]
    qs = [q for q in allq if not q.get("unanswerable")]          # retrieval metrics: answerable only
    ol = Ollama(a.ollama)
    kb = KnowledgeBase(ol, EMBED_MODEL, a.index)
    assert kb.chunks, "empty index: run `python3 rag_cli.py index` first"

    # ---- 1. retrieval only
    ret = {}
    if a.skip_retrieval:
        qs_ret = []
    else:
        qs_ret = qs
    for mode in (("bm25", "dense", "hybrid") if qs_ret else ()):
        ranks, ms = [], []
        for q in qs:
            hits, tm = kb.search(q["q"], k=10, mode=mode)
            ranks.append(gold_rank(hits, q))
            ms.append(tm["embed_ms"] + tm["search_ms"])
        ret[mode] = {"recall": {k: sum(1 for x in ranks if x and x <= k) / len(qs) for k in range(1, 11)},
                     "mrr": sum(1 / x for x in ranks if x) / len(qs), "ranks": ranks,
                     "latency_ms": sum(ms) / len(ms)}
        print(f"{mode:7s} R@1 {ret[mode]['recall'][1]:.2f}  R@4 {ret[mode]['recall'][4]:.2f}  "
              f"R@10 {ret[mode]['recall'][10]:.2f}  MRR {ret[mode]['mrr']:.3f}  {ret[mode]['latency_ms']:.0f} ms", flush=True)
    if ret:
        (out / "retrieval.json").write_text(json.dumps(ret, indent=1))

    # ---- 2. end-to-end answers
    sp = pathlib.Path(out / "summary.json")
    summary = json.loads(sp.read_text()) if sp.exists() else {}
    for cfg in a.configs.split(","):
        parts = cfg.split(":")
        mode, k, prompt, rev = (parts + ["", "4", "", ""][len(parts):])[:4]
        if not a.configs.strip():
            break
        rows = []
        for q in allq:
            t = time.perf_counter()
            r = answer(ol, kb, q["q"], rag=mode != "closed", k=int(k), mode=mode or "hybrid", prompt=prompt or None,
                       reverse=rev == "r", options={"temperature": 0, "num_predict": 96, "seed": 0})
            wall = time.perf_counter() - t
            st = r["stats"]
            rows.append({"id": q["id"], "q": q["q"], "answer": r["reply"], "unanswerable": bool(q.get("unanswerable")),
                         "correct": abstained(r["reply"]) if q.get("unanswerable") else correct(r["reply"], q["keys"]),
                         "abstained": abstained(r["reply"]),
                         "gold_rank": gold_rank(r.get("sources", []), q) if mode != "closed" else None,
                         "wall_s": round(wall, 2), **{x: st.get(x) for x in
                         ("ttft_ms", "total_ms", "prompt_tokens", "cached_tokens", "prefill_ms", "gen_tokens",
                          "decode_tok_s", "embed_ms", "search_ms")}})
        ans = [x for x in rows if not x["unanswerable"]]
        una = [x for x in rows if x["unanswerable"]]
        acc = sum(x["correct"] for x in ans) / len(ans)
        mean = lambda key: sum(x[key] or 0 for x in rows) / len(rows)
        summary[cfg] = {"accuracy": acc, "n": len(ans),
                        "false_abstain": sum(x["abstained"] for x in ans) / len(ans),
                        "abstain_unanswerable": sum(x["abstained"] for x in una) / max(1, len(una)), "ttft_ms": mean("ttft_ms"), "total_ms": mean("total_ms"),
                        "prompt_tokens": mean("prompt_tokens"), "prefill_ms": mean("prefill_ms")}
        if mode != "closed":   # answer accuracy split by whether the answer was retrieved
            hit = [x for x in ans if x["gold_rank"]]
            summary[cfg]["acc_when_retrieved"] = sum(x["correct"] for x in hit) / max(1, len(hit))
            summary[cfg]["retrieved"] = len(hit) / len(ans)
        (out / f"answers_{cfg.replace(':', 'k')}.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in rows))
        print(f"{cfg:18s} acc {acc:.3f}  false-IDK {summary[cfg]['false_abstain']:.2f}  "
              f"IDK-on-unanswerable {summary[cfg]['abstain_unanswerable']:.2f}  TTFT {summary[cfg]['ttft_ms']:.0f} ms  total {summary[cfg]['total_ms']:.0f} ms  "
              f"prompt {summary[cfg]['prompt_tokens']:.0f} tok", flush=True)
        (out / "summary.json").write_text(json.dumps(summary, indent=1))

    # ---- 3. latency vs. k
    if not a.skip_latency:
        lat = {}
        for k in (0, 1, 2, 4, 6):
            rows = []
            for q in LATENCY_QS:
                kw = dict(rag=k > 0, k=k, options={"num_predict": 32})
                cold = answer(ol, kb, q, prefix=f"(request {uuid.uuid4().hex[:8]})\n", **kw)["stats"]
                answer(ol, kb, q, **kw)
                warm = answer(ol, kb, q, **kw)["stats"]          # same request again: prompt prefix cached
                rows.append({"cold_ttft_ms": cold["ttft_ms"], "warm_ttft_ms": warm["ttft_ms"],
                             "retrieval_ms": cold["embed_ms"] + cold["search_ms"], "prefill_ms": cold["prefill_ms"],
                             "prompt_tokens": cold["prompt_tokens"], "computed_tokens": cold["computed_tokens"],
                             "warm_cached": warm["cached_tokens"]})
            lat[k] = {key: round(statistics.mean(r[key] for r in rows), 1) for key in rows[0]}
            print("latency k =", k, lat[k], flush=True)
        (out / "latency.json").write_text(json.dumps(lat, indent=1))


LATENCY_QS = ["Who founded the AI company DeepSeek?", "When did Meta release Llama 3?",
              "How many transistors does the Apple M4 contain?", "Which team won UEFA Euro 2024?",
              "Which sport made its Olympic debut at Paris 2024?", "Near which town was the longest totality of the April 8, 2024 eclipse observed?"]


if __name__ == "__main__":
    main()
