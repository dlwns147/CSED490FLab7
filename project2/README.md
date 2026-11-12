# Project 2 — RAG with Llama 3.2 1B on Raspberry Pi 5

```
question ──▶ retrieve top-k chunks ──▶ prompt = passages + question ──▶ Llama 3.2 1B (Ollama) ──▶ answer + sources
             (nomic-embed-text + BM25, fused with RRF)
```

Everything runs on the Pi's CPU. Only `requests` and `numpy` are needed (no LangChain).

| File | What |
|---|---|
| `rag.py` | knowledge base: chunking, embeddings (`nomic-embed-text`) + BM25 index, hybrid retrieval (RRF), prompt building, URL / Wikipedia ingest |
| `rag_cli.py` | command-line program: build the index, add a URL, search, ask, interactive chat |
| `ollama_client.py` | small client for the Ollama REST API |
| `kb/*.txt` | knowledge base: 8 Wikipedia articles on 2024–2025 topics (after the model's training cutoff), CC BY-SA 4.0 — see `kb/README.md` |
| `eval/questions.jsonl` | 36 questions about those articles (answer keys) and 6 questions the knowledge base cannot answer |
| `eval/rag_eval.py`, `eval/analyze.py` | retrieval recall, answer accuracy, latency vs. k |
| `eval/results/` | the results used in the report |
| `tools/fetch_wiki.py` | downloads the Wikipedia snapshots into `kb/` |
| `tools/figures.py`, `tools/render_terminal.py` | report figures |
| `pi_setup/llama3.2-1b.Modelfile` | Ollama model definition for the Llama 3.2 1B GGUF |

## Setup

With Ollama installed and running on the Pi:

```bash
wget -P pi_setup https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf
ollama create llama3.2-1b -f pi_setup/llama3.2-1b.Modelfile
ollama pull nomic-embed-text
```

## Use

```bash
python3 rag_cli.py index                                   # index kb/*.txt (290 chunks, ~70 s)
python3 rag_cli.py add-url https://en.wikipedia.org/wiki/Raspberry_Pi   # add an external source
python3 rag_cli.py ask "Which team won UEFA Euro 2024?"    # answer + sources + timing
python3 rag_cli.py ask "Which team won UEFA Euro 2024?" --no-rag         # model alone, for comparison
python3 rag_cli.py chat                                    # interactive (/rag on|off, /k N, /quit)
python3 eval/rag_eval.py && python3 eval/analyze.py        # evaluation (writes eval/results/)
```

## Results (36 questions, temperature 0, Raspberry Pi 5 CPU)

| Setting | Accuracy | Time to first token |
|---|---|---|
| Model only | 0.0% | 0.3 s |
| RAG, k = 1 | 88.9% | 2.2 s |
| RAG, k = 2 | 91.7% | 3.8 s |
| RAG, k = 4 (default) | 94.4% | 6.9 s |
| RAG, k = 6 | 91.7% | 10.5 s |
