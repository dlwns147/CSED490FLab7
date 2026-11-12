"""Recompute RAG metrics from the saved answers (same scoring code as rag_eval.py) and print a table.

    python analyze.py [results_dir]
"""
import json
import pathlib
import sys

from rag_eval import abstained, correct

R = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else pathlib.Path(__file__).with_name("results"))
Q = {q["id"]: q for q in map(json.loads, open(pathlib.Path(__file__).with_name("questions.jsonl")))}

rows = []
for f in sorted(R.glob("answers_*.jsonl")):
    cfg = f.stem.removeprefix("answers_")
    xs = [json.loads(l) for l in open(f)]
    ans = [x for x in xs if not Q[x["id"]].get("unanswerable")]
    una = [x for x in xs if Q[x["id"]].get("unanswerable")]
    acc = sum(correct(x["answer"], Q[x["id"]]["keys"]) for x in ans) / len(ans)
    fa = sum(abstained(x["answer"]) for x in ans) / len(ans)
    ab = sum(abstained(x["answer"]) for x in una) / max(1, len(una))
    tok = sum(x["prompt_tokens"] or 0 for x in xs) / len(xs)                     # whole prompt
    ttft = sum(x["ttft_ms"] or 0 for x in xs) / len(xs)
    pre_rate = sum((x["prompt_tokens"] or 0) - (x["cached_tokens"] or 0) for x in xs) / max(1e-9, sum(x["prefill_ms"] or 0 for x in xs)) * 1e3
    cached = sum(x["cached_tokens"] or 0 for x in xs) / max(1, sum(x["prompt_tokens"] or 0 for x in xs))
    tot = sum(x["total_ms"] or 0 for x in xs) / len(xs)
    rows.append(dict(cfg=cfg, acc=acc, false_idk=fa, idk_unans=ab if una else None, prompt_tok=tok, ttft_ms=ttft,
                     total_ms=tot, prefill_tok_s=pre_rate, cached_frac=cached, n=len(ans), n_unans=len(una)))

print(f"{'config':24s} {'acc':>6s} {'falseIDK':>8s} {'IDK|unans':>9s} {'prompt':>7s} {'TTFT':>7s} {'total':>7s} {'pp tok/s':>8s} {'cached':>6s}")
for r in rows:
    u = f"{r['idk_unans']:.2f}" if r["idk_unans"] is not None else "-"
    print(f"{r['cfg']:24s} {r['acc']:6.3f} {r['false_idk']:8.2f} {u:>9s} {r['prompt_tok']:7.0f} {r['ttft_ms']:7.0f} "
          f"{r['total_ms']:7.0f} {r['prefill_tok_s']:8.1f} {r['cached_frac']:6.2f}")
(R / "table.json").write_text(json.dumps(rows, indent=1))
