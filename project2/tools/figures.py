"""Report figure: RAG accuracy and time to first token vs. number of retrieved passages k."""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent.parent
R = ROOT / "eval/results"
OUT = pathlib.Path(__import__("sys").argv[1]) if len(__import__("sys").argv) > 1 else ROOT / "figures"
OUT.mkdir(exist_ok=True)
C1, C2, INK, INK2, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e6e5e0"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True})


def label_end(ax, x, y, text, dy=0):
    ax.annotate(text, (x, y), xytext=(6, dy), textcoords="offset points", color=INK, va="center", fontsize=8.5)


tab = {r["cfg"]: r for r in json.load(open(R / "table.json"))}
lat = json.load(open(R / "latency.json"))
ks = [1, 2, 4, 6]
fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.3), constrained_layout=True)
ax = axs[0]
for name, c, lab, dy in (("plain", C1, "plain prompt", 7), ("strict", C2, "strict prompt\n(\"say I don't know\")", -12)):
    y = [100 * tab[f"hybridk{k}k{name}"]["acc"] for k in ks]
    ax.plot(ks, y, color=c, lw=2, marker="o", ms=4)
    label_end(ax, ks[-1], y[-1], lab, dy)
ax.axhline(100 * tab["closed"]["acc"], color=INK2, lw=1, ls="--")
ax.annotate("no retrieval", (1, 100 * tab["closed"]["acc"]), xytext=(0, 4), textcoords="offset points", fontsize=8.5, color=INK2)
ax.set_xticks(ks); ax.set_xlim(0.6, 9.5); ax.set_ylim(0, 100)
ax.set_xlabel("passages retrieved (k)"); ax.set_ylabel("answer accuracy (%)")
ax = axs[1]
kk = [0, 1, 2, 4, 6]
y = [lat[str(k)]["cold_ttft_ms"] / 1000 for k in kk]
w = [lat[str(k)]["warm_ttft_ms"] / 1000 for k in kk]
ax.plot(kk, y, color=C1, lw=2, marker="o", ms=4)
ax.plot(kk, w, color=C2, lw=2, marker="o", ms=4)
label_end(ax, kk[-1], y[-1], "new passages")
label_end(ax, kk[-1], w[-1], "same passages\nagain (cached)", dy=6)
for k, v in zip(kk, y):
    if k in (4, 6):
        ax.annotate(f"{v:.1f} s", (k, v), xytext=(-4, 7), textcoords="offset points", ha="right", fontsize=8, color=INK)
ax.set_xticks(kk); ax.set_xlim(-0.3, 9.3); ax.set_ylim(0, max(y) * 1.15)
ax.set_xlabel("passages retrieved (k)"); ax.set_ylabel("time to first token (s)")
fig.savefig(OUT / "fig_rag_k.png", dpi=220)
print(OUT / "fig_rag_k.png")
