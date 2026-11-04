"""KerONet's test error versus slot count S on Burgers and 1D Allen-Cahn.

Reads experiments/eval/main_200k/<pde>/keronet and s_sweep/<pde>/keronet_S* (differing only in
S, hparam "L"); points are mean +- std (ddof=1) over five seeds, hollow above the 200k cap.
Writes s_sweep.pdf to --out.
Usage (from the repository root): uv run plotting/plot_s_sweep.py --out figures
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "plotting"))
import matplotlib.pyplot as plt  # noqa: E402
from paper_style import ARCH_STYLE, PDE_DISPLAY, apply_paper_style, fig_size, save_fig  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, default=Path("figures"))
a = ap.parse_args()
CAP = 200_000
E = REPO / "experiments/eval"


def load(d):
    metas = [json.load(open(d / f"seed{s}/meta.json")) for s in range(5)]
    hp = yaml.safe_load(open(d / "seed0/config_used.yaml"))["hparams"]
    return np.array([m["rel_l2_mean"] for m in metas]) * 100, metas[0]["param_count"], hp


series = {}
for pde in ("burgers", "allen_cahn"):
    v, n, hp_sel = load(E / f"main_200k/{pde}/keronet")
    pts = {hp_sel["L"]: (v, n, True)}
    for d in sorted((E / f"s_sweep/{pde}").glob("keronet_S*")):
        v, n, hp = load(d)
        diff = {k for k in set(hp) | set(hp_sel) if hp.get(k) != hp_sel.get(k)}
        if diff != {"L"}:
            sys.exit(f"{d}: differs from the selected configuration in {sorted(diff)}")
        pts[hp["L"]] = (v, n, False)
    series[pde] = dict(sorted(pts.items()))

apply_paper_style()
fig, axes = plt.subplots(1, 2, figsize=fig_size("col", aspect=0.34))
c = ARCH_STYLE["keronet"]["color"]
for ax, (pde, pts) in zip(axes, series.items()):
    S = np.array(list(pts))
    mean = np.array([p[0].mean() for p in pts.values()])
    std = np.array([p[0].std(ddof=1) for p in pts.values()])
    ax.plot(S, mean, color=c, lw=1.1, zorder=2)
    for s, (v, n, sel) in pts.items():
        over = n > CAP
        ax.errorbar([s], [v.mean()], yerr=[v.std(ddof=1)], fmt="*" if sel else "o", ms=7.5 if sel else 4.2, color=c,
                    mfc="white" if over else c, mew=1.0, capsize=2, elinewidth=0.8, zorder=3)
        ax.scatter(np.full(5, s) * np.exp(np.linspace(-0.04, 0.04, 5)), v, s=4, color="0.6", lw=0, zorder=1)
    ax.set_xscale("log", base=2)
    ax.set_xticks(S)
    ax.set_xticklabels([f"{s}\n{pts[s][1] / 1000:.0f}k" for s in S])
    ax.minorticks_off()
    ax.set_xlabel("Slot count $S$ (parameters)")
    ax.set_title(PDE_DISPLAY[pde])
    ax.grid(True, axis="y", alpha=0.3)
    lo, hi = (mean - std).min(), (mean + std).max()
    pad = 0.25 * (hi - lo) if pde == "burgers" else 0.6 * (hi - lo)
    ax.set_ylim(max(0, lo - pad), hi + pad)
axes[0].set_ylabel("Test rel-$L^2$ (%)")
handles = [plt.Line2D([], [], marker="*", ls="", color=c, mfc=c, ms=7.5, label="HPO selection"),
           plt.Line2D([], [], marker="o", ls="", color=c, mfc=c, ms=4.2, label="within the 200k cap"),
           plt.Line2D([], [], marker="o", ls="", color=c, mfc="white", ms=4.2, label="above the cap"),
           plt.Line2D([], [], marker=".", ls="", color="0.6", ms=4, label="individual seeds")]
axes[0].legend(handles=handles, loc="upper right", fontsize=6.5)
fig.tight_layout(w_pad=1.5)
print("wrote", save_fig(fig, a.out / "s_sweep.pdf"))
for pde, pts in series.items():
    for s, (v, n, sel) in pts.items():
        print(f"  {pde:10s} S={s:2d}: {v.mean():.2f} +- {v.std(ddof=1):.2f}%  {n:,} params{'  (selected)' if sel else ''}")
