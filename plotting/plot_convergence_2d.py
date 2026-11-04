"""2D Allen-Cahn (200k) panel of the HPO convergence figure.

Plots best validation rel-L2 so far vs completed trials from each architecture's study.db in
experiments/manifest.json, up to its rule_completed; writes convergence_allen_cahn_2d.pdf.
Usage (from the repository root): uv run plotting/plot_convergence_2d.py --out figures
"""
import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "plotting"))
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402
from paper_style import ARCH_DISPLAY, ARCH_STYLE, PDE_DISPLAY, apply_paper_style, fig_size, save_fig  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, default=Path("figures"))
a = ap.parse_args()
CELL = "2d_200k/allen_cahn_2d"
man = {(e["cell"], e["arch"]): e for e in json.load(open(REPO / "experiments/manifest.json"))["entries"]}
apply_paper_style()
fig, ax = plt.subplots(figsize=fig_size("col", aspect=0.6))
for arch in ("keronet", "pinto", "pit", "pi_deeponet"):  # the 1D panels' legend order
    e = man[(CELL, arch)]
    con = sqlite3.connect(f"file:{REPO / e['hpo_dir'] / 'study.db'}?mode=ro", uri=True)
    vals = [v for (v,) in con.execute("select v.value from trials t join trial_values v using(trial_id) "
                                      "where t.state='COMPLETE' order by t.number")]
    con.close()
    n = e["rule_completed"]
    best = np.minimum.accumulate(np.array(vals[:n]))
    s = ARCH_STYLE[arch]
    ax.plot(np.arange(1, n + 1), best, label=ARCH_DISPLAY[arch], color=s["color"], linestyle=s["ls"],
            linewidth=1.2, marker=s["marker"], markersize=2.8, markevery=1)
    print(f"  {arch}: {n} trials, best {best[-1]:.5f}")
ax.set_yscale("log")
ax.set_xlabel("Completed trials")
ax.set_ylabel(r"Best validation rel-$L^2$ so far")
ax.set_title(PDE_DISPLAY["allen_cahn_2d"])
ax.grid(True, which="major", alpha=0.3, linewidth=0.4)
ax.grid(True, which="minor", alpha=0.12, linewidth=0.3)
ax.xaxis.set_major_locator(MultipleLocator(2))
ax.set_axisbelow(True)
ax.legend(loc="upper right")
fig.tight_layout()
print("wrote", save_fig(fig, a.out / "convergence_allen_cahn_2d.pdf"))
