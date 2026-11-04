"""Allen-Cahn test error versus time, 1D (left) and 2D (right).

Per time: snapshot rel-L2 averaged over test inputs within each seed, then over seeds. Reads
predictions from each ARCH=DIR evaluate.py output (checked against experiments/eval/ where
released); writes the figure to --out and the curves (mean, std over seeds) as .json beside it.
Usage (from the repository root): uv run plotting/plot_ac_error_by_time.py --ac1d keronet=DIR pinto=DIR --ac2d keronet=DIR pinto=DIR --out figures/ac_error_by_time.pdf
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "plotting"))

ap = argparse.ArgumentParser()
ap.add_argument("--ac1d", nargs="+", required=True, metavar="ARCH=DIR")
ap.add_argument("--ac2d", nargs="+", required=True, metavar="ARCH=DIR")
ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
ap.add_argument("--out", required=True)
a = ap.parse_args()

from pdes import get_pde  # noqa: E402

PANELS = [("allen_cahn", "main_200k", 1, a.ac1d), ("allen_cahn_2d", "2d_200k", 2, a.ac2d)]


def curves(pde, cell, dim, specs):
    tk = get_pde(pde).default_test_kwargs
    nn, nt = int(tk["nn"]), int(tk["steps"]) + 1
    nx = nn ** dim
    out, gt = {}, None
    for arch, d in (x.split("=", 1) for x in specs):
        d = Path(d)
        meta = json.load(open(d / "evaluation_meta.json"))
        ntest = int(meta["N_test"])
        if nx * nt != int(meta["pts_per_traj"]):
            sys.exit(f"{d}: the spec's test grid does not match evaluation_meta.json")
        g = np.load(d / "ground_truth.npy", mmap_mode="r").reshape(ntest, nt, nx)
        if gt is None:
            gt = np.asarray(g, dtype=np.float64)
            gnorm = np.linalg.norm(gt, axis=2)
        elif not np.array_equal(np.asarray(g, dtype=np.float64), gt):
            sys.exit(f"{d}: a different reference solution from the first directory")
        per_seed = []
        for s in a.seeds:
            run = d / f"{arch}_seed{s}"
            rel = REPO / f"experiments/eval/{cell}/{pde}/{arch}/seed{s}/per_ic_errors.npy"
            if rel.exists() and not np.array_equal(np.load(rel), np.load(run / "per_ic_errors.npy")):
                sys.exit(f"{run}: per-IC errors differ from {rel}")
            p = np.load(run / "predictions.npy", mmap_mode="r").reshape(ntest, nt, nx)
            per_seed.append([np.mean(np.linalg.norm(p[:, i] - gt[:, i], axis=1) / gnorm[:, i])
                             for i in range(nt)])
        r = 100 * np.array(per_seed)
        out[arch] = {"t": np.linspace(0, 1, nt).tolist(), "mean": r.mean(0).tolist(),
                     "std": r.std(0, ddof=1).tolist()}
    return out


data = {pde: curves(pde, cell, dim, specs) for pde, cell, dim, specs in PANELS}

import matplotlib.pyplot as plt  # noqa: E402
from paper_style import ARCH_DISPLAY, ARCH_STYLE, apply_paper_style, fig_size, save_fig  # noqa: E402

apply_paper_style()
fig, axes = plt.subplots(1, 2, figsize=fig_size("col", aspect=0.2))
for ax, (pde, title) in zip(axes, [("allen_cahn", "1D Allen-Cahn"), ("allen_cahn_2d", "2D Allen-Cahn")]):
    for arch, c in data[pde].items():
        st = ARCH_STYLE[arch]
        # KerONet and PINTO have adjacent palette colors; separate them by line style as well.
        ls = {"keronet": "-", "pinto": "--"}.get(arch, st["ls"])
        ax.plot(c["t"], c["mean"], color=st["color"], ls=ls, lw=1.2, label=ARCH_DISPLAY[arch])
    ax.set_title(title, pad=2)
    ax.set_xlabel("$t$", labelpad=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
axes[0].set_ylabel(r"rel-$L^2$ (\%)" if plt.rcParams.get("text.usetex") else "rel-$L^2$ (%)")
axes[1].legend(loc="upper left", ncol=2, frameon=False, handlelength=1.6)
out = save_fig(fig, a.out)
json.dump(data, open(Path(a.out).with_suffix(".json"), "w"))
print(f"wrote {out}")
for pde in data:
    for arch, c in data[pde].items():
        print(f"{pde:16s} {arch:8s} t=0 {c['mean'][0]:5.2f}  t=0.5 {c['mean'][50]:5.2f}  t=1 {c['mean'][-1]:5.2f}")
