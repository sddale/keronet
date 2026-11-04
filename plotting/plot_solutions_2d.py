"""2D Allen-Cahn solution panels for one test input function.

Rows: reference, then each architecture's --seed run; columns: s and absolute error at t = 0, 1/2, 1.
Labels give rel-L2 on that input, mean +- std (ddof=1) over seeds 0-4. The input is at --rank
(0 = lowest) in KerONet's per-IC errors on --seed. Writes solutions_allen_cahn_2d_seed<S>_rank<R>_ic<I>.pdf.
Usage (from the repository root): uv run plotting/plot_solutions_2d.py --seed 0 --rank 50 --out figures --results keronet=DIR pi_deeponet=DIR pinto=DIR pit=DIR
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.colors import LogNorm, Normalize

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
ARCHS = ("keronet", "pi_deeponet", "pinto", "pit")  # the 1D figures' order
NAMES = {"keronet": "KerONet", "pi_deeponet": "PI-DeepONet", "pinto": "PINTO", "pit": "PIT"}
PDE = "allen_cahn_2d"

mpl.rcParams.update({  # as plotting/make_pde_figures.py
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.linewidth": 0.6, "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2.5,
    "ytick.major.size": 2.5, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42, "ps.fonttype": 42})

ap = argparse.ArgumentParser()
ap.add_argument("--results", nargs=4, required=True, metavar="ARCH=DIR",
                help="evaluate.py output directory of each architecture")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--rank", type=int, default=50, help="KerONet difficulty rank of the IC on --seed (0 = easiest)")
ap.add_argument("--out", type=Path, default=Path("figures"))
a = ap.parse_args()
dirs = dict(x.split("=", 1) for x in a.results)
if set(dirs) != set(ARCHS):
    sys.exit(f"--results needs exactly {ARCHS}")
dirs = {k: Path(v) for k, v in dirs.items()}

from pdes import get_pde  # noqa: E402

tk = get_pde(PDE).default_test_kwargs
NN, NT = int(tk["nn"]), int(tk["steps"]) + 1
meta = json.load(open(dirs["keronet"] / "evaluation_meta.json"))
NTEST = int(meta["N_test"])
if NN * NN * NT != int(meta["pts_per_traj"]):
    sys.exit("the spec's test grid does not match evaluation_meta.json")
TIMES = (0, (NT - 1) // 2, NT - 1)

seed_dir = lambda k, s: dirs[k] / f"{k}_seed{s}"  # noqa: E731
errs = {k: np.stack([np.load(seed_dir(k, s) / "per_ic_errors.npy") for s in range(5)]) for k in ARCHS}
for k in ARCHS:
    rel = REPO / f"experiments/eval/2d_200k/{PDE}/{k}"
    if rel.is_dir() and not all(np.array_equal(np.load(rel / f"seed{s}/per_ic_errors.npy"), errs[k][s]) for s in range(5)):
        sys.exit(f"{k}: per-IC errors differ from {rel}")
ic = int(np.argsort(errs["keronet"][a.seed])[a.rank])
gts = [np.load(dirs[k] / "ground_truth.npy", mmap_mode="r") for k in ARCHS]
if not all(np.array_equal(gts[0][:NN * NN * NT * NTEST], g[:NN * NN * NT * NTEST]) for g in gts[1:]):
    sys.exit("the output directories hold different reference solutions")
gt = np.asarray(gts[0]).reshape(NTEST, NT, NN, NN)[ic]
pred = {k: np.asarray(np.load(seed_dir(k, a.seed) / "predictions.npy", mmap_mode="r").reshape(NTEST, NT, NN, NN)[ic])
        for k in ARCHS}
for k in ARCHS:  # the predictions reproduce the recorded per-IC error
    e = np.linalg.norm(pred[k] - gt) / np.linalg.norm(gt)
    if abs(e - errs[k][a.seed, ic]) > 1e-4 * max(1.0, e):
        sys.exit(f"{k}: recomputed error {e:.6f} != recorded {errs[k][a.seed, ic]:.6f}")
m = int(yaml.safe_load(open(seed_dir("keronet", a.seed) / "config_used.yaml"))["hparams"]["n_sensors"])

# Shared norms, as in 1D: solution symmetric about 0; error log-scaled from the 1st percentile.
sel = list(TIMES)
sol_max = max(np.abs(gt[sel]).max(), max(np.abs(pred[k][sel]).max() for k in ARCHS))
sol_norm = Normalize(-sol_max, sol_max)
err = {k: np.abs(pred[k][sel] - gt[sel]) for k in ARCHS}
allerr = np.concatenate([e.ravel() for e in err.values()])
lo = max(1e-8, np.percentile(allerr[allerr > 1e-8], 1))
err_norm = LogNorm(lo, max(10 * lo, allerr.max()))

fig = plt.figure(figsize=(5.5, 4.6))
# columns: s at three times | s colorbar | gap | error at three times | error colorbar
gs = fig.add_gridspec(5, 9, width_ratios=[1, 1, 1, 0.07, 0.62, 1, 1, 1, 0.07], wspace=0.06, hspace=0.10,
                      left=0.115, right=0.93, top=0.865, bottom=0.02)
ext = [0, 1, 0, 1]
g = np.linspace(0.0, 1.0, int(round(np.sqrt(m))), endpoint=False)
SX, SY = np.meshgrid(g, g, indexing="ij")
for r, k in enumerate(("reference",) + ARCHS):
    for c, t in enumerate(TIMES):
        ax = fig.add_subplot(gs[r, c])
        field = gt[t] if k == "reference" else pred[k][t]
        im_s = ax.imshow(field.T, origin="lower", extent=ext, norm=sol_norm, cmap="RdBu_r", interpolation="nearest")
        if k == "reference" and t == 0:
            ax.plot(SX.ravel(), SY.ravel(), ".", ms=1.6, color="k", alpha=0.7)
        if r == 0:
            ax.set_title(f"$t = {t / (NT - 1):g}$", pad=2)
        if c == 0:
            label = "Reference" if k == "reference" else (
                f"{NAMES[k]}\n{100 * errs[k][:, ic].mean():.2f}$\\pm${100 * errs[k][:, ic].std(ddof=1):.2f}%")
            ax.set_ylabel(label, labelpad=2, fontsize=8)
        ax.set_xticks([]), ax.set_yticks([])
        if k == "reference":
            continue
        ax = fig.add_subplot(gs[r, c + 5])
        im_e = ax.imshow(err[k][c].T, origin="lower", extent=ext, norm=err_norm, cmap="magma", interpolation="nearest")
        if r == 1:
            ax.set_title(f"$t = {t / (NT - 1):g}$", pad=2)
        ax.set_xticks([]), ax.set_yticks([])
cb = fig.colorbar(im_s, cax=fig.add_subplot(gs[:, 3]))
cb.ax.tick_params(labelsize=6, length=2)
cb = fig.colorbar(im_e, cax=fig.add_subplot(gs[1:, 8]))
cb.set_label("$|s_\\mathrm{pred} - s|$", labelpad=1)
cb.ax.tick_params(labelsize=6, length=2)
note = fig.add_subplot(gs[0, 5:8])
note.axis("off")
side = int(round(np.sqrt(m)))
note.text(0.5, 0.06, "Absolute error $|s_\\mathrm{pred} - s|$", fontsize=9, ha="center", va="bottom", transform=note.transAxes)
fig.text(0.115 + 0.5 * 0.33, 0.905, "Reference and predictions $s(x, y, t)$", fontsize=9, ha="center")
fig.suptitle("2D Allen-Cahn ($\\varepsilon^2 = 10^{-2}$)", y=0.995, fontsize=9)
a.out.mkdir(parents=True, exist_ok=True)
out = a.out / f"solutions_{PDE}_seed{a.seed}_rank{a.rank}_ic{ic}.pdf"
fig.savefig(out)
print("wrote", out)
for k in ARCHS:
    print(f"  {k}: IC #{ic} {100 * errs[k][:, ic].mean():.2f} +- {100 * errs[k][:, ic].std(ddof=1):.2f}% (5 seeds)")
