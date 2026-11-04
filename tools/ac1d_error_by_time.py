"""Per-architecture 1D Allen-Cahn test rel-L2 of the snapshot s(., t) at selected times, mean +- std over seeds.

Reads evaluate.py output directories incl. predictions.npy (not shipped); per-IC errors are checked against experiments/eval/.
Usage: uv run tools/ac1d_error_by_time.py --results keronet=DIR pi_deeponet=DIR pinto=DIR pit=DIR
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
PDE = "allen_cahn"

ap = argparse.ArgumentParser()
ap.add_argument("--results", nargs="+", required=True, metavar="ARCH=DIR",
                help="evaluate.py output directory of each architecture")
ap.add_argument("--times", nargs="+", type=float, default=[0.0, 0.1, 0.25, 0.5, 0.75, 1.0])
ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
a = ap.parse_args()
dirs = {k: Path(v) for k, v in (x.split("=", 1) for x in a.results)}

from pdes import get_pde  # noqa: E402

tk = get_pde(PDE).default_test_kwargs
NN, NT = int(tk["nn"]), int(tk["steps"]) + 1  # flatten_test_trajectory stores (t, x) row-major
gt = None
print("rel-L2 (%) of the snapshot at time t: mean over test inputs per seed, then mean +- std over seeds")
print("arch          " + "".join(f"  t={t:<12g}" for t in a.times))
for arch, d in dirs.items():
    meta = json.load(open(d / "evaluation_meta.json"))
    ntest = int(meta["N_test"])
    if NN * NT != int(meta["pts_per_traj"]):
        sys.exit(f"{d}: the spec's test grid does not match evaluation_meta.json")
    g = np.load(d / "ground_truth.npy").reshape(ntest, NT, NN)
    if gt is None:
        gt, gnorm = g, np.linalg.norm(g, axis=2)
    elif not np.array_equal(g, gt):
        sys.exit(f"{d}: a different reference solution from the first directory")
    idx = [int(round(t * (NT - 1))) for t in a.times]
    per_seed = []
    for s in a.seeds:
        cell = d / f"{arch}_seed{s}"
        p = np.load(cell / "predictions.npy", mmap_mode="r").reshape(ntest, NT, NN)
        rel = REPO / f"experiments/eval/main_200k/{PDE}/{arch}/seed{s}/per_ic_errors.npy"
        if rel.exists() and not np.array_equal(np.load(rel), np.load(cell / "per_ic_errors.npy")):
            sys.exit(f"{cell}: per-IC errors differ from {rel}")
        per_seed.append([np.mean(np.linalg.norm(p[:, i] - gt[:, i], axis=1) / gnorm[:, i]) for i in idx])
    r = 100 * np.array(per_seed)
    print(f"{arch:12s}  " + "".join(f"  {m:6.2f} +- {sd:4.2f}" for m, sd in zip(r.mean(0), r.std(0, ddof=1))))
