"""Regenerate the paper's reported numbers from experiments/eval/.

Per (cell, arch): mean +- sample std of the five per-seed rel-L2 errors, parameter count, and mean training time;
per cell: paired bootstrap CIs (plotting.report.bootstrap_diff_ci) against KerONet and mKerONet where present.
Usage: uv run tools/paper_tables.py [--cell PREFIX]
"""
import argparse
import json
import sys
from collections import defaultdict

import numpy as np

_REPO = __import__("pathlib").Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "plotting"))  # report.py imports its sibling paper_style

from plotting.report import bootstrap_diff_ci  # noqa: E402
from tools._common import REPO, load_manifest  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--cell", default="", help="only cells starting with this prefix")
p.add_argument("--include-superseded", action="store_true")
a = p.parse_args()

cells = defaultdict(dict)
for e in load_manifest():
    if not e["cell"].startswith(a.cell):
        continue
    if e["status"] == "superseded" and not a.include_superseded:
        continue
    metas = [json.load(open(REPO / e["eval_dir"] / f"seed{s}" / "meta.json")) for s in range(5)]
    cells[e["cell"]][e["arch"]] = {
        "per_seed": np.array([m["rel_l2_mean"] for m in metas]),
        "params": metas[0]["param_count"],
        "time": float(np.mean([m["train_time_s"] for m in metas])),
        "n_test": sorted({m["n_test_ics"] for m in metas}),
        "status": e["status"],
    }

ORDER = ("pi_deeponet", "pit", "pinto", "keronet", "mkeronet")
for cell, archs in cells.items():
    print(f"\n## {cell}")
    print(f"| arch | rel-L2 (%) | params | train (s) | N_test | status |")
    print("|---|---|---|---|---|---|")
    keys = sorted(archs, key=lambda k: (ORDER.index(k.split("_S")[0]) if k.split("_S")[0] in ORDER else 9, k))
    for k in keys:
        r = archs[k]
        v = r["per_seed"]
        print(f"| {k} | {100 * v.mean():.2f} ± {100 * v.std(ddof=1):.2f} | {r['params']:,} | "
              f"{r['time']:.1f} | {r['n_test']} | {r['status']} |")
    for ref in ("keronet", "mkeronet"):
        if ref not in archs:
            continue
        rv = archs[ref]["per_seed"]
        rows = []
        for k in keys:
            if k == ref or k.startswith("keronet_S"):
                continue
            ci = bootstrap_diff_ci(archs[k]["per_seed"], rv)
            wins = int(np.sum(rv < archs[k]["per_seed"]))
            rows.append(f"| {k} − {ref} | {100 * ci['diff_mean']:+.2f} | "
                        f"[{100 * ci['ci_lo']:+.2f}, {100 * ci['ci_hi']:+.2f}] | {wins}/5 |")
        if rows:
            print(f"\n| comparison | Δ (pp) | 95% CI (pp) | seeds favoring {ref} |\n|---|---|---|---|")
            print("\n".join(rows))
