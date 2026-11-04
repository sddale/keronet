"""Recompute each study's HPO trial count, selected trial, and GPU-hours from the study files.

Count: the first n >= 15 non-pruned trials at which every study of the group has had an unchanged best-so-far
for the last two (mKerONet: alone, from the group's count). Exits 1 on a mismatch with the manifest.
Usage: uv run tools/hpo_compute.py [--cell PREFIX] [--json OUT]
"""
import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime

_REPO = __import__("pathlib").Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from tools._common import REPO, load_manifest  # noqa: E402

MAIN = ("pi_deeponet", "pit", "pinto", "keronet")
FLOOR, PATIENCE, EPS = 15, 2, 1e-8


def load(db):
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    comp = con.execute("select t.number, v.value from trials t join trial_values v using(trial_id) "
                       "where t.state='COMPLETE' order by t.number").fetchall()
    times = con.execute("select number, datetime_start, datetime_complete from trials order by number").fetchall()
    params = {}
    for num, name, val in con.execute("select t.number, p.param_name, p.param_value from trial_params p "
                                      "join trials t using(trial_id)"):
        params.setdefault(num, {})[name] = val
    con.close()
    return comp, times, params


def at(comp, n):
    """(best trial, trials since best) over the first n completed trials."""
    best, bn, since = None, None, 0
    for num, v in comp[:n]:
        if best is None or v < best - EPS:
            best, bn, since = v, num, 0
        else:
            since += 1
    return bn, since


def rule(studies, floor):
    """First n >= floor at which every study's best is at least PATIENCE completed trials old, or None."""
    for n in range(floor, min(len(s[0]) for s in studies) + 1):
        if all(at(s[0], n)[1] >= PATIENCE for s in studies):
            return n
    return None


def gpu_hours(comp, times, n, correction_s):
    last = comp[n - 1][0]
    s = sum((datetime.fromisoformat(b) - datetime.fromisoformat(a)).total_seconds()
            for num, a, b in times if num <= last and a and b)
    return (s + correction_s) / 3600


p = argparse.ArgumentParser()
p.add_argument("--cell", default="", help="only cells starting with this prefix")
p.add_argument("--json", help="also write the results to this file")
a = p.parse_args()

cells = defaultdict(dict)
for e in load_manifest():
    if e.get("hpo_dir") and e["status"] in ("reported", "supplementary"):
        cells[e["cell"]][e["arch"]] = (e, load(REPO / e["hpo_dir"] / "study.db"))

nstar = {}
for cell, st in cells.items():
    group = [d for arch, (e, d) in st.items() if arch in MAIN and e["status"] == "reported"]
    if len(group) > 1:
        nstar[cell] = rule(group, FLOOR)

out, bad, main_sum = {}, [], defaultdict(float)
for cell, st in cells.items():
    if not cell.startswith(a.cell):
        continue
    print(f"\n## {cell}" + (f"  (group rule holds at {nstar[cell]})" if cell in nstar else ""))
    print("| arch | trials | selected | evaluated | HPO GPU-h |\n|---|---|---|---|---|")
    for arch, (e, (comp, times, params)) in st.items():
        if cell in nstar and arch in MAIN:
            n = nstar[cell]
        else:
            floor = nstar.get(cell, FLOOR)
            n = rule([(comp,)], floor or FLOOR)
        if n is None:
            bad.append(f"{cell}/{arch}: the rule does not hold within the study's {len(comp)} completed trials")
            continue
        sel = at(comp, n)[0]
        ev = e.get("evaluated_trial")
        same = ev is not None and (ev == sel or params.get(ev) == params.get(sel))
        gh = gpu_hours(comp, times, n, e.get("gpu_seconds_correction", 0.0))
        if n != e.get("rule_completed", e["completed_trials"]):
            bad.append(f"{cell}/{arch}: rule holds at {n}, manifest says {e.get('rule_completed', e['completed_trials'])}")
        if not same:
            bad.append(f"{cell}/{arch}: selection at {n} is #{sel}, evaluated #{ev}")
        note = "" if ev == sel else f" (same hyperparameters as #{sel})" if same else " MISMATCH"
        print(f"| {arch} | {n} | #{sel} | #{ev}{note} | {gh:.2f} |")
        out.setdefault(cell, {})[arch] = {"trials": n, "selected": sel, "evaluated": ev, "gpu_h": gh}
        if cell.startswith("main_200k/") and arch in MAIN:
            main_sum[arch] += gh
if main_sum:
    print("\n## main_200k HPO GPU-hours per architecture (four 1D PDEs)")
    for arch in MAIN:
        print(f"| {arch} | {main_sum[arch]:.2f} |")
    print(f"| total | {sum(main_sum.values()):.2f} |")
    out["main_200k_total"] = dict(main_sum)
if a.json:
    json.dump(out, open(a.json, "w"), indent=1)
if bad:
    print("\nMISMATCHES:\n  " + "\n  ".join(bad))
    sys.exit(1)
print("\nevery study's trial count and selection agree with the manifest")
