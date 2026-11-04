"""Extend groups of HPO studies until the convergence rule holds, appending trials to their study DBs.

Rule: at least --base non-pruned trials, and every study of the group with a best-so-far unchanged for its last two.
groups.json: {"groups": {"<name>": {"pde": "burgers", "max_params": 300000, "min_completed": 15,
  "studies": {"<arch>": {"db": "path/to/study.db", "space": "300k", "sigma": null}}}}}
Usage: uv run tools/extend_groups.py groups.json [--groups a,b] [--gpus 0 1] [--dry-run]
"""
import argparse
import json
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools._common import REPO, convergence  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("groups_file")
p.add_argument("--groups", default=None, help="comma-separated subset of group names")
p.add_argument("--gpus", type=int, nargs="+", default=[0])
p.add_argument("--base", type=int, default=15)
p.add_argument("--work_dir", default="extension_runs")
p.add_argument("--dry-run", action="store_true")
a = p.parse_args()

spec_all = json.load(open(a.groups_file))["groups"]
names = a.groups.split(",") if a.groups else list(spec_all)
work = Path(a.work_dir)
work.mkdir(parents=True, exist_ok=True)
ledger = open(work / "ledger.jsonl", "a")


def log(rec):
    rec["time"] = time.strftime("%F %T")
    ledger.write(json.dumps(rec) + "\n")
    ledger.flush()
    print(json.dumps(rec), flush=True)


def study_name(db):
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    (n,), = con.execute("select study_name from studies").fetchall()
    con.close()
    return n


def plan(g):
    spec = spec_all[g]
    st = {arch: convergence(s["db"]) for arch, s in spec["studies"].items()}
    ns = [n for n, _, _ in st.values()]
    level = max(max(ns), a.base, spec.get("min_completed", a.base))
    log({"event": "check", "group": g, "studies": {k: {"n": n, "best": b, "since": s} for k, (n, b, s) in st.items()}})
    if min(ns) < level:
        targets = {arch: level for arch, (n, _, _) in st.items() if n < level}
    else:
        k = max(max(0, 2 - s) for _, _, s in st.values())
        if k == 0:
            log({"event": "converged", "group": g, "completed": level})
            return []
        targets = {arch: level + k for arch in st}
    log({"event": "round", "group": g, "targets": targets})
    return [dict(group=g, arch=arch, target=t, **spec["studies"][arch], pde=spec["pde"],
                 cap=spec["max_params"]) for arch, t in targets.items()]


def launch(gpu, j):
    cmd = [sys.executable, "-u", str(REPO / "hpo" / f"tune_{j['arch']}.py"), "--pde", j["pde"],
           "--max_params", str(j["cap"]), "--space", j["space"], "--storage", f"sqlite:///{j['db']}",
           "--study_name", study_name(j["db"]), "--target_completed", str(j["target"]), "--n_trials", "100",
           "--config_dir", str(work / "configs" / j["group"]), "--out_dir", str(work / "trials" / j["group"])]
    if j.get("sigma") is not None:
        cmd += ["--sigma", str(j["sigma"])]
    logf = open(work / f"{j['group']}__{j['arch']}__{j['target']}.log", "w")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=str(REPO))
    log({"event": "launch", "gpu": gpu, "group": j["group"], "arch": j["arch"], "target": j["target"]})
    return subprocess.Popen(cmd, cwd=REPO, env=env, stdout=logf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)


queue = [j for g in names for j in plan(g)]
if a.dry_run:
    sys.exit(0)
running, inflight, done, retries = {}, {g: 0 for g in names}, set(), {}
while queue or running:
    for gpu in a.gpus:
        if gpu not in running and queue:
            j = queue.pop(0)
            running[gpu] = (launch(gpu, j), j)
            inflight[j["group"]] += 1
    time.sleep(30)
    for gpu, (proc, j) in list(running.items()):
        if proc.poll() is None:
            continue
        del running[gpu]
        n, _, _ = convergence(j["db"])
        log({"event": "finish", "group": j["group"], "arch": j["arch"], "rc": proc.returncode, "n": n})
        g = j["group"]
        if n < j["target"]:
            key = (g, j["arch"], j["target"])
            retries[key] = retries.get(key, 0) + 1
            if retries[key] <= 2:
                queue.append(j)
                inflight[g] -= 1  # the relaunch increments it again
                continue
            log({"event": "abort", "group": g, "arch": j["arch"]})
            done.add(g)
            queue = [q for q in queue if q["group"] != g]
        inflight[g] -= 1
        if inflight[g] == 0 and g not in done and not any(q["group"] == g for q in queue):
            queue += plan(g)
log({"event": "all_done"})
