"""Small helpers shared by the tools/ scripts."""
import json
import sqlite3
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ARCHS = ("pi_deeponet", "pit", "pinto", "keronet", "mkeronet")


def base_arch(name):
    """Map a manifest arch label (e.g. 'keronet_S4') to its base architecture."""
    for a in sorted(ARCHS, key=len, reverse=True):
        if name == a or name.startswith(a + "_"):
            return a
    raise ValueError(f"unknown architecture label {name!r}")


def load_manifest():
    return json.load(open(REPO / "experiments" / "manifest.json"))["entries"]


def db_space(path):
    """Ordered [(name, choices)] stored in a study DB (first appearance by trial, param id)."""
    import optuna

    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    rows = con.execute("select param_name, distribution_json from trial_params "
                       "order by trial_id, param_id").fetchall()
    con.close()
    seen, out = set(), []
    for name, dj in rows:
        if name in seen:
            continue
        seen.add(name)
        d = optuna.distributions.json_to_distribution(dj)
        out.append((name, list(d.choices)))
    return out


def convergence(path, eps=1e-8):
    """(completed, best_trial_number, trials_since_best) for a study DB (read-only)."""
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    rows = con.execute("select t.number, v.value from trials t join trial_values v using(trial_id) "
                       "where t.state='COMPLETE' order by t.number").fetchall()
    con.close()
    best, bn, since = None, None, 0
    for n, v in rows:
        if best is None or v < best - eps:
            best, bn, since = v, n, 0
        else:
            since += 1
    return len(rows), bn, since
