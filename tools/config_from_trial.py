"""Write an evaluate.py config for a completed trial of an HPO study (the DB is opened read-only).

Sampled hyperparameters and metadata come from the trial; everything else from --template (same arch and PDE).
Usage: uv run tools/config_from_trial.py --db STUDY.db --trial N --template TEMPLATE.yaml --out CFG.yaml
"""
import argparse
import json
import sqlite3

import optuna
import yaml

p = argparse.ArgumentParser()
p.add_argument("--db", required=True)
p.add_argument("--trial", type=int, required=True)
p.add_argument("--template", required=True)
p.add_argument("--out", required=True)
a = p.parse_args()

cfg = yaml.safe_load(open(a.template))
con = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True)
(sid, sname), = con.execute("select study_id, study_name from studies").fetchall()
(tid, value), = con.execute(
    "select t.trial_id, v.value from trials t join trial_values v using(trial_id) "
    "where t.study_id=? and t.number=? and t.state='COMPLETE'", (sid, a.trial)).fetchall()
params = {name: optuna.distributions.json_to_distribution(dj).to_external_repr(val)
          for name, val, dj in con.execute("select param_name, param_value, distribution_json "
                                           "from trial_params where trial_id=?", (tid,))}
attrs = {k: json.loads(v) for k, v in con.execute(
    "select key, value_json from trial_user_attributes where trial_id=?", (tid,))}
n_comp = con.execute("select count(*) from trials where study_id=? and state='COMPLETE'", (sid,)).fetchone()[0]
n_pr = con.execute("select count(*) from trials where study_id=? and state='PRUNED'", (sid,)).fetchone()[0]
con.close()

sampled = set(cfg["hparams"]) - {"n_sensors", "weight_decay", "sigma"}
assert sampled == set(params), f"template/trial parameter mismatch: {sorted(sampled)} vs {sorted(params)}"
cfg["hparams"].update(params)
cfg["metadata"].update({
    "optuna_study": sname, "best_trial_number": a.trial, "best_rel_l2_mean": float(value),
    "best_rel_l2_std": float(attrs.get("rel_l2_std", 0.0)), "param_count": int(attrs.get("param_count", 0)),
    "train_time_sec": float(attrs.get("train_time_sec", 0.0)), "completed_trials": n_comp,
    "pruned_trials": n_pr, "note": f"written by tools/config_from_trial.py (trial {a.trial})"})
yaml.safe_dump(cfg, open(a.out, "w"), sort_keys=False)
print(f"wrote {a.out}")
