"""Retrain one completed trial of an HPO study and compare its validation rel-L2 with the recorded value.

Uses the stored hyperparameters and trial number (which seeds the mini-batch order); Optuna reads a copy of the DB.
Usage: uv run tools/replay_trial.py --db STUDY.db --arch keronet --pde burgers --max_params 200000 --trial 17 [--sigma 0.5]
"""
import argparse
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

p = argparse.ArgumentParser()
p.add_argument("--db", required=True)
p.add_argument("--arch", required=True, choices=["keronet", "mkeronet", "pi_deeponet", "pit", "pinto"])
p.add_argument("--pde", required=True)
p.add_argument("--max_params", type=int, required=True)
p.add_argument("--trial", type=int, required=True)
p.add_argument("--sigma", type=float, default=None)
a = p.parse_args()

import optuna  # noqa: E402

from architectures import KerONet, MKerONet, PIDeepONet, PINTO, PIT  # noqa: E402
from hpo.tune_common import SearchConfig, make_objective  # noqa: E402
from pdes import get_pde  # noqa: E402

ARCH = {"keronet": KerONet, "mkeronet": MKerONet, "pi_deeponet": PIDeepONet, "pit": PIT, "pinto": PINTO}
optuna.logging.set_verbosity(optuna.logging.WARNING)

with tempfile.TemporaryDirectory() as tmp:
    snap = os.path.join(tmp, "study.db")
    shutil.copy2(a.db, snap)
    storage = f"sqlite:///{snap}"
    name = optuna.get_all_study_summaries(storage=storage)[0].study_name
    t = next(tr for tr in optuna.load_study(study_name=name, storage=storage).trials if tr.number == a.trial)
assert t.state == optuna.trial.TrialState.COMPLETE, t.state
dists = t.distributions


def sample_hparams(trial):
    return {k: trial.suggest_categorical(k, d.choices) for k, d in dists.items()}


spec = get_pde(a.pde)
fixed = {"n_sensors": spec.default_training_kwargs.get("m", 80), "weight_decay": 1e-4}
if a.sigma is not None:
    fixed["sigma"] = a.sigma
search = SearchConfig(arch_name=a.arch, arch_class=ARCH[a.arch], sample_hparams=sample_hparams,
                      fixed_hparams=fixed, pde_spec=spec, pde_kwargs={})
obj = make_objective(search, n_iter=20000, max_params=a.max_params, N_test=100, eval_seed=1234567,
                     train_seed=0, intermediate_iter=5000, predict_chunk=4096, bcs_batch=14,
                     res_batch=14, init_lr=1e-3, end_lr_frac=0.01, warmup_frac=0.0)
value = obj(optuna.trial.FixedTrial(dict(t.params), number=t.number))
print(json.dumps({"study": name, "trial": t.number, "recorded": t.value, "replayed": value,
                  "rel_diff": abs(value - t.value) / t.value}))
