"""HPO protocol shared by the hpo/tune_*.py scripts.

run_study runs an Optuna study minimizing mean validation rel-L2; each trial trains for n_iter
iterations and can be pruned at intermediate_iter. Data are cached under DATA_CACHE_DIR.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Any, Type

import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

import jax
import jax.numpy as jnp
from jax import random
from jax.flatten_util import ravel_pytree

from pdes import get_pde, PDESpec
from architectures.model_config import save_config


# Caches  (process-local L1 + disk L2; both reused across trials and studies)


_TEST_CACHE: Dict[Any, Any] = {}
_TRAIN_CACHE: Dict[Any, Any] = {}

DATA_CACHE_DIR = Path(os.environ.get("DATA_CACHE_DIR", "data_cache"))


def _disk_cache_path(kind: str, key_parts: tuple) -> Path:
    """Return DATA_CACHE_DIR/<kind>_<key parts joined by _>.npz, creating the directory."""
    DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    parts_str = "_".join(str(p) for p in key_parts)
    return DATA_CACHE_DIR / f"{kind}_{parts_str}.npz"


def _meta_path(npz_path: Path) -> Path:
    """Sidecar metadata path for an .npz cache file."""
    return npz_path.with_suffix(".meta.json")


def _build_metadata(pde_spec: PDESpec, **extra) -> dict:
    """Return cache metadata: the PDE name and params (numbers as floats) plus the extra fields."""
    import json  # noqa: F401  (json is imported when this is used)

    pde_params = {}
    for k, v in pde_spec.params.items():
        if isinstance(v, (int, float, np.number)):
            pde_params[k] = float(v)
        else:
            pde_params[k] = v
    return {"pde": pde_spec.name, "pde_params": pde_params, **extra}


def _save_arrays_to_disk(path: Path, arrays: tuple, metadata: dict = None) -> None:
    """Save arrays to an .npz as arr0, arr1, ..., and metadata (if given) to a .meta.json sidecar."""
    import json

    np.savez(path, **{f"arr{i}": np.asarray(a) for i, a in enumerate(arrays)})
    if metadata is not None:
        with open(_meta_path(path), "w") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)


def _load_arrays_from_disk(path: Path, n: int, device: bool = True) -> tuple:
    """Load arr0..arr<n-1> from an .npz as jax arrays (device=True) or numpy arrays (device=False)."""
    data = np.load(path)
    conv = jnp.asarray if device else np.asarray
    return tuple(conv(data[f"arr{i}"]) for i in range(n))


def _metadata_matches(npz_path: Path, expected: dict) -> bool:
    """Return True if the .meta.json sidecar of npz_path equals expected or does not exist."""
    import json

    meta = _meta_path(npz_path)
    if not meta.exists():
        return True
    try:
        with open(meta) as f:
            cached = json.load(f)
        return cached == expected
    except Exception:
        return False


def get_test_set(pde_spec: PDESpec, N_test: int, seed: int):
    """Return the evaluation set (u, y, s) of N_test functions drawn with seed, as numpy arrays.

    Cached in memory and in DATA_CACHE_DIR; regenerated if the cached PDE params, N_test, or seed differ.
    """
    cache_key = (pde_spec.name, N_test, seed)
    if cache_key in _TEST_CACHE:
        return _TEST_CACHE[cache_key]

    expected_meta = _build_metadata(pde_spec, N_test=int(N_test), seed=int(seed))
    disk_path = _disk_cache_path("test", cache_key)

    if disk_path.exists() and _metadata_matches(disk_path, expected_meta):
        print(f"  Loading test set from disk cache: {disk_path}")
        t0 = time.time()
        triple = _load_arrays_from_disk(disk_path, n=3, device=False)
        print(f"    Loaded in {time.time() - t0:.1f}s")
        _TEST_CACHE[cache_key] = triple
        return triple
    elif disk_path.exists():
        print(f"  Disk cache exists but metadata mismatch; regenerating: {disk_path}")

    print(f"  Generating test set: pde={pde_spec.name}, N_test={N_test}")
    t0 = time.time()
    kwargs = dict(pde_spec.default_test_kwargs)
    kwargs["N"] = N_test
    u_test, y_test, s_test = (
        pde_spec.generate_test_data.__wrapped__(random.PRNGKey(seed), **kwargs)
        if hasattr(pde_spec.generate_test_data, "__wrapped__")
        else _call_with_overrides(
            pde_spec.generate_test_data, random.PRNGKey(seed), N=N_test
        )
    )
    # triple = (u_test, y_test, s_test)
    triple = tuple(np.asarray(a) for a in (u_test, y_test, s_test))
    print(f"    Test set ready in {time.time() - t0:.1f}s")

    _save_arrays_to_disk(disk_path, triple, metadata=expected_meta)
    print(f"    Cached to disk: {disk_path}")

    _TEST_CACHE[cache_key] = triple
    return triple


def get_training_data(pde_spec: PDESpec, seed: int):
    """Return the training data for seed, cached like get_test_set in DATA_CACHE_DIR/train_<pde>_<seed>.npz."""
    cache_key = (pde_spec.name, seed)
    if cache_key in _TRAIN_CACHE:
        return _TRAIN_CACHE[cache_key]

    expected_meta = _build_metadata(pde_spec, seed=int(seed))
    disk_path = _disk_cache_path("train", cache_key)

    if disk_path.exists() and _metadata_matches(disk_path, expected_meta):
        print(f"  Loading training set from disk cache: {disk_path}")
        t0 = time.time()
        # generate_training_data return shape varies by PDE --- we autodetect
        # the array count by inspecting the .npz on disk.
        with np.load(disk_path) as data:
            n_arrs = len(data.files)
        out = _load_arrays_from_disk(disk_path, n=n_arrs)
        print(f"    Loaded in {time.time() - t0:.1f}s")
        _TRAIN_CACHE[cache_key] = out
        return out
    elif disk_path.exists():
        print(f"  Disk cache exists but metadata mismatch; regenerating: {disk_path}")

    print(f"  Generating training set: pde={pde_spec.name}")
    t0 = time.time()
    out = pde_spec.generate_training_data(random.PRNGKey(seed))
    print(f"    Training set ready in {time.time() - t0:.1f}s")

    _save_arrays_to_disk(disk_path, out, metadata=expected_meta)
    print(f"    Cached to disk: {disk_path}")

    _TRAIN_CACHE[cache_key] = out
    return out


def _call_with_overrides(fn, *args, **overrides):
    """Call fn, overriding some of its bound keyword arguments if it is a functools.partial."""
    # functools.partial keeps the original `func` and `keywords` accessible.
    if hasattr(fn, "func") and hasattr(fn, "keywords"):
        merged = {**fn.keywords, **overrides}
        return fn.func(*args, **merged)
    return fn(*args, **overrides)


# Evaluation  (one-shot rel-L2 across cached test set)


def evaluate_rel_l2(
    model, params, pde_spec: PDESpec, N_test: int, seed: int, predict_chunk: int = 4096
):
    """Return the per-function relative L2 errors of the model on the (N_test, seed) evaluation set."""
    u_te, y_te, s_te = get_test_set(pde_spec, N_test, seed)

    # Number of evaluation points per trajectory comes from the test kwargs
    nn = pde_spec.default_test_kwargs.get("nn", 101)
    steps = pde_spec.default_test_kwargs.get("steps", 100)
    pts_per_traj = pde_spec.default_test_kwargs.get("pts_per_traj", nn * (steps + 1))

    n_total = u_te.shape[0]
    s_pred_chunks = []
    for start in range(0, n_total, predict_chunk):
        end = min(start + predict_chunk, n_total)
        s_pred_chunks.append(
            np.asarray(model.predict_s(params, u_te[start:end], y_te[start:end]))
        )
    s_pred = np.concatenate(s_pred_chunks, axis=0)

    s_pred_arr = s_pred.reshape(N_test, pts_per_traj)
    s_test_arr = np.asarray(s_te).reshape(N_test, pts_per_traj)
    num = np.linalg.norm(s_pred_arr - s_test_arr, axis=1)
    den = np.linalg.norm(s_test_arr, axis=1) + 1e-12
    return num / den


# Search configuration


@dataclass
class SearchConfig:
    """Configuration for one HPO study.

    Each trial builds arch_class(pde_spec=pde_spec, ...) from fixed_hparams updated with sample_hparams(trial).
    """

    arch_name: str
    arch_class: Type
    sample_hparams: Callable[[optuna.Trial], Dict[str, Any]]
    fixed_hparams: Dict[str, Any]
    pde_spec: PDESpec
    pde_kwargs: Dict[str, Any] = field(default_factory=dict)


# Per-trial objective


def make_objective(
    search: SearchConfig,
    n_iter: int,
    max_params: int,
    N_test: int,
    eval_seed: int,
    train_seed: int,
    intermediate_iter: int,
    predict_chunk: int,
    bcs_batch: int,
    res_batch: int,
    init_lr: float,
    end_lr_frac: float,
    warmup_frac: float,
):
    """Return the Optuna objective: train one sampled model and return its mean validation rel-L2.

    Prunes trials that fail to build, exceed max_params, run out of memory, or are stopped by the pruner at intermediate_iter.
    """

    def objective(trial: optuna.Trial) -> float:
        #  Sample hyperparameters
        sampled = search.sample_hparams(trial)
        all_hparams = {**search.fixed_hparams, **sampled}

        #  Build model
        try:
            model = search.arch_class(
                pde_spec=search.pde_spec,
                **all_hparams,
            )
        except Exception as e:
            raise optuna.TrialPruned(f"model init failed: {e}")

        flat, _ = ravel_pytree(model.params)
        param_count = int(len(flat))
        trial.set_user_attr("param_count", param_count)

        if max_params is not None and param_count > max_params:
            raise optuna.TrialPruned(
                f"params {param_count:,} > max_params {max_params:,}"
            )

        #  Datasets
        train_data = get_training_data(search.pde_spec, train_seed)
        bcs_cls = search.pde_spec.default_dataset_classes["bcs"]
        res_cls = search.pde_spec.default_dataset_classes["res"]

        # Training tuple layout depends on the PDE's BCS structure:
        #   periodic_ic        (Burgers, Advection, allen-cahn):
        #     (u, y_ic, s_ic, y_bc_lo, y_bc_hi, y_res, s_res)
        #   dirichlet_zero_ic  (Diffusion-Reaction):
        #     (u, y_ic, s_ic, y_bc, s_bc, y_res, s_res)
        #
        # In both cases, the residual fields are the last two and the BCS
        # fields are the first 5 (just unpacked into their respective
        # dataset class).
        kind = search.pde_spec.bcs_structure.kind
        u, y_res, s_res = train_data[0], train_data[-2], train_data[-1]
        bcs_fields = train_data[:-2]  # u, y_ic, s_ic, *bc_fields

        # BCS and RES share an rng_key so each iteration samples the same
        # 14 input-function indices for both losses (shared-batch
        # convention; matches evaluate.py).
        batch_key = random.PRNGKey(trial.number)
        bcs_ds = bcs_cls(
            *bcs_fields,
            batch_size=bcs_batch,
            rng_key=batch_key,
        )
        res_ds = res_cls(
            u,
            y_res,
            s_res,
            batch_size=res_batch,
            rng_key=batch_key,
        )

        #  Build full-horizon LR schedule once
        from architectures.training_schedules import rebuild_optimizer

        rebuild_optimizer(
            model,
            n_iter,
            init_lr=init_lr,
            end_lr_frac=end_lr_frac,
            warmup_frac=warmup_frac,
            weight_decay=all_hparams.get("weight_decay", 1e-4),
        )

        #  Train with intermediate-value reporting
        # OOM during training/eval is caught and treated as a trial pruning
        # rather than a worker crash. This handles configs that fit within
        # the parameter budget but exceed activation-memory budget under
        # higher-order AD: Burgers requires u_xx (2nd-order), with the latter dominating memory
        # cost on iterated-attention architectures with large
        # n_heads x n_cross_attn.
        t0 = time.time()
        try:
            if intermediate_iter > 0 and intermediate_iter < n_iter:
                model.train(
                    bcs_ds, res_ds, nIter=intermediate_iter, rebuild_schedule=False
                )
                mid_errs = evaluate_rel_l2(
                    model,
                    model.params,
                    search.pde_spec,
                    N_test,
                    eval_seed,
                    predict_chunk,
                )
                mid_mean = float(np.mean(mid_errs))
                trial.report(mid_mean, step=intermediate_iter)
                if trial.should_prune():
                    raise optuna.TrialPruned(f"pruned at iter {intermediate_iter}")
                model.train(
                    bcs_ds,
                    res_ds,
                    nIter=n_iter - intermediate_iter,
                    rebuild_schedule=False,
                )
            else:
                model.train(bcs_ds, res_ds, nIter=n_iter, rebuild_schedule=False)
            train_time = time.time() - t0

            #  Final evaluation
            errs = evaluate_rel_l2(
                model,
                model.params,
                search.pde_spec,
                N_test,
                eval_seed,
                predict_chunk,
            )
        except optuna.TrialPruned:
            # Let Optuna's pruning signal propagate.
            raise
        except Exception as e:
            msg = str(e)
            if (
                "RESOURCE_EXHAUSTED" in msg
                or "Out of memory" in msg
                or "OOM" in msg
                or "XlaRuntimeError" in type(e).__name__
            ):
                # Free resources before next trial.
                try:
                    del model
                except Exception:
                    pass
                jax.clear_caches()
                raise optuna.TrialPruned(
                    f"OOM during training/eval: {type(e).__name__}: {msg[:200]}"
                )
            # Re-raise unrelated errors.
            raise

        rel_l2_mean = float(np.mean(errs))
        rel_l2_std = float(np.std(errs))

        trial.set_user_attr("rel_l2_std", rel_l2_std)
        trial.set_user_attr("train_time_sec", float(train_time))

        del model
        jax.clear_caches()
        return rel_l2_mean

    return objective


# Top-level orchestration  (study + summary + config save)


def run_study(
    search: SearchConfig,
    *,
    n_trials: int,
    target_completed: int | None = None,
    n_iter: int,
    intermediate_iter: int = 5000,
    N_test: int = 20,
    eval_seed: int = 1234567,
    train_seed: int = 0,
    max_params: int | None = 200_000,
    predict_chunk: int = 4096,
    bcs_batch: int = 14,
    res_batch: int = 14,
    init_lr: float = 1e-3,
    end_lr_frac: float = 0.01,
    warmup_frac: float = 0.0,
    weight_decay: float = 1e-4,
    study_name: str | None = None,
    storage: str | None = None,
    out_dir: str = "optuna_runs",
    config_dir: str = "configs",
    config_version: int = 1,
    sampler_seed: int = 42,
):
    """Run or resume an Optuna study (TPE; median pruning if intermediate_iter > 0); save trials CSV and best config.

    Runs up to n_trials more trials, stopping at target_completed completed ones; saves nothing if none completed.
    """
    if study_name is None:
        study_name = f"{search.arch_name}_{search.pde_spec.name}"
    if storage is None:
        storage = f"sqlite:///optuna_{study_name}.db"

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sampler = TPESampler(seed=sampler_seed)
    pruner = (
        MedianPruner(n_startup_trials=5, n_warmup_steps=0)
        if intermediate_iter > 0
        else optuna.pruners.NopPruner()
    )

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
        direction="minimize",
    )

    print(f"Study: {study_name}")
    print(f"  storage:        {storage}")
    print(f"  arch:           {search.arch_name}")
    print(f"  pde:            {search.pde_spec.name}")
    print(f"  trials so far:  {len(study.trials)}")
    if target_completed is not None:
        print(
            f"  target:         {target_completed} completed trials "
            f"(hard cap: {n_trials} more this session)"
        )
    else:
        print(f"  target:         {n_trials} additional trials")
    print(f"  per-trial:      {n_iter} iters, {N_test} test ICs")
    print(
        f"  param cap:      {max_params:,}" if max_params else f"  param cap:      none"
    )

    objective = make_objective(
        search,
        n_iter=n_iter,
        max_params=max_params,
        N_test=N_test,
        eval_seed=eval_seed,
        train_seed=train_seed,
        intermediate_iter=intermediate_iter,
        predict_chunk=predict_chunk,
        bcs_batch=bcs_batch,
        res_batch=res_batch,
        init_lr=init_lr,
        end_lr_frac=end_lr_frac,
        warmup_frac=warmup_frac,
    )

    callbacks = []
    if target_completed is not None:

        def _stop_when_target_completed(study_, trial_):
            n_done = sum(
                1 for t in study_.trials if t.state == optuna.trial.TrialState.COMPLETE
            )
            if n_done >= target_completed:
                study_.stop()

        callbacks.append(_stop_when_target_completed)

    study.optimize(
        objective,
        n_trials=n_trials,
        gc_after_trial=True,
        show_progress_bar=False,
        callbacks=callbacks,
    )

    #  Summary
    print("\n" + "--" * 72)
    print(f"Study complete: {len(study.trials)} total trials")
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    print(f"  completed: {len(completed)},  pruned: {len(pruned)}")

    if not completed:
        print("No completed trials; not writing a config.")
        return study

    best = study.best_trial
    print(f"\nBest trial #{best.number}:")
    print(f"  rel_L2 mean: {best.value:.4e}")
    print(f"  rel_L2 std:  {best.user_attrs.get('rel_l2_std', 'n/a')}")
    print(f"  params:      {best.user_attrs.get('param_count', 'n/a'):,}")
    print(f"  time:        {best.user_attrs.get('train_time_sec', 0):.1f}s")
    print(f"  hparams:")
    for k, v in best.params.items():
        print(f"    {k:>18}: {v}")

    #  Save the trials CSV for inspection
    df = study.trials_dataframe(
        attrs=(
            "number",
            "value",
            "state",
            "datetime_start",
            "duration",
            "params",
            "user_attrs",
        ),
    )
    csv_path = out_dir / f"{study_name}_trials.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nTrials CSV: {csv_path}")

    #  Save model config YAML for the best trial
    # Combine sampled hparams (from best.params) with fixed_hparams. We
    # re-evaluate sample_hparams on a frozen-best trial so any derived
    # values (e.g. defaults that depend on a sampled value) come through.
    # Easiest: just take best.params directly, since all sampled values
    # were the suggest_* keys.
    best_hparams = {**search.fixed_hparams, **best.params}
    training_kwargs = {
        "n_iter": n_iter,
        "init_lr": init_lr,
        "end_lr_frac": end_lr_frac,
        "warmup_frac": warmup_frac,
        "weight_decay": weight_decay,
        "bcs_batch_size": bcs_batch,
        "res_batch_size": res_batch,
    }
    metadata = {
        "optuna_study": study_name,
        "best_trial_number": best.number,
        "best_rel_l2_mean": float(best.value),
        "best_rel_l2_std": float(best.user_attrs.get("rel_l2_std", 0.0)),
        "param_count": int(best.user_attrs.get("param_count", 0)),
        "train_time_sec": float(best.user_attrs.get("train_time_sec", 0.0)),
        "completed_trials": len(completed),
        "pruned_trials": len(pruned),
    }

    config_path = (
        Path(config_dir)
        / f"{search.arch_name}_{search.pde_spec.name}_v{config_version}.yaml"
    )
    save_config(
        path=config_path,
        arch=search.arch_name,
        pde_name=search.pde_spec.name,
        pde_kwargs=search.pde_kwargs,
        hparams=best_hparams,
        training=training_kwargs,
        metadata=metadata,
    )
    print(f"Config:     {config_path}")

    return study


# Argparse helper used by per-architecture tune scripts


def add_common_args(p: argparse.ArgumentParser) -> None:
    """Add the command-line arguments shared by all hpo/tune_*.py scripts."""
    p.add_argument(
        "--n_trials",
        type=int,
        default=30,
        help="Number of trials to run (counting pruned). If "
        "--target_completed is also set, this becomes a "
        "session-level safety cap.",
    )
    p.add_argument(
        "--target_completed",
        type=int,
        default=None,
        help="If set, run until this many *completed* trials "
        "exist in the study (across sessions if storage is "
        "shared). Pruned trials don't count toward this. "
        "--n_trials caps how many trials this session can "
        "attempt before giving up.",
    )
    p.add_argument(
        "--n_iter",
        type=int,
        default=20_000,
        help="Training iters per trial (final eval at this step)",
    )
    p.add_argument(
        "--intermediate_iter",
        type=int,
        default=5000,
        help="Iter at which to report for pruning. 0 disables.",
    )
    p.add_argument(
        "--N_test",
        type=int,
        default=100,
        help="Validation trajectories for HPO evaluation.",
    )
    p.add_argument("--eval_seed", type=int, default=1_234_567)
    p.add_argument("--train_seed", type=int, default=0)
    p.add_argument("--max_params", type=int, default=200_000)
    p.add_argument("--predict_chunk", type=int, default=4096)
    p.add_argument("--bcs_batch", type=int, default=14)
    p.add_argument("--res_batch", type=int, default=14)
    p.add_argument("--init_lr", type=float, default=1e-3)
    p.add_argument("--end_lr_frac", type=float, default=0.01)
    p.add_argument("--warmup_frac", type=float, default=0.0)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument(
        "--study_name", type=str, default=None, help="Defaults to '{arch}_{pde}'."
    )
    p.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Defaults to sqlite:///optuna_{study_name}.db",
    )
    p.add_argument("--out_dir", type=str, default="optuna_runs")
    p.add_argument("--config_dir", type=str, default="configs")
    p.add_argument("--config_version", type=int, default=1)
    p.add_argument("--sampler_seed", type=int, default=42)
    p.add_argument(
        "--pde", type=str, default="burgers", help="PDE name (registered in pdes/)."
    )
    # PDE-specific kwargs are not exposed here in general; if a particular
    # study needs to vary PDE params (e.g. viscosity sweep), the per-arch
    # tune script can add those itself.
