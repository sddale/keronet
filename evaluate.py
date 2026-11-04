"""Train and evaluate model configs (one per architecture) on one PDE for several seeds.

Each cell <output_dir>/<arch>_seed<seed>/ gets a checkpoint, loss log, per-function errors,
predictions, config, meta.json, and a done marker (done cells are skipped) or failed.json.

Usage: uv run evaluate.py --pde burgers --configs CONFIG.yaml ... --output_dir results/burgers
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import platform
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from jax.flatten_util import ravel_pytree

from pdes import get_pde, PDESpec
from architectures.model_config import load_config, build_model_from_config


# Utilities
def env_info() -> Dict[str, str]:
    """Return Python and JAX versions, platform, command line, timestamp, and git commit if available."""
    info = {
        "python_version": sys.version.split()[0],
        "jax_version": jax.__version__,
        "platform": platform.platform(),
        "command_line": " ".join(sys.argv),
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
    }
    # Best-effort git commit
    try:
        import subprocess

        sha = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
        info["git_commit"] = sha
    except Exception:
        pass
    return info


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=str)


def write_yaml_passthrough(src: Path, dst: Path) -> None:
    """Copy a config YAML byte-for-byte to dst, creating its parent directory."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(src.read_bytes())


# Test-set caching  (one per PDE, per output_dir)


def get_or_make_test_set(
    pde_spec: PDESpec,
    output_dir: Path,
    N_test: int,
    seed: int,
) -> tuple:
    """Return the test set as numpy arrays via hpo.tune_common.get_test_set (output_dir is unused)."""
    from hpo.tune_common import get_test_set as shared_get_test_set

    u_test, y_test, s_test = shared_get_test_set(pde_spec, N_test, seed)
    # tune_common returns jnp arrays for in-memory cache compatibility; the
    # rest of evaluate.py expects numpy.
    return np.asarray(u_test), np.asarray(y_test), np.asarray(s_test)


# Per-cell evaluation


def per_ic_rel_l2(
    s_pred: np.ndarray, s_test: np.ndarray, N_test: int, pts_per_traj: int
) -> np.ndarray:
    """Per-IC relative L2 error from flat prediction/target arrays."""
    s_pred_arr = s_pred.reshape(N_test, pts_per_traj)
    s_test_arr = s_test.reshape(N_test, pts_per_traj)
    num = np.linalg.norm(s_pred_arr - s_test_arr, axis=1)
    den = np.linalg.norm(s_test_arr, axis=1) + 1e-12
    return num / den


def predict_chunked(
    model, params, u_test, y_test, predict_chunk: int = 4096
) -> np.ndarray:
    """Return model.predict_s on the test points, computed in chunks of predict_chunk rows."""
    n_total = u_test.shape[0]
    chunks = []
    for start in range(0, n_total, predict_chunk):
        end = min(start + predict_chunk, n_total)
        chunks.append(
            np.asarray(model.predict_s(params, u_test[start:end], y_test[start:end]))
        )
    return np.concatenate(chunks, axis=0)


def run_one_cell(
    cfg_path: Path,
    cfg: Dict[str, Any],
    pde_spec: PDESpec,
    seed: int,
    test_set: tuple,
    pts_per_traj: int,
    cell_dir: Path,
    *,
    n_iter: int | None = None,
    predict_chunk: int = 4096,
    init_lr: float | None = None,
    end_lr_frac: float | None = None,
    warmup_frac: float | None = None,
    weight_decay: float | None = None,
) -> Dict[str, Any]:
    """Train and evaluate one (arch, seed) cell, write its artifacts to cell_dir, and return its meta dict.

    The seed sets initialization, training data, and batch order; n_iter, LR and weight_decay arguments
    override the config.
    """
    arch_name = cfg["arch"]
    cell_dir.mkdir(parents=True, exist_ok=True)

    # 1. Build model with this cell's seed for parameter initialization.
    # We splice init_seed=seed into hparams so the architecture's PRNGKey
    # uses this cell's seed rather than the protocol-fixed 0. This makes
    # multi-seed evaluation vary BOTH training-data realization AND
    # parameter initialization across seeds. HPO continues to use the
    # default init_seed=0 (set in each architecture constructor) for
    # variance reduction across trials.
    cfg_with_init = dict(cfg)
    cfg_with_init["hparams"] = {**cfg["hparams"], "init_seed": int(seed)}
    model, _ = build_model_from_config(cfg_with_init)
    flat, _ = ravel_pytree(model.params)
    param_count = int(len(flat))

    # 2. Resolve training kwargs (CLI overrides config)
    cfg_train = cfg.get("training", {})
    kw = {
        "n_iter": n_iter if n_iter is not None else cfg_train.get("n_iter", 20_000),
        "init_lr": init_lr if init_lr is not None else cfg_train.get("init_lr", 1e-3),
        "end_lr_frac": end_lr_frac
        if end_lr_frac is not None
        else cfg_train.get("end_lr_frac", 0.01),
        "warmup_frac": warmup_frac
        if warmup_frac is not None
        else cfg_train.get("warmup_frac", 0.0),
        "weight_decay": weight_decay
        if weight_decay is not None
        else cfg_train.get("weight_decay", 1e-4),
        "bcs_batch": cfg_train.get("bcs_batch_size", 14),
        "res_batch": cfg_train.get("res_batch_size", 14),
    }

    # 3. Generate training data for this seed
    print(f"[{arch_name} seed={seed}] generating training data")
    train_data = pde_spec.generate_training_data(random.PRNGKey(seed))

    # 4. Build datasets --- same tuple-slicing logic as tune_common.
    # BCS and RES generators share an rng_key so that each iteration
    # samples the same 14 input-function indices for both losses,
    # matching the shared-batch convention used in PI-DeepONet (Wang
    # 2021) and most operator-PINN follow-ups.
    bcs_cls = pde_spec.default_dataset_classes["bcs"]
    res_cls = pde_spec.default_dataset_classes["res"]
    u, y_res, s_res = train_data[0], train_data[-2], train_data[-1]
    bcs_fields = train_data[:-2]
    batch_key = random.PRNGKey(seed + 1000)
    bcs_ds = bcs_cls(*bcs_fields, batch_size=kw["bcs_batch"], rng_key=batch_key)
    res_ds = res_cls(u, y_res, s_res, batch_size=kw["res_batch"], rng_key=batch_key)

    # 5. Train
    print(
        f"[{arch_name} seed={seed}] training: {kw['n_iter']:,} iters, "
        f"{param_count:,} params"
    )
    t0 = time.time()
    model.train(
        bcs_ds,
        res_ds,
        nIter=kw["n_iter"],
        init_lr=kw["init_lr"],
        end_lr_frac=kw["end_lr_frac"],
        warmup_frac=kw["warmup_frac"],
        weight_decay=kw["weight_decay"],
        rebuild_schedule=True,
    )
    train_time = time.time() - t0
    print(f"[{arch_name} seed={seed}] trained in {train_time:.1f}s")

    # 6. Evaluate on test set
    u_test, y_test, s_test = test_set
    N_test = s_test.size // pts_per_traj
    print(f"[{arch_name} seed={seed}] evaluating on {N_test} test ICs")
    s_pred = predict_chunked(
        model, model.params, u_test, y_test, predict_chunk=predict_chunk
    )
    errs = per_ic_rel_l2(s_pred, s_test, N_test, pts_per_traj)
    rel_l2_mean = float(errs.mean())
    rel_l2_std = float(errs.std())
    rel_l2_min = float(errs.min())
    rel_l2_max = float(errs.max())
    worst_ic = int(errs.argmax())
    print(
        f"[{arch_name} seed={seed}] rel_L2 mean={rel_l2_mean:.4e} "
        f"std={rel_l2_std:.4e} max={rel_l2_max:.4e}"
    )

    # 7. Save artifacts
    # Checkpoint
    with open(cell_dir / "checkpoint.pkl", "wb") as f:
        pickle.dump({"params": model.params}, f)

    # Loss log
    np.savez(
        cell_dir / "loss_log.npz",
        loss_total=np.asarray(model.loss_log),
        loss_bcs=np.asarray(model.loss_bcs_log),
        loss_res=np.asarray(model.loss_res_log),
    )

    # Per-IC errors and predictions
    np.save(cell_dir / "per_ic_errors.npy", errs)
    np.save(cell_dir / "predictions.npy", s_pred)
    # Ground truth is shared across all cells; saved once at PDE level.

    # Config passthrough
    write_yaml_passthrough(cfg_path, cell_dir / "config_used.yaml")

    # Meta
    meta = {
        "arch": arch_name,
        "pde": pde_spec.name,
        "seed": int(seed),
        "param_count": param_count,
        "train_time_s": float(train_time),
        "n_iter": int(kw["n_iter"]),
        "rel_l2_mean": rel_l2_mean,
        "rel_l2_std": rel_l2_std,
        "rel_l2_min": rel_l2_min,
        "rel_l2_max": rel_l2_max,
        "worst_ic_index": worst_ic,
        "n_test_ics": int(N_test),
        "training_kwargs": kw,
        **env_info(),
    }
    write_json(cell_dir / "meta.json", meta)

    # 8. Mark done
    (cell_dir / "done").write_text(datetime.now().isoformat(timespec="seconds"))

    # 9. Free JAX state before next cell
    del model
    jax.clear_caches()

    return meta


# Orchestration


def cell_dirname(arch: str, seed: int) -> str:
    return f"{arch}_seed{seed}"


def run_evaluation(
    pde_name: str,
    pde_kwargs: Dict[str, Any],
    config_paths: List[Path],
    seeds: List[int],
    output_dir: Path,
    *,
    N_test: int = 100,
    test_seed: int = 7_654_321,
    skip_existing: bool = True,
    n_iter: int | None = None,
    predict_chunk: int = 4096,
    init_lr: float | None = None,
    end_lr_frac: float | None = None,
    warmup_frac: float | None = None,
    weight_decay: float | None = None,
):
    """Evaluate every config for every seed on one PDE; return the summary also written to summary.json."""
    output_dir.mkdir(parents=True, exist_ok=True)

    #  Build PDE spec, generate/load test set
    pde_spec = get_pde(pde_name, **pde_kwargs)
    nn = pde_spec.default_test_kwargs.get("nn", 101)
    steps = pde_spec.default_test_kwargs.get("steps", 100)
    pts_per_traj = pde_spec.default_test_kwargs.get("pts_per_traj", nn * (steps + 1))

    test_set = get_or_make_test_set(pde_spec, output_dir, N_test, test_seed)

    # Save ground truth once at PDE level for downstream plotting
    gt_path = output_dir / "ground_truth.npy"
    if not gt_path.exists():
        np.save(gt_path, test_set[2])

    # PDE-level meta
    write_json(
        output_dir / "evaluation_meta.json",
        {
            "pde": pde_spec.name,
            "pde_kwargs": pde_kwargs,
            "N_test": int(N_test),
            "test_seed": int(test_seed),
            "pts_per_traj": int(pts_per_traj),
            "seeds": [int(s) for s in seeds],
            "configs": [str(p) for p in config_paths],
            **env_info(),
        },
    )

    #  Load all configs upfront so a typo dies early
    cfgs = []
    for p in config_paths:
        cfg = load_config(p)
        if cfg["pde"]["name"] != pde_name:
            raise ValueError(
                f"Config {p} is for PDE {cfg['pde']['name']!r}, "
                f"but --pde is {pde_name!r}"
            )
        cfgs.append((Path(p), cfg))

    #  Run all (arch, seed) cells
    summary: Dict[str, Any] = {
        "pde": pde_spec.name,
        "seeds": [int(s) for s in seeds],
        "by_cell": {},
        "by_arch": {},
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }

    total = len(cfgs) * len(seeds)
    done_count = 0
    skipped_count = 0
    failed_count = 0

    for cfg_path, cfg in cfgs:
        arch = cfg["arch"]
        for seed in seeds:
            cell = output_dir / cell_dirname(arch, seed)
            cell_id = f"{arch}_seed{seed}"
            done_marker = cell / "done"

            print(f"\n{'━' * 72}")
            print(
                f"Cell {cell_id}  ({done_count + skipped_count + failed_count + 1}/{total})"
            )
            print(f"{'━' * 72}")

            if done_marker.exists() and skip_existing:
                print(f"[skip] {cell_id} already done")
                # Load its meta into the summary
                if (cell / "meta.json").exists():
                    with open(cell / "meta.json") as f:
                        summary["by_cell"][cell_id] = json.load(f)
                skipped_count += 1
                continue

            try:
                meta = run_one_cell(
                    cfg_path=cfg_path,
                    cfg=cfg,
                    pde_spec=pde_spec,
                    seed=seed,
                    test_set=test_set,
                    pts_per_traj=pts_per_traj,
                    cell_dir=cell,
                    n_iter=n_iter,
                    predict_chunk=predict_chunk,
                    init_lr=init_lr,
                    end_lr_frac=end_lr_frac,
                    warmup_frac=warmup_frac,
                    weight_decay=weight_decay,
                )
                summary["by_cell"][cell_id] = meta
                done_count += 1
            except Exception as e:
                tb = traceback.format_exc()
                print(f"[FAIL] {cell_id}: {e}")
                cell.mkdir(parents=True, exist_ok=True)
                write_json(
                    cell / "failed.json",
                    {
                        "arch": arch,
                        "seed": int(seed),
                        "error_type": type(e).__name__,
                        "error_msg": str(e),
                        "traceback": tb,
                        "failed_at": datetime.now().isoformat(timespec="seconds"),
                    },
                )
                failed_count += 1
                # continue to next cell
                continue

    #  Aggregate by arch
    for cfg_path, cfg in cfgs:
        arch = cfg["arch"]
        cell_metas = [
            summary["by_cell"][f"{arch}_seed{s}"]
            for s in seeds
            if f"{arch}_seed{s}" in summary["by_cell"]
        ]
        if not cell_metas:
            continue
        means = np.array([m["rel_l2_mean"] for m in cell_metas])
        stds = np.array([m["rel_l2_std"] for m in cell_metas])
        params = cell_metas[0]["param_count"]
        times = np.array([m["train_time_s"] for m in cell_metas])
        summary["by_arch"][arch] = {
            "n_seeds": len(cell_metas),
            "rel_l2_mean_mean": float(means.mean()),
            "rel_l2_mean_std": float(means.std()),
            "rel_l2_std_mean": float(stds.mean()),
            "param_count": int(params),
            "train_time_mean": float(times.mean()),
            "train_time_std": float(times.std()),
        }

    summary["done_count"] = done_count
    summary["skipped_count"] = skipped_count
    summary["failed_count"] = failed_count
    summary["finished_at"] = datetime.now().isoformat(timespec="seconds")
    write_json(output_dir / "summary.json", summary)

    #  Print final summary table
    print(f"\n{'--' * 72}")
    print(
        f"Evaluation complete: {done_count} done, {skipped_count} skipped, "
        f"{failed_count} failed"
    )
    print(f"{'--' * 72}")
    print(f"Output: {output_dir}/summary.json\n")
    if summary["by_arch"]:
        print(
            f"{'arch':>14} {'rel_L2 mean':>14} {'mean(std_per_seed)':>20} "
            f"{'params':>10} {'time/run':>10}"
        )
        for arch, agg in summary["by_arch"].items():
            print(
                f"{arch:>14} "
                f"{agg['rel_l2_mean_mean']:>9.4e} ± {agg['rel_l2_mean_std']:.2e}  "
                f"{agg['rel_l2_std_mean']:>17.4e}  "
                f"{agg['param_count']:>10,}  "
                f"{agg['train_time_mean']:>9.1f}s"
            )

    return summary


# CLI


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--pde", type=str, required=True, help="PDE name (registered in pdes/)."
    )
    p.add_argument(
        "--configs",
        type=str,
        nargs="+",
        required=True,
        help="Paths to model config YAMLs (one per architecture).",
    )
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--N_test", type=int, default=100)
    p.add_argument(
        "--test_seed",
        type=int,
        default=7_654_321,
        help="RNG seed for the held-out test set. MUST differ "
        "from HPO's eval_seed (default 1234567) to avoid "
        "evaluating on the same input functions HPO selected "
        "hyperparameters against.",
    )
    p.add_argument("--predict_chunk", type=int, default=4096)
    p.add_argument(
        "--no_skip_existing",
        action="store_true",
        help="Re-run cells even if 'done' marker is present.",
    )

    # Optional CLI overrides for training kwargs from the config
    p.add_argument("--n_iter", type=int, default=None)
    p.add_argument("--init_lr", type=float, default=None)
    p.add_argument("--end_lr_frac", type=float, default=None)
    p.add_argument("--warmup_frac", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=None)

    args = p.parse_args()

    run_evaluation(
        pde_name=args.pde,
        pde_kwargs={},
        config_paths=[Path(c) for c in args.configs],
        seeds=args.seeds,
        output_dir=Path(args.output_dir),
        N_test=args.N_test,
        test_seed=args.test_seed,
        skip_existing=not args.no_skip_existing,
        n_iter=args.n_iter,
        predict_chunk=args.predict_chunk,
        init_lr=args.init_lr,
        end_lr_frac=args.end_lr_frac,
        warmup_frac=args.warmup_frac,
        weight_decay=args.weight_decay,
    )


if __name__ == "__main__":
    main()
