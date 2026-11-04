"""
Smoke tests for evaluate.run_evaluation: outputs, test-set caching, resume, and failure handling.
Later tests reuse test_happy_path's outputs, so run the file as a whole.
"""

import json
from pathlib import Path

import pytest
import numpy as np

from architectures.model_config import save_config
from evaluate import run_evaluation
from hpo import tune_common


@pytest.fixture(scope="module")
def shared_tmp(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("evaluate_smoke")
    cache_dir = tmp / "data_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    orig_cache_dir = tune_common.DATA_CACHE_DIR
    orig_test_cache = dict(tune_common._TEST_CACHE)
    orig_train_cache = dict(tune_common._TRAIN_CACHE)
    tune_common.DATA_CACHE_DIR = cache_dir
    tune_common._TEST_CACHE.clear()
    tune_common._TRAIN_CACHE.clear()
    yield tmp
    tune_common.DATA_CACHE_DIR = orig_cache_dir
    tune_common._TEST_CACHE.clear()
    tune_common._TEST_CACHE.update(orig_test_cache)
    tune_common._TRAIN_CACHE.clear()
    tune_common._TRAIN_CACHE.update(orig_train_cache)


def _test_set_cache_path(tmp: Path, pde_name: str, N_test: int, test_seed: int) -> Path:
    return tmp / "data_cache" / f"test_{pde_name}_{N_test}_{test_seed}.npz"


def _write_tiny_keronet_config(path: Path, *, n_sensors: int = 80) -> Path:
    save_config(
        path=path,
        arch="keronet",
        pde_name="burgers",
        hparams={
            "n_sensors": n_sensors,
            "branch_hidden": 32,
            "branch_layers_n": 2,
            "trunk_hidden": 32,
            "trunk_layers_n": 2,
            "L": 4,
            "d_k": 8,
            "d_v": 8,
            "post_mlp_hidden": 8,
            "mapping_size": 16,
            "sigma": 1.0,
        },
        training={
            "n_iter": 200,
            "init_lr": 1e-3,
            "end_lr_frac": 0.01,
            "warmup_frac": 0.0,
            "weight_decay": 1e-4,
            "bcs_batch_size": 4,
            "res_batch_size": 4,
        },
        metadata={"source": "test_evaluate smoke"},
    )
    return path


def _write_broken_pinto_config(path: Path) -> Path:
    save_config(
        path=path,
        arch="pinto",
        pde_name="burgers",
        hparams={
            "n_sensors": 99,
            "d_model": 32,
            "n_heads": 2,
            "key_dim": 32,
            "n_cross_attn": 1,
        },
        training={
            "n_iter": 200,
            "init_lr": 1e-3,
            "end_lr_frac": 0.01,
            "warmup_frac": 0.0,
            "weight_decay": 1e-4,
            "bcs_batch_size": 4,
            "res_batch_size": 4,
        },
        metadata={"source": "test_evaluate broken-config smoke"},
    )
    return path


def test_happy_path(shared_tmp):
    cfg_path = shared_tmp / "configs" / "keronet_burgers_smoke.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    _write_tiny_keronet_config(cfg_path)

    out_dir = shared_tmp / "results" / "burgers_smoke"

    summary = run_evaluation(
        pde_name="burgers",
        pde_kwargs={},
        config_paths=[cfg_path],
        seeds=[0, 1],
        output_dir=out_dir,
        N_test=2,
        test_seed=42,
        skip_existing=True,
        n_iter=200,
        predict_chunk=512,
    )

    test_set_path = _test_set_cache_path(shared_tmp, "burgers", 2, 42)
    assert test_set_path.exists(), f"test set not cached at {test_set_path}"
    assert test_set_path.with_suffix(".meta.json").exists()
    assert (out_dir / "ground_truth.npy").exists()
    assert (out_dir / "evaluation_meta.json").exists()
    assert (out_dir / "summary.json").exists()

    for seed in [0, 1]:
        cell = out_dir / f"keronet_seed{seed}"
        for fname in [
            "checkpoint.pkl",
            "loss_log.npz",
            "per_ic_errors.npy",
            "predictions.npy",
            "config_used.yaml",
            "meta.json",
            "done",
        ]:
            assert (cell / fname).exists(), f"missing {fname} for seed={seed}"
        errs = np.load(cell / "per_ic_errors.npy")
        assert errs.shape == (2,), f"per_ic_errors shape: {errs.shape}"
        assert np.all(np.isfinite(errs))
        preds = np.load(cell / "predictions.npy")
        gt = np.load(out_dir / "ground_truth.npy")
        assert preds.shape == gt.shape, (
            f"predictions {preds.shape} != ground truth {gt.shape}"
        )

    assert summary["done_count"] == 2
    assert summary["skipped_count"] == 0
    assert summary["failed_count"] == 0
    assert "keronet" in summary["by_arch"]
    arch_agg = summary["by_arch"]["keronet"]
    assert arch_agg["n_seeds"] == 2


def test_resume(shared_tmp):
    cfg_path = shared_tmp / "configs" / "keronet_burgers_smoke.yaml"
    out_dir = shared_tmp / "results" / "burgers_smoke"

    assert (out_dir / "keronet_seed0" / "done").exists(), (
        "test_happy_path didn't leave completed cells; cannot test resume"
    )

    ckpt = out_dir / "keronet_seed0" / "checkpoint.pkl"
    pre_mtime = ckpt.stat().st_mtime

    summary = run_evaluation(
        pde_name="burgers",
        pde_kwargs={},
        config_paths=[cfg_path],
        seeds=[0, 1],
        output_dir=out_dir,
        N_test=2,
        test_seed=42,
        skip_existing=True,
        n_iter=200,
        predict_chunk=512,
    )

    assert summary["skipped_count"] == 2, (
        f"expected to skip both cells, got skipped={summary['skipped_count']}"
    )
    assert summary["done_count"] == 0
    assert summary["failed_count"] == 0

    post_mtime = ckpt.stat().st_mtime
    assert post_mtime == pre_mtime, (
        "checkpoint was rewritten despite skip_existing=True"
    )


def test_test_set_cache(shared_tmp):
    out_dir = shared_tmp / "results" / "burgers_smoke"
    test_set_path = _test_set_cache_path(shared_tmp, "burgers", 2, 42)
    assert test_set_path.exists()
    pre_mtime = test_set_path.stat().st_mtime

    cfg_path = shared_tmp / "configs" / "keronet_burgers_smoke.yaml"
    run_evaluation(
        pde_name="burgers",
        pde_kwargs={},
        config_paths=[cfg_path],
        seeds=[0],
        output_dir=out_dir,
        N_test=2,
        test_seed=42,
        skip_existing=True,
        n_iter=200,
        predict_chunk=512,
    )
    post_mtime = test_set_path.stat().st_mtime
    assert post_mtime == pre_mtime, (
        "test_set.npz was rewritten despite identical cache key"
    )


def test_failure_handling(shared_tmp):
    good_cfg = shared_tmp / "configs" / "keronet_burgers_smoke.yaml"
    bad_cfg = shared_tmp / "configs" / "pinto_broken.yaml"
    _write_broken_pinto_config(bad_cfg)

    out_dir = shared_tmp / "results" / "burgers_failtest"

    summary = run_evaluation(
        pde_name="burgers",
        pde_kwargs={},
        config_paths=[good_cfg, bad_cfg],
        seeds=[0],
        output_dir=out_dir,
        N_test=2,
        test_seed=42,
        skip_existing=True,
        n_iter=200,
        predict_chunk=512,
    )

    good_cell = out_dir / "keronet_seed0"
    assert (good_cell / "done").exists(), "good cell did not complete"

    bad_cell = out_dir / "pinto_seed0"
    assert bad_cell.exists(), "failed cell directory not created"
    assert (bad_cell / "failed.json").exists(), (
        "failed.json was not written for the broken cell"
    )
    assert not (bad_cell / "done").exists(), (
        "done marker should NOT exist for failed cell"
    )
    with open(bad_cell / "failed.json") as f:
        fail_meta = json.load(f)
    assert fail_meta["arch"] == "pinto"
    assert fail_meta["seed"] == 0
    assert "error_type" in fail_meta
    assert "traceback" in fail_meta

    assert summary["done_count"] == 1
    assert summary["failed_count"] == 1

    assert "keronet" in summary["by_arch"]
    assert "pinto" not in summary["by_arch"], (
        "failed arch should not appear in by_arch aggregation"
    )
