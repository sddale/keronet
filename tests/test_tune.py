"""
Smoke test of hpo.tune_common.run_study on Burgers with toy search spaces: outputs and the written config.
"""

import os
from pathlib import Path

import pytest
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

from architectures import KerONet, PINTO, PIT, PIDeepONet
from pdes import get_pde
from architectures.model_config import load_config, build_model_from_config
from hpo.tune_common import SearchConfig, run_study


def sample_keronet(trial):
    return {
        "L": trial.suggest_categorical("L", [4, 8]),
        "d_k": trial.suggest_categorical("d_k", [16]),
        "d_v": trial.suggest_categorical("d_v", [16]),
        "post_mlp_hidden": trial.suggest_categorical("post_mlp_hidden", [16]),
        "branch_hidden": trial.suggest_categorical("branch_hidden", [64]),
        "trunk_hidden": trial.suggest_categorical("trunk_hidden", [64]),
        "branch_layers_n": trial.suggest_categorical("branch_layers_n", [2]),
        "trunk_layers_n": trial.suggest_categorical("trunk_layers_n", [2]),
        "mapping_size": trial.suggest_categorical("mapping_size", [32]),
        "sigma": trial.suggest_float("sigma", 0.8, 1.5, log=True),
    }


def sample_pinto(trial):
    return {
        "d_model": trial.suggest_categorical("d_model", [32, 64]),
        "n_heads": trial.suggest_categorical("n_heads", [2]),
        "key_dim": trial.suggest_categorical("key_dim", [32]),
        "n_cross_attn": trial.suggest_categorical("n_cross_attn", [1, 2]),
    }


def sample_pit(trial):
    return {
        "d_model": trial.suggest_categorical("d_model", [32, 64]),
        "n_heads": trial.suggest_categorical("n_heads", [2]),
        "key_dim": trial.suggest_categorical("key_dim", [16, 32]),
        "n_blocks": trial.suggest_categorical("n_blocks", [1, 2]),
        "pe_dim": trial.suggest_categorical("pe_dim", [16, 32]),
    }


def sample_pi_deeponet(trial):
    return {
        "branch_hidden": trial.suggest_categorical("branch_hidden", [64, 96]),
        "trunk_hidden": trial.suggest_categorical("trunk_hidden", [64, 96]),
        "branch_layers_n": trial.suggest_categorical("branch_layers_n", [2]),
        "trunk_layers_n": trial.suggest_categorical("trunk_layers_n", [2]),
        "p": trial.suggest_categorical("p", [64]),
        "mapping_size": trial.suggest_categorical("mapping_size", [32]),
        "sigma": trial.suggest_float("sigma", 0.8, 1.5, log=True),
    }


SMOKE_CASES = [
    ("keronet", KerONet, sample_keronet),
    ("pinto", PINTO, sample_pinto),
    ("pit", PIT, sample_pit),
    ("pi_deeponet", PIDeepONet, sample_pi_deeponet),
]


@pytest.mark.parametrize("arch_name,arch_class,sample_fn", SMOKE_CASES)
def test_hpo_smoke(arch_name, arch_class, sample_fn, tmp_path):
    pde_spec = get_pde("burgers")
    search = SearchConfig(
        arch_name=arch_name,
        arch_class=arch_class,
        sample_hparams=sample_fn,
        fixed_hparams={"n_sensors": 80, "weight_decay": 1e-4},
        pde_spec=pde_spec,
        pde_kwargs={},
    )

    out_dir = os.path.join(tmp_path, "optuna_runs")
    config_dir = os.path.join(tmp_path, "configs")
    storage = f"sqlite:///{os.path.join(tmp_path, f'{arch_name}_smoke.db')}"

    _ = run_study(
        search,
        n_trials=2,
        n_iter=400,
        intermediate_iter=200,
        N_test=3,
        eval_seed=42,
        train_seed=0,
        max_params=400_000,
        predict_chunk=512,
        bcs_batch=8,
        res_batch=8,
        init_lr=1e-3,
        end_lr_frac=0.01,
        warmup_frac=0.0,
        weight_decay=1e-4,
        study_name=f"{arch_name}_smoke",
        storage=storage,
        out_dir=out_dir,
        config_dir=config_dir,
        config_version=1,
        sampler_seed=42,
    )

    csv_path = Path(out_dir) / f"{arch_name}_smoke_trials.csv"
    assert csv_path.exists(), f"trials CSV not written: {csv_path}"

    cfg_path = Path(config_dir) / f"{arch_name}_burgers_v1.yaml"
    assert cfg_path.exists(), f"config not written: {cfg_path}"

    cfg = load_config(cfg_path)
    assert cfg["arch"] == arch_name
    assert cfg["pde"]["name"] == "burgers"
    assert "best_rel_l2_mean" in cfg["metadata"]
    assert "param_count" in cfg["metadata"]
    assert "best_trial_number" in cfg["metadata"]

    model, pde_spec_loaded = build_model_from_config(cfg)
    assert isinstance(model, arch_class), (
        f"Built wrong class: {type(model)} vs {arch_class}"
    )
    assert pde_spec_loaded.name == "burgers"
