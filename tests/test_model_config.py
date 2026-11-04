"""
Tests for architectures/model_config.py: registry, save/load round trip, model building, and config validation.
"""

import os
import pytest
import yaml as _yaml
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from architectures.model_config import (
    save_config,
    load_config,
    build_model_from_config,
    list_architectures,
    get_architecture,
    CONFIG_VERSION,
)


def test_registry():
    archs = list_architectures()
    expected = {"pi_deeponet", "pit", "pinto", "keronet", "mkeronet"}
    assert set(archs) == expected, f"Expected {expected}, got {set(archs)}"


def test_save_config_writes_valid_yaml(tmp_path):
    path = str(tmp_path / "keronet_burgers_test.yaml")
    save_config(
        path=path,
        arch="keronet",
        pde_name="burgers",
        pde_kwargs={"visc": 0.01},
        hparams={
            "n_sensors": 80,
            "branch_hidden": 64,
            "branch_layers_n": 2,
            "trunk_hidden": 64,
            "trunk_layers_n": 2,
            "L": 8,
            "d_k": 16,
            "d_v": 16,
            "mapping_size": 32,
            "sigma": 1.0,
        },
        training={
            "n_iter": 10000,
            "init_lr": 1e-3,
            "end_lr_frac": 0.01,
            "warmup_frac": 0.0,
            "weight_decay": 1e-4,
            "bcs_batch_size": 14,
            "res_batch_size": 14,
        },
        metadata={
            "best_rel_l2": 0.0234,
            "param_count": 35906,
            "optuna_trial": 7,
        },
    )
    assert os.path.exists(path)
    with open(path) as f:
        contents = f.read()
    assert "keronet" in contents
    assert "burgers" in contents


def test_load_config_round_trip(tmp_path):
    path = str(tmp_path / "test.yaml")
    orig_hparams = {
        "n_sensors": 80,
        "branch_hidden": 64,
        "branch_layers_n": 2,
        "trunk_hidden": 64,
        "trunk_layers_n": 2,
        "L": 8,
        "d_k": 16,
        "d_v": 16,
        "mapping_size": 32,
        "sigma": 1.0,
    }
    save_config(
        path=path,
        arch="keronet",
        pde_name="burgers",
        hparams=orig_hparams,
        training={
            "n_iter": 1000,
            "init_lr": 1e-3,
            "end_lr_frac": 0.01,
            "warmup_frac": 0.0,
            "weight_decay": 1e-4,
            "bcs_batch_size": 14,
            "res_batch_size": 14,
        },
    )
    cfg = load_config(path)
    assert cfg["arch"] == "keronet"
    assert cfg["pde"]["name"] == "burgers"
    for k, v in orig_hparams.items():
        assert cfg["hparams"][k] == v, f"hparam {k} round-trip mismatch"
    assert "generated_at" in cfg["metadata"]
    assert cfg["config_version"] == CONFIG_VERSION


BUILD_CASES = [
    (
        "pi_deeponet",
        {
            "n_sensors": 80,
            "branch_hidden": 64,
            "branch_layers_n": 2,
            "trunk_hidden": 64,
            "trunk_layers_n": 2,
            "p": 64,
            "mapping_size": 32,
            "sigma": 1.0,
        },
    ),
    (
        "pit",
        {
            "n_sensors": 80,
            "d_model": 64,
            "n_heads": 2,
            "key_dim": 32,
            "n_blocks": 1,
            "pe_dim": 32,
        },
    ),
    (
        "pinto",
        {
            "n_sensors": 80,
            "d_model": 64,
            "n_heads": 2,
            "key_dim": 64,
            "n_cross_attn": 3,
        },
    ),
    (
        "keronet",
        {
            "n_sensors": 80,
            "branch_hidden": 64,
            "branch_layers_n": 2,
            "trunk_hidden": 64,
            "trunk_layers_n": 2,
            "L": 8,
            "d_k": 16,
            "d_v": 16,
            "mapping_size": 32,
            "sigma": 1.0,
        },
    ),
]


@pytest.mark.parametrize("arch_name,hparams", BUILD_CASES)
def test_build_model_from_config(arch_name, hparams, tmp_path):
    path = str(tmp_path / f"{arch_name}.yaml")
    save_config(
        path=path,
        arch=arch_name,
        pde_name="burgers",
        hparams=hparams,
        training={
            "n_iter": 100,
            "init_lr": 1e-3,
            "end_lr_frac": 0.01,
            "warmup_frac": 0.0,
            "weight_decay": 1e-4,
            "bcs_batch_size": 4,
            "res_batch_size": 4,
        },
    )
    cfg = load_config(path)
    model, pde_spec = build_model_from_config(cfg)
    cls = get_architecture(arch_name)
    assert isinstance(model, cls), f"Wrong class for {arch_name}"
    flat, _ = ravel_pytree(model.params)
    assert len(flat) > 0
    u_test = jnp.ones(80) * 0.3
    out = model.operator_net(model.params, u_test, 0.5, 0.2)
    assert jnp.isfinite(out)
    assert pde_spec.name == "burgers"


def test_pde_kwargs_round_trip(tmp_path):
    path = str(tmp_path / "low_visc.yaml")
    save_config(
        path=path,
        arch="keronet",
        pde_name="burgers",
        pde_kwargs={"visc": 0.005},
        hparams={
            "n_sensors": 80,
            "branch_hidden": 64,
            "branch_layers_n": 2,
            "trunk_hidden": 64,
            "trunk_layers_n": 2,
            "L": 8,
            "d_k": 16,
            "d_v": 16,
            "mapping_size": 32,
            "sigma": 1.0,
        },
        training={
            "n_iter": 100,
            "init_lr": 1e-3,
            "end_lr_frac": 0.01,
            "warmup_frac": 0.0,
            "weight_decay": 1e-4,
            "bcs_batch_size": 4,
            "res_batch_size": 4,
        },
    )
    cfg = load_config(path)
    _model, pde_spec = build_model_from_config(cfg)
    assert pde_spec.params["visc"] == 0.005, (
        f"viscosity round-trip failed: got {pde_spec.params['visc']}"
    )


def test_unknown_arch_rejected(tmp_path):
    path = str(tmp_path / "bad.yaml")
    with pytest.raises(KeyError):
        save_config(
            path=path,
            arch="not_a_real_arch",
            pde_name="burgers",
            hparams={},
            training={},
        )


def test_unknown_pde_rejected(tmp_path):
    path = str(tmp_path / "bad.yaml")
    with pytest.raises(KeyError):
        save_config(
            path=path,
            arch="keronet",
            pde_name="not_a_real_pde",
            hparams={},
            training={},
        )


def test_incomplete_config_rejected(tmp_path):
    path = str(tmp_path / "incomplete.yaml")
    with open(path, "w") as f:
        _yaml.safe_dump({"arch": "keronet"}, f)
    with pytest.raises(ValueError):
        load_config(path)
