"""
Smoke tests for the 1D PDE specs: registry, training and test data, residual and BC losses, and datasets.
"""

import pytest
import jax.numpy as jnp
from jax import random

from pdes import get_pde, list_pdes, PDESpec


class ToyModel:
    def operator_net(self, params, u, x, t):
        return params["a"] * jnp.mean(u) + params["b"] * x + params["c"] * t


TOY_PARAMS = {
    "a": jnp.float32(0.5),
    "b": jnp.float32(0.1),
    "c": jnp.float32(0.1),
}


def test_registry():
    assert set(list_pdes()) == {
        "burgers",
        "advection",
        "allen_cahn",
        "diffusion_reaction",
        "allen_cahn_2d",
    }, f"Unexpected PDE registry: {list_pdes()}"


PERIODIC_CASES = [
    ("burgers", dict(N_test=2, nn=21, s=512, steps=10)),
    ("advection", dict(N_test=2, nn=21, steps=10)),
    ("allen_cahn", dict(N_test=2, nn=21, s=512, steps=10)),
]

DIRICHLET_CASES = [
    ("diffusion_reaction", dict(N_test=2, nn=21, steps=10)),
]


@pytest.mark.parametrize("name,build_overrides", PERIODIC_CASES)
def test_periodic_ic_smoke(name, build_overrides):
    spec = get_pde(
        name,
        N_train=4,
        P_ic=20,
        P_bc=10,
        Q_train=100,
        m=80,
        **build_overrides,
    )
    assert isinstance(spec, PDESpec)
    assert spec.bcs_structure.kind == "periodic_ic"
    assert spec.bcs_structure.n_inputs == 4

    out = spec.generate_training_data(random.PRNGKey(0))
    u, y_ic, s_ic, y_bc_lo, y_bc_hi, y_res, s_res = out
    assert u.shape == (4, 80)
    assert y_ic.shape == (4, 20, 2)
    assert s_ic.shape == (4, 20, 1)
    assert y_bc_lo.shape == (4, 10, 2)
    assert y_bc_hi.shape == (4, 10, 2)
    assert y_res.shape == (4, 100, 2)
    assert s_res.shape == (4, 100, 1)
    assert jnp.allclose(s_res, 0.0)

    def toy_op(x, t):
        return 0.3 * jnp.sin(2.0 * jnp.pi * x) * jnp.exp(-t)

    res_val = spec.residual_fn(toy_op, jnp.zeros(80), 0.4, 0.2)
    assert jnp.isfinite(res_val)

    batch = (
        (u[:2], y_ic[:2], y_bc_lo[:2], y_bc_hi[:2]),
        s_ic[:2],
    )
    loss = spec.bcs_loss_fn(ToyModel(), TOY_PARAMS, batch)
    assert jnp.isfinite(loss) and loss >= 0

    DataGen = spec.default_dataset_classes["res"]
    BCSGen = spec.default_dataset_classes["bcs"]
    res_ds = DataGen(u, y_res, s_res, batch_size=2, rng_key=random.PRNGKey(1))
    bcs_ds = BCSGen(
        u, y_ic, s_ic, y_bc_lo, y_bc_hi, batch_size=2, rng_key=random.PRNGKey(2)
    )
    inputs_res, outputs_res = res_ds[0]
    u_b, y_b = inputs_res
    assert u_b.shape == (2, 80) and y_b.shape == (2, 100, 2)
    assert outputs_res.shape == (2, 100, 1)
    inputs_bcs, outputs_bcs = bcs_ds[0]
    u_b2, y_ic_b, y_bc_lo_b, y_bc_hi_b = inputs_bcs
    assert u_b2.shape == (2, 80) and y_ic_b.shape == (2, 20, 2)
    assert y_bc_lo_b.shape == (2, 10, 2) and y_bc_hi_b.shape == (2, 10, 2)
    assert outputs_bcs.shape == (2, 20, 1)


@pytest.mark.parametrize("name,build_overrides", DIRICHLET_CASES)
def test_dirichlet_smoke(name, build_overrides):
    spec = get_pde(
        name,
        N_train=4,
        P_ic=20,
        P_bc=20,
        Q_train=100,
        m=80,
        **build_overrides,
    )
    assert isinstance(spec, PDESpec)
    assert spec.bcs_structure.kind == "dirichlet_zero_ic"
    assert spec.bcs_structure.n_inputs == 3

    out = spec.generate_training_data(random.PRNGKey(0))
    u, y_ic, s_ic, y_bc, s_bc, y_res, s_res = out
    assert u.shape == (4, 80)
    assert y_ic.shape == (4, 20, 2)
    assert s_ic.shape == (4, 20, 1)
    assert y_bc.shape == (4, 20, 2)
    assert s_bc.shape == (4, 20, 1)
    assert y_res.shape == (4, 100, 2)
    assert s_res.shape == (4, 100, 1)

    assert jnp.allclose(s_ic, 0.0)
    assert jnp.allclose(s_bc, 0.0)
    assert not jnp.allclose(s_res, 0.0), (
        "diffusion-reaction residual targets should be nonzero (forcing)"
    )

    bc_xs = y_bc[..., 0].flatten()
    assert jnp.all((bc_xs == 0.0) | (bc_xs == 1.0)), "BC points must lie at x=0 or x=1"

    def toy_op(x, t):
        return 0.3 * jnp.sin(2.0 * jnp.pi * x) * (1.0 - jnp.exp(-t))

    res_val = spec.residual_fn(toy_op, u[0], 0.4, 0.2)
    assert jnp.isfinite(res_val)

    batch = (
        (u[:2], y_ic[:2], y_bc[:2]),
        (s_ic[:2], s_bc[:2]),
    )
    loss = spec.bcs_loss_fn(ToyModel(), TOY_PARAMS, batch)
    assert jnp.isfinite(loss) and loss >= 0

    DataGen = spec.default_dataset_classes["res"]
    BCSGen = spec.default_dataset_classes["bcs"]
    res_ds = DataGen(u, y_res, s_res, batch_size=2, rng_key=random.PRNGKey(1))
    bcs_ds = BCSGen(u, y_ic, s_ic, y_bc, s_bc, batch_size=2, rng_key=random.PRNGKey(2))
    inputs_res, outputs_res = res_ds[0]
    u_b, y_b = inputs_res
    assert u_b.shape == (2, 80) and y_b.shape == (2, 100, 2)
    assert outputs_res.shape == (2, 100, 1)
    inputs_bcs, outputs_bcs = bcs_ds[0]
    u_b2, y_ic_b, y_bc_b = inputs_bcs
    s_ic_b, s_bc_b = outputs_bcs
    assert u_b2.shape == (2, 80) and y_ic_b.shape == (2, 20, 2)
    assert y_bc_b.shape == (2, 20, 2)
    assert s_ic_b.shape == (2, 20, 1) and s_bc_b.shape == (2, 20, 1)


@pytest.mark.parametrize("name,kwargs", PERIODIC_CASES + DIRICHLET_CASES)
def test_test_data_generation(name, kwargs):
    spec = get_pde(name, **kwargs)
    u_test, y_test, s_test = spec.generate_test_data(random.PRNGKey(123))
    expected_n = 2 * 21 * (10 + 1)
    assert u_test.shape == (expected_n, 80)
    assert y_test.shape == (expected_n, 2)
    assert s_test.shape == (expected_n,)
    assert jnp.all(jnp.isfinite(s_test)), f"{name} produced non-finite test data"
