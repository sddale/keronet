"""
Smoke test of each main architecture's forward pass, losses, predictions, and one optimizer step on Burgers.
"""

import pytest
import jax.numpy as jnp
from jax import random
from jax.flatten_util import ravel_pytree

from pdes import get_pde
from architectures import PIDeepONet, PIT, PINTO, KerONet


SPEC = get_pde("burgers")
M = 80
N = 4
P_IC, P_BC, Q = 50, 10, 200

_key = random.PRNGKey(99)
_key, k_u, k_i, k_l, k_s, k_r = random.split(_key, 6)
U_STAR = random.normal(k_u, (N, M))
Y_IC = jnp.stack([random.uniform(k_i, (N, P_IC)), jnp.zeros((N, P_IC))], axis=-1)
S_IC = random.normal(k_s, (N, P_IC, 1)) * 0.1
T_BC = random.uniform(k_l, (N, P_BC))
Y_BC_LO = jnp.stack([jnp.zeros((N, P_BC)), T_BC], axis=-1)
Y_BC_HI = jnp.stack([jnp.ones((N, P_BC)), T_BC], axis=-1)
BCS_BATCH = ((U_STAR, Y_IC, Y_BC_LO, Y_BC_HI), S_IC)

Y_RES_TEST = random.uniform(k_r, (N, Q, 2))
S_RES_TEST = jnp.zeros((N, Q, 1))
RES_BATCH = ((U_STAR, Y_RES_TEST), S_RES_TEST)


def _make_pi_deeponet():
    return PIDeepONet(
        pde_spec=SPEC,
        n_sensors=M,
        branch_hidden=64,
        branch_layers_n=2,
        trunk_hidden=64,
        trunk_layers_n=2,
        p=64,
        mapping_size=32,
        sigma=1.0,
    )


def _make_pit():
    return PIT(
        pde_spec=SPEC,
        n_sensors=M,
        d_model=64,
        n_heads=2,
        key_dim=32,
        n_blocks=1,
        pe_dim=32,
    )


def _make_pinto():
    return PINTO(
        pde_spec=SPEC,
        n_sensors=M,
        d_model=64,
        n_heads=2,
        key_dim=64,
        n_cross_attn=3,
    )


def _make_keronet():
    return KerONet(
        pde_spec=SPEC,
        n_sensors=M,
        branch_hidden=64,
        branch_layers_n=2,
        trunk_hidden=64,
        trunk_layers_n=2,
        L=8,
        d_k=16,
        d_v=16,
        mapping_size=32,
        sigma=1.0,
    )


@pytest.mark.parametrize(
    "name,factory",
    [
        ("pi_deeponet", _make_pi_deeponet),
        ("pit", _make_pit),
        ("pinto", _make_pinto),
        ("keronet", _make_keronet),
    ],
)
def test_arch_smoke(name, factory):
    model = factory()
    params = model.params

    flat, _ = ravel_pytree(params)
    assert len(flat) > 0

    out = model.operator_net(params, U_STAR[0], 0.5, 0.3)
    assert jnp.isfinite(out), "operator_net produced NaN"

    res = model.residual_net(params, U_STAR[0], 0.5, 0.3)
    assert jnp.isfinite(res), "residual_net produced NaN"

    lbc = model.loss_bcs(params, BCS_BATCH)
    assert jnp.isfinite(lbc) and lbc >= 0

    lr = model.loss_res(params, RES_BATCH)
    assert jnp.isfinite(lr) and lr >= 0

    N_pts = 16
    U_p = U_STAR
    Y_p = random.uniform(random.PRNGKey(0), (N_pts, 2))

    U_in = U_p[:N_pts] if N_pts <= N else jnp.tile(U_p, (4, 1))[:N_pts]
    s_pred = model.predict_s(params, U_in, Y_p)
    assert s_pred.shape == (N_pts,), f"predict_s shape: {s_pred.shape}"
    s_r = model.predict_res(params, U_in, Y_p)
    assert s_r.shape == (N_pts,)

    new_params, _new_opt_state = model.step(
        model.params, model.opt_state, BCS_BATCH, RES_BATCH
    )
    new_flat, _ = ravel_pytree(new_params)
    delta = float(jnp.linalg.norm(new_flat - flat))
    assert jnp.isfinite(jnp.array(delta)), "params became NaN after step"
    assert delta > 0, "params unchanged after step"
