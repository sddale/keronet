"""
Smoke tests for pdes/allen_cahn_2d.py: spec, training data, periodic face pairs, and bcs_loss_fn.
"""

import jax.numpy as jnp
import numpy as np
from jax import random

from pdes import get_pde

N, P_IC, Q = 4, 20, 50


def _spec():
    return get_pde("allen_cahn_2d", N_train=N, P_ic=P_IC, Q_pool=Q, Q_batch=Q)


def test_spec():
    spec = _spec()
    assert spec.params["ic_amp"] == 2.8
    assert spec.params["P_bc"] == 400
    assert spec.bcs_structure.kind == "initial_condition_2d_periodic"
    assert spec.bcs_structure.n_inputs == 4


def test_training_data():
    u, y_ic, s_ic, lo, hi, y_res, s_res = _spec().generate_training_data(random.PRNGKey(0))
    assert u.shape == (N, 81)
    assert y_ic.shape == (N, P_IC, 3) and s_ic.shape == (N, P_IC, 1)
    assert lo.shape == hi.shape == (N, 800, 3)
    assert y_res.shape == (N, Q, 3) and s_res.shape == (N, Q, 1)
    assert jnp.allclose(y_ic[..., 2], 0.0)
    assert jnp.allclose(s_res, 0.0)
    assert jnp.all(jnp.abs(u) <= 1.0) and jnp.all(jnp.abs(s_ic) <= 1.0)
    assert float(jnp.mean(jnp.abs(s_ic) > 0.9)) > 0.3


def test_face_pairs():
    _, _, _, lo, hi, _, _ = _spec().generate_training_data(random.PRNGKey(1))
    lo, hi = np.asarray(lo), np.asarray(hi)
    xf, yf = slice(0, 400), slice(400, 800)
    assert np.all(lo[:, xf, 0] == 0.0) and np.all(hi[:, xf, 0] == 1.0)
    assert np.array_equal(lo[:, xf, 1:], hi[:, xf, 1:])
    assert np.all(lo[:, yf, 1] == 0.0) and np.all(hi[:, yf, 1] == 1.0)
    assert np.array_equal(lo[:, yf][..., [0, 2]], hi[:, yf][..., [0, 2]])


class ToyModel:
    """Periodic in (x, y) except for the d * x * t term, which vanishes at t = 0."""

    def operator_net(self, params, u, x, y, t):
        return (params["a"] * jnp.mean(u) + params["b"] * jnp.sin(2 * jnp.pi * x) * jnp.cos(2 * jnp.pi * y)
                + params["d"] * x * t)


def test_bcs_loss_periodic_term():
    spec = _spec()
    u, y_ic, s_ic, lo, hi, _, _ = spec.generate_training_data(random.PRNGKey(2))
    batch = ((u, y_ic, lo, hi), s_ic)
    base = {"a": jnp.float32(0.3), "b": jnp.float32(0.2), "d": jnp.float32(0.0)}
    d = 0.7
    loss0 = float(spec.bcs_loss_fn(ToyModel(), base, batch))
    loss1 = float(spec.bcs_loss_fn(ToyModel(), {**base, "d": jnp.float32(d)}, batch))
    lo, hi = np.asarray(lo), np.asarray(hi)
    expected = np.mean(np.mean((d * lo[..., 2] * (lo[..., 0] - hi[..., 0])) ** 2, axis=1))
    np.testing.assert_allclose(loss1 - loss0, expected, rtol=1e-4, atol=1e-6)
