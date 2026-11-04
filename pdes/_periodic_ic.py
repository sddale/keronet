"""
Shared pieces of the 1D periodic-IC PDEs (Burgers, advection, Allen-Cahn).

The input is the IC, a zero-mean periodic GRF on [0, 1] whose mode-k cos/sin amplitudes
are N(0, lam_k^2), lam_k = sqrt(2)*|sigma|*((2*pi*k)^2 + tau^2)^(-gamma/2). Training
uses IC points, periodic (x = 0, x = 1) pairs and zero-target residual points.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from jax import random, vmap, jit
from functools import partial
from torch.utils import data


### GRF  (JAX-native; sample coefs and evaluate anywhere) ###
def sample_GRF_coefs(key, N_modes, gamma, tau, sigma):
    """Sample complex GRF coefficients (2*N_modes + 1,) for modes -N_modes..N_modes."""
    k_arr = jnp.arange(1, N_modes + 1)
    my_eigs = (
        jnp.sqrt(2.0)
        * jnp.abs(sigma)
        * ((2.0 * jnp.pi * k_arr) ** 2 + tau**2) ** (-gamma / 2.0)
    )

    key_a, key_b = random.split(key)
    alpha = my_eigs * random.normal(key_a, (N_modes,))
    beta = my_eigs * random.normal(key_b, (N_modes,))

    a = alpha / 2.0
    b = -beta / 2.0

    c = jnp.concatenate(
        [
            jnp.flip(a) - 1j * jnp.flip(b),
            jnp.array([0.0 + 0j]),
            a + 1j * b,
        ]
    )
    return c


def eval_GRF_at(c, x_points):
    """Evaluate the GRF with coefficients c at x_points (n,) in [0, 1]; returns (n,)."""
    N_modes = (c.shape[0] - 1) // 2
    modes = jnp.arange(-N_modes, N_modes + 1)
    phases = 2.0 * jnp.pi * jnp.outer(x_points - 0.5, modes)
    return (jnp.exp(1j * phases) @ c).real


### Per-sample training-data generation  (JAX-only; no solver call) ###
def generate_one_training_data(
    key, P_ic, P_bc, Q, m, N_modes, gamma, tau, sigma, transform_fn=None
):
    """
    Arrays for one GRF: u_sens (m,), y_ic/s_ic at random (x, 0), y_bc_lo/y_bc_hi at
    (0, t)/(1, t), y_res at random (x, t), s_res = 0; transform_fn maps u_sens and s_ic.
    """
    k_grf, k_ic_x, k_bc_t, k_rx, k_rt = random.split(key, 5)
    c = sample_GRF_coefs(k_grf, N_modes, gamma, tau, sigma)

    x_sensors = jnp.linspace(0.0, 1.0, m)
    u_sens = eval_GRF_at(c, x_sensors)

    x_ic = random.uniform(k_ic_x, (P_ic,), minval=0.0, maxval=1.0)
    t_ic = jnp.zeros((P_ic,))
    y_ic = jnp.stack([x_ic, t_ic], axis=-1)
    s_ic = eval_GRF_at(c, x_ic)[:, None]

    if transform_fn is not None:
        u_sens = transform_fn(u_sens)
        s_ic = transform_fn(s_ic)

    t_bc = random.uniform(k_bc_t, (P_bc,), minval=0.0, maxval=1.0)
    y_bc_lo = jnp.stack([jnp.zeros((P_bc,)), t_bc], axis=-1)
    y_bc_hi = jnp.stack([jnp.ones((P_bc,)), t_bc], axis=-1)

    x_r = random.uniform(k_rx, (Q,), minval=0.0, maxval=1.0)
    t_r = random.uniform(k_rt, (Q,), minval=0.0, maxval=1.0)
    y_res = jnp.stack([x_r, t_r], axis=-1)
    s_res = jnp.zeros((Q, 1))

    return u_sens, y_ic, s_ic, y_bc_lo, y_bc_hi, y_res, s_res


def generate_training_data(
    key,
    N,
    P_ic,
    P_bc,
    Q,
    m,
    gamma=4.0,
    tau=5.0,
    sigma=25.0**2,
    N_modes=None,
    transform_fn=None,
):
    """generate_one_training_data for N GRF samples, stacked on a leading N axis."""
    if N_modes is None:
        N_modes = max(128, m // 2)

    jax.config.update("jax_enable_x64", True)
    keys = random.split(key, N)
    fn = partial(
        generate_one_training_data,
        P_ic=P_ic,
        P_bc=P_bc,
        Q=Q,
        m=m,
        N_modes=N_modes,
        gamma=gamma,
        tau=tau,
        sigma=sigma,
        transform_fn=transform_fn,
    )
    u, y_ic, s_ic, y_bc_lo, y_bc_hi, y_res, s_res = vmap(fn)(keys)

    u = jnp.float32(u)
    y_ic = jnp.float32(y_ic)
    s_ic = jnp.float32(s_ic)
    y_bc_lo = jnp.float32(y_bc_lo)
    y_bc_hi = jnp.float32(y_bc_hi)
    y_res = jnp.float32(y_res)
    s_res = jnp.float32(s_res)
    jax.config.update("jax_enable_x64", False)
    return u, y_ic, s_ic, y_bc_lo, y_bc_hi, y_res, s_res


### RES Data Generation Class for convenience ###
class DataGenerator(data.Dataset):
    """Residual batches ((u, y), s) of batch_size randomly chosen functions."""

    def __init__(self, u, y, s, batch_size=32, rng_key=random.PRNGKey(1234)):
        self.u, self.y, self.s = u, y, s
        self.N = u.shape[0]
        self.batch_size = batch_size
        self.key = rng_key

    def __getitem__(self, index):
        self.key, subkey = random.split(self.key)
        return self.__data_generation(subkey)

    @partial(jit, static_argnums=(0,))
    def __data_generation(self, key):
        idx = random.choice(key, self.N, (self.batch_size,), replace=False)
        return (self.u[idx], self.y[idx]), self.s[idx]


### BCS Data Generation Class for convenience ###
class BCSDataGenerator(data.Dataset):
    """Batches ((u, y_ic, y_bc_lo, y_bc_hi), s_ic) of batch_size random functions."""

    def __init__(
        self,
        u,
        y_ic,
        s_ic,
        y_bc_lo,
        y_bc_hi,
        batch_size=32,
        rng_key=random.PRNGKey(1234),
    ):
        self.u = u
        self.y_ic, self.s_ic = y_ic, s_ic
        self.y_bc_lo, self.y_bc_hi = y_bc_lo, y_bc_hi
        self.N = u.shape[0]
        self.batch_size = batch_size
        self.key = rng_key

    def __getitem__(self, index):
        self.key, subkey = random.split(self.key)
        return self.__data_generation(subkey)

    @partial(jit, static_argnums=(0,))
    def __data_generation(self, key):
        idx = random.choice(key, self.N, (self.batch_size,), replace=False)
        return (
            (self.u[idx], self.y_ic[idx], self.y_bc_lo[idx], self.y_bc_hi[idx]),
            self.s_ic[idx],
        )


### Periodic-IC BCS loss ###
def bcs_loss_fn(model, params, batch):
    """Loss MSE(pred(y_ic) - s_ic) + MSE(pred(y_bc_lo) - pred(y_bc_hi))."""
    inputs, outputs = batch
    u, y_ic, y_bc_lo, y_bc_hi = inputs
    s_ic = outputs

    def single_function_loss(u_i, y_ic_i, y_bc_lo_i, y_bc_hi_i, s_ic_i):
        def eval_point(y_pt):
            return model.operator_net(params, u_i, y_pt[0], y_pt[1])

        s_pred_ic = vmap(eval_point)(y_ic_i)
        s_pred_lo = vmap(eval_point)(y_bc_lo_i)
        s_pred_hi = vmap(eval_point)(y_bc_hi_i)
        loss_ic = jnp.mean((s_ic_i.flatten() - s_pred_ic) ** 2)
        loss_per = jnp.mean((s_pred_lo - s_pred_hi) ** 2)
        return loss_ic + loss_per

    losses = vmap(single_function_loss)(u, y_ic, y_bc_lo, y_bc_hi, s_ic)
    return jnp.mean(losses)


### Test-data flattening helper ###
def flatten_test_trajectory(u_sens, traj_eval, x_eval, tspan):
    """
    Flatten traj_eval (len(tspan), len(x_eval)) to time-major rows: u_test (n, m) of
    repeated sensor values, y_test (n, 2) of (x, t), s_test (n,).
    """
    nn = len(x_eval)
    nt = len(tspan)
    XX, TT = np.meshgrid(x_eval, tspan)
    n_pts = nn * nt
    u_test = np.tile(u_sens, (n_pts, 1))
    y_test = np.stack([XX.flatten(), TT.flatten()], axis=-1)
    s_test = traj_eval.flatten()
    return u_test, y_test, s_test
