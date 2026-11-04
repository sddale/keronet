"""
2D Allen-Cahn equation (the paper's 2D benchmark), registered as "allen_cahn_2d".

    s_t = eps2*(s_xx + s_yy) + s - s^3 on [0, 1]^2 x [0, 1], periodic in x and y, eps2 = 1e-2
    Input: s(x, y, 0) = tanh(2.8 * g / std(g)), g a zero-mean 2D periodic GRF
    Periodicity is supervised by value matching on P_bc = 400 point pairs per pair of x/y faces.
    Reference: 2D ETDRK4 on 96 x 96 points, dt = 5e-3.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from jax import random, grad, vmap, jit
from functools import partial
from torch.utils import data

from pdes.base import PDESpec, BCSStructure, register_pde
from pdes._periodic2d import (
    half_plane_modes,
    sample_GRF2d_coefs,
    eval_GRF2d_at,
    sample_interior_square,
)
from pdes._periodic_ic import BCSDataGenerator

# Pre-tanh IC amplitude: the GRF is normalised to unit std and scaled by IC_AMP before tanh.
IC_AMP = 2.8
# Periodic value-matching pairs per pair of opposite x/y faces.
P_BC = 400


# ============================================================================
# Reference solver: ETDRK4 on a periodic 2D Fourier grid
# ============================================================================


def _allen_cahn_2d_etdrk4(u0, tspan, eps2, dt=1e-4):
    """
    Solve s_t = eps2*(s_xx + s_yy) + s - s^3 on a periodic n x n grid by 2D ETDRK4.

    u0 is (n, n); returns (len(tspan), n, n).
    """
    n = u0.shape[0]
    k1 = 2.0 * np.pi * np.fft.fftfreq(n, d=1.0 / n)
    KX, KY = np.meshgrid(k1, k1, indexing="ij")
    K2 = KX**2 + KY**2
    L = (-eps2 * K2).ravel()

    M = 16
    r = np.exp(1j * np.pi * (np.arange(1, M + 1) - 0.5) / M)

    def _coefs(h):
        LR = h * L[:, None] + r[None, :]
        Q = h * ((np.exp(LR / 2.0) - 1.0) / LR).mean(axis=1).real
        f1 = (
            h
            * ((-4.0 - LR + np.exp(LR) * (4.0 - 3.0 * LR + LR**2)) / LR**3)
            .mean(axis=1)
            .real
        )
        f2 = h * ((2.0 + LR + np.exp(LR) * (-2.0 + LR)) / LR**3).mean(axis=1).real
        f3 = (
            h
            * ((-4.0 - 3.0 * LR - LR**2 + np.exp(LR) * (4.0 - LR)) / LR**3)
            .mean(axis=1)
            .real
        )
        E = np.exp(h * L)
        E2 = np.exp(h * L / 2.0)
        return Q, f1, f2, f3, E, E2

    Q, f1, f2, f3, E, E2 = _coefs(dt)

    def N_hat(v_hat):
        v = np.fft.ifft2(v_hat.reshape(n, n)).real
        nl = v - v**3
        return np.fft.fft2(nl).ravel()

    v = np.fft.fft2(u0).ravel()
    t_targets = np.asarray(tspan, dtype=float)
    out = np.zeros((len(t_targets), n, n))
    out[0] = u0
    t, next_idx, eps_t = 0.0, 1, 1e-12

    while next_idx < len(t_targets):
        target = t_targets[next_idx]
        remaining = target - t
        if remaining <= eps_t:
            out[next_idx] = np.fft.ifft2(v.reshape(n, n)).real
            next_idx += 1
            continue
        h = min(dt, remaining)
        Qh, f1h, f2h, f3h, Eh, E2h = (Q, f1, f2, f3, E, E2) if h == dt else _coefs(h)
        Nv = N_hat(v)
        a = E2h * v + Qh * Nv
        Na = N_hat(a)
        b_ = E2h * v + Qh * Na
        Nb = N_hat(b_)
        c_ = E2h * a + Qh * (2.0 * Nb - Nv)
        Nc = N_hat(c_)
        v = Eh * v + f1h * Nv + 2.0 * f2h * (Na + Nb) + f3h * Nc
        t += h
        if abs(t - target) <= eps_t:
            out[next_idx] = np.fft.ifft2(v.reshape(n, n)).real
            next_idx += 1

    return out


def _eval_grf2d_np(modes, a, b, pts, chunk=20000):
    """NumPy version of eval_GRF2d_at, chunked over points; pts (n, 2) -> (n,)."""
    modes = np.asarray(modes, float)
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    pts = np.asarray(pts, float)
    out = np.empty(len(pts))
    mT = modes.T
    for i in range(0, len(pts), chunk):
        theta = 2.0 * np.pi * (pts[i : i + chunk] @ mT)
        out[i : i + chunk] = np.cos(theta) @ a + np.sin(theta) @ b
    return out


# ============================================================================
# Evaluation grid  (full periodic square)
# ============================================================================


def eval_grid_spacetime(nn, steps):
    """
    Time-major (x, y, t) points of an nn x nn grid on [0, 1)^2 at steps + 1 times.

    Returns (points (n_space * n_time, 3), n_space, n_time).
    """
    g = np.linspace(0.0, 1.0, nn, endpoint=False)
    X, Y = np.meshgrid(g, g, indexing="ij")
    space = np.stack([X.ravel(), Y.ravel()], axis=-1)  # (nn^2, 2)
    tvals = np.linspace(0.0, 1.0, steps + 1)
    pts = np.concatenate(
        [np.concatenate([space, np.full((len(space), 1), t)], axis=1) for t in tvals],
        axis=0,
    )
    return pts.astype(np.float32), space.shape[0], len(tvals)


# ============================================================================
# Training-data generation
# ============================================================================


def _grf_field_std(modes, gamma, tau, sigma):
    """Pointwise std of the 2D GRF, sqrt(sum_k lam_k^2), from the mode amplitudes."""
    k2 = jnp.sum(jnp.asarray(modes) ** 2, axis=1)
    lam2 = (
        jnp.sqrt(2.0)
        * jnp.abs(sigma)
        * ((2.0 * jnp.pi) ** 2 * k2 + tau**2) ** (-gamma / 2.0)
    ) ** 2
    return jnp.sqrt(jnp.sum(lam2))


def _grf_field_std_np(modes, gamma, tau, sigma):
    k2 = np.sum(np.asarray(modes, float) ** 2, axis=1)
    lam2 = (
        np.sqrt(2.0) * abs(sigma) * ((2.0 * np.pi) ** 2 * k2 + tau**2) ** (-gamma / 2.0)
    ) ** 2
    return float(np.sqrt(np.sum(lam2)))


def _sensor_grid(m):
    """Sensors (m, 2) on a sqrt(m) x sqrt(m) grid on [0, 1)^2; m must be a square."""
    r = int(round(np.sqrt(m)))
    if r * r != m:
        raise ValueError(
            f"2D Allen-Cahn expects square m; got m={m} (use 64, 81, 100...)"
        )
    g = np.linspace(0.0, 1.0, r, endpoint=False)
    X, Y = np.meshgrid(g, g, indexing="ij")
    return np.stack([X.ravel(), Y.ravel()], axis=-1).astype(np.float32)


def _face_pairs(key, n):
    """
    Periodic pairs (lo, hi), each (2n, 3): n on x = 0 vs x = 1, then n on y = 0 vs
    y = 1, each pair sharing its other two (uniform random) coordinates.
    """
    k1, k2 = random.split(key)
    a = random.uniform(k1, (n, 2))          # (y, t) on the x-faces
    b = random.uniform(k2, (n, 2))          # (x, t) on the y-faces
    zeros, ones = jnp.zeros((n, 1)), jnp.ones((n, 1))
    lo = jnp.concatenate([jnp.concatenate([zeros, a], 1),
                          jnp.concatenate([b[:, :1], zeros, b[:, 1:]], 1)], 0)
    hi = jnp.concatenate([jnp.concatenate([ones, a], 1),
                          jnp.concatenate([b[:, :1], ones, b[:, 1:]], 1)], 0)
    return lo, hi


def bcs_loss_fn(model, params, batch):
    """IC MSE plus periodic-pair mismatch MSE; batch = ((u, y_ic, lo, hi), s_ic)."""
    (u, y_ic, y_lo, y_hi), s_ic = batch

    def single(u_i, y_ic_i, lo_i, hi_i, s_i):
        f = lambda pt: model.operator_net(params, u_i, pt[0], pt[1], pt[2])
        loss_ic = jnp.mean((s_i.flatten() - vmap(f)(y_ic_i)) ** 2)
        loss_per = jnp.mean((vmap(f)(lo_i) - vmap(f)(hi_i)) ** 2)
        return loss_ic + loss_per

    return jnp.mean(vmap(single)(u, y_ic, y_lo, y_hi, s_ic))


def _one_training(key, modes, sensors, P_ic, Q_pool, gamma, tau, sigma, ic_amp):
    """generate_one_training_data with ic_amp as an argument."""
    k_grf, k_ic, k_rx = random.split(key, 3)
    a, b = sample_GRF2d_coefs(k_grf, modes, gamma, tau, sigma)
    scale = ic_amp / _grf_field_std(modes, gamma, tau, sigma)
    u_sens = jnp.tanh(scale * eval_GRF2d_at(modes, a, b, sensors))
    xy_ic = random.uniform(k_ic, (P_ic, 2))
    y_ic = jnp.concatenate([xy_ic, jnp.zeros((P_ic, 1))], axis=1)
    s_ic = jnp.tanh(scale * eval_GRF2d_at(modes, a, b, xy_ic))[:, None]
    xy_r = sample_interior_square(random.fold_in(k_rx, 0), Q_pool)
    t_r = random.uniform(random.fold_in(k_rx, 1), (Q_pool, 1))
    return u_sens, y_ic, s_ic, jnp.concatenate([xy_r, t_r], axis=1), jnp.zeros((Q_pool, 1))


def generate_training_data(key, N, P_ic, Q_pool, m, gamma=4.0, tau=5.0, sigma=25.0 ** 2, n_modes=16,
                           chunk=20, ic_amp=IC_AMP, P_bc=P_BC, **_ignored):
    """
    (u, y_ic, s_ic, y_bc_lo, y_bc_hi, y_res, s_res) for N functions.

    y_bc_lo and y_bc_hi are (N, 2 * P_bc, 3) periodic pairs (see _face_pairs).
    """
    jax.config.update("jax_enable_x64", True)
    modes = half_plane_modes(n_modes)
    sensors = jnp.asarray(_sensor_grid(m), dtype=jnp.float64)
    keys = random.split(key, N)
    fn = partial(_one_training, modes=modes, sensors=sensors, P_ic=P_ic, Q_pool=Q_pool,
                 gamma=gamma, tau=tau, sigma=sigma, ic_amp=ic_amp)
    pieces = [vmap(fn)(keys[i:i + chunk]) for i in range(0, N, chunk)]
    u, y_ic, s_ic, y_res, s_res = (jnp.float32(jnp.concatenate([p[j] for p in pieces], 0)) for j in range(5))
    jax.config.update("jax_enable_x64", False)
    fkeys = random.split(random.fold_in(key, 7919), N)
    lo, hi = vmap(lambda k: _face_pairs(k, P_bc))(fkeys)
    return u, y_ic, s_ic, jnp.float32(lo), jnp.float32(hi), y_res, s_res


# ============================================================================
# Test-data generation  (spectral ETDRK4, parallel over CPU cores)
# ============================================================================


def _solve_one_test(args):
    """
    Test worker: solve one IC on the s_fine grid and bilinearly interpolate to `space`.

    Returns (u_sens (m,), trajectory (n_time, n_space)).
    """
    modes, a, b, sensors, space, tvals, eps2, s_fine, dt, scale = args
    # fine periodic IC grid
    g = np.linspace(0.0, 1.0, s_fine, endpoint=False)
    X, Y = np.meshgrid(g, g, indexing="ij")
    u0 = np.tanh(
        scale
        * _eval_grf2d_np(modes, a, b, np.stack([X.ravel(), Y.ravel()], -1)).reshape(
            s_fine, s_fine
        )
    )
    traj = _allen_cahn_2d_etdrk4(u0, tvals, eps2=eps2, dt=dt)  # (nt, s_fine, s_fine)

    # bilinear-periodic interpolation of each frame to the eval spatial grid
    n_eval = int(round(np.sqrt(space.shape[0])))
    gx = np.linspace(0.0, 1.0, n_eval, endpoint=False)
    xi = (space[:, 0] * s_fine) % s_fine
    yi = (space[:, 1] * s_fine) % s_fine
    x0 = np.floor(xi).astype(int) % s_fine
    y0 = np.floor(yi).astype(int) % s_fine
    x1 = (x0 + 1) % s_fine
    y1 = (y0 + 1) % s_fine
    fx = (xi - np.floor(xi))[None, :]
    fy = (yi - np.floor(yi))[None, :]
    F = traj.reshape(traj.shape[0], -1)
    idx = lambda ix, iy: ix * s_fine + iy
    interp = (
        F[:, idx(x0, y0)] * (1 - fx) * (1 - fy)
        + F[:, idx(x1, y0)] * fx * (1 - fy)
        + F[:, idx(x0, y1)] * (1 - fx) * fy
        + F[:, idx(x1, y1)] * fx * fy
    )  # (nt, n_space)

    u_sens = np.tanh(scale * _eval_grf2d_np(modes, a, b, sensors)).astype(np.float32)
    return u_sens, interp.astype(np.float32)


def generate_test_data(key, N, m, nn=64, steps=100, eps2=1e-2, s_fine=96, dt=5e-3, gamma=4.0, tau=5.0,
                       sigma=25.0 ** 2, n_modes=16, ic_amp=IC_AMP, workers=None, **_ignored):
    """generate_test_data with ic_amp as an argument; always solves in a process pool."""
    import os
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    modes = half_plane_modes(n_modes)
    modes_np = np.asarray(modes)
    sensors = _sensor_grid(m)
    scale = ic_amp / _grf_field_std_np(modes_np, gamma, tau, sigma)
    pts, n_space, n_time = eval_grid_spacetime(nn, steps)
    space = pts[:n_space, :2]
    tvals = np.linspace(0.0, 1.0, steps + 1)
    keys = random.split(key, N)
    jobs = []
    for i in range(N):
        a, b = sample_GRF2d_coefs(keys[i], modes, gamma, tau, sigma)
        jobs.append((modes_np, np.asarray(a), np.asarray(b), sensors, space, tvals, eps2, s_fine, dt, scale))
    workers = workers or min(os.cpu_count() or 1, 8)
    guard = {"JAX_PLATFORMS": "cpu", "CUDA_VISIBLE_DEVICES": ""}
    saved = {k: os.environ.get(k) for k in guard}
    os.environ.update(guard)
    try:
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
            results = list(ex.map(_solve_one_test, jobs, chunksize=1))
    finally:
        for k, v in saved.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
    u_all, y_all, s_all = [], [], []
    for u_sens, interp in results:
        y_all.append(pts)
        s_all.append(interp.reshape(-1))
        u_all.append(np.repeat(u_sens[None, :], pts.shape[0], axis=0))
    return (jnp.float32(np.concatenate(u_all, 0)), jnp.float32(np.concatenate(y_all, 0)),
            jnp.float32(np.concatenate(s_all, 0)))


# ============================================================================
# Dataset classes
# ============================================================================


class DataGenerator(data.Dataset):
    """
    Residual batches ((u, y), s) of batch_size random functions, n_colloc_batch pool
    points each (drawn with replacement; the whole pool if n_colloc_batch >= its size).
    """

    def __init__(
        self, u, y, s, batch_size=32, rng_key=random.PRNGKey(1234), n_colloc_batch=2000
    ):
        self.u, self.y, self.s = u, y, s
        self.N = u.shape[0]
        self.n_pool = y.shape[1]
        self.n_colloc_batch = min(n_colloc_batch, self.n_pool)
        self.batch_size = batch_size
        self.key = rng_key

    def __getitem__(self, index):
        self.key, subkey = random.split(self.key)
        return self.__data_generation(subkey)

    @partial(jit, static_argnums=(0,))
    def __data_generation(self, key):
        k_fn, k_pt = random.split(key)
        idx = random.choice(k_fn, self.N, (self.batch_size,), replace=False)
        if self.n_colloc_batch == self.n_pool:
            return (self.u[idx], self.y[idx]), self.s[idx]
        pidx = random.randint(
            k_pt, (self.batch_size, self.n_colloc_batch), 0, self.n_pool
        )
        y = jnp.take_along_axis(self.y[idx], pidx[..., None], axis=1)
        s = jnp.take_along_axis(self.s[idx], pidx[..., None], axis=1)
        return (self.u[idx], y), s


# ============================================================================
# PDE-spec contract
# ============================================================================


def make_residual_fn(eps2):
    """Allen-Cahn residual: s_t - eps2*(s_xx + s_yy) - (s - s^3)."""

    def residual_fn(operator_closure, u, x, y, t):
        def s_fn(x_, y_, t_):
            return operator_closure(x_, y_, t_)

        s = s_fn(x, y, t)
        s_t = grad(s_fn, argnums=2)(x, y, t)
        s_xx = grad(grad(s_fn, argnums=0), argnums=0)(x, y, t)
        s_yy = grad(grad(s_fn, argnums=1), argnums=1)(x, y, t)
        return s_t - eps2 * (s_xx + s_yy) - (s - s**3)

    return residual_fn


@register_pde("allen_cahn_2d")
def build_spec(
    eps2=1e-2,
    gamma=4.0,
    tau=5.0,
    sigma=25.0**2,
    N_train=3200,
    P_ic=1600,
    Q_pool=3000,
    Q_batch=3000,
    m=81,
    N_test=100,
    nn=64,
    steps=100,
    s_fine=96,
    dt=5e-3,
    n_modes=16,
    ic_amp=IC_AMP,
    P_bc=P_BC,
):
    """
    Build the 2D Allen-Cahn PDESpec (m must be a perfect square).

    Each iteration uses Q_batch of each function's Q_pool residual points.
    """
    training_kwargs = {
        "N": N_train,
        "P_ic": P_ic,
        "Q_pool": Q_pool,
        "m": m,
        "gamma": gamma,
        "tau": tau,
        "sigma": sigma,
        "n_modes": n_modes,
        "ic_amp": ic_amp,
        "P_bc": P_bc,
    }
    test_kwargs = {
        "N": N_test,
        "m": m,
        "nn": nn,
        "steps": steps,
        "eps2": eps2,
        "s_fine": s_fine,
        "dt": dt,
        "gamma": gamma,
        "tau": tau,
        "sigma": sigma,
        "n_modes": n_modes,
        # Full periodic grid: n_space * n_time points.
        "pts_per_traj": int((nn * nn) * (steps + 1)),
        "ic_amp": ic_amp,
    }
    return PDESpec(
        name="allen_cahn_2d",
        coord_range=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),  # (x, y, t)
        params={
            "eps2": eps2,
            "gamma": gamma,
            "tau": tau,
            "sigma": sigma,
            "n_modes": n_modes,
            "Q_pool": int(Q_pool),
            "Q_batch": int(Q_batch),
            "sensor_layout": "square_grid_2d",
            "ic_amp": float(ic_amp),
            "P_bc": int(P_bc),
            "bc": "periodic_value_matching",
        },
        bcs_structure=BCSStructure(kind="initial_condition_2d_periodic", n_inputs=4,
                                   description="(u, y_ic, y_bc_lo, y_bc_hi) -> s_ic"),
        residual_fn=make_residual_fn(eps2),
        bcs_loss_fn=bcs_loss_fn,
        generate_training_data=partial(generate_training_data, **training_kwargs),
        generate_test_data=partial(
            generate_test_data, **{k: v for k, v in test_kwargs.items() if k != "pts_per_traj"}
        ),
        default_training_kwargs=training_kwargs,
        default_test_kwargs=test_kwargs,
        default_dataset_classes={
            "bcs": BCSDataGenerator,
            "res": partial(DataGenerator, n_colloc_batch=Q_batch),
        },
        description=f"2D Allen-Cahn (ic_amp={ic_amp}), periodic BCs supervised by value matching.",
    )
