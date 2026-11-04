"""
1D viscous Burgers equation, registered as "burgers".

    u_t + u*u_x - visc*u_xx = 0 on [0, 1] x [0, 1], periodic in x, visc = 0.01
    Input: u(x, 0) = u0(x), a zero-mean periodic GRF (gamma=4, tau=5, sigma=625)
    Reference: 2/3-dealiased ETDRK4 on 4096 points, dt = 1e-4.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
from jax import random, grad
from functools import partial
from tqdm import trange

from pdes.base import PDESpec, BCSStructure, register_pde
from pdes._periodic_ic import (
    sample_GRF_coefs,
    eval_GRF_at,
    generate_training_data,
    DataGenerator,
    BCSDataGenerator,
    bcs_loss_fn,
    flatten_test_trajectory,
)


### Burgers solver (numpy, ETDRK4 on periodic Fourier grid) ###
def _burgers_etdrk4(u0_vals, tspan, visc, dt=1e-4):
    """
    Solve u_t + (u^2/2)_x = visc*u_xx on a periodic grid by 2/3-dealiased ETDRK4.

    Returns (len(tspan), len(u0_vals)).
    """
    s = u0_vals.shape[0]
    k = 2.0 * np.pi * np.fft.fftfreq(s, d=1.0 / s)
    ik = 1j * k
    L = -visc * k**2

    kmax = np.max(np.abs(k))
    dealias = np.abs(k) < (2.0 / 3.0) * kmax

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
        v = np.fft.ifft(v_hat).real
        v2_hat = np.fft.fft(v * v) * dealias
        return -0.5 * ik * v2_hat

    v = np.fft.fft(u0_vals)
    t_targets = np.asarray(tspan, dtype=float)
    out = np.zeros((len(t_targets), s))
    out[0] = u0_vals
    t, next_idx, eps = 0.0, 1, 1e-12

    while next_idx < len(t_targets):
        target = t_targets[next_idx]
        remaining = target - t
        if remaining <= eps:
            out[next_idx] = np.fft.ifft(v).real
            next_idx += 1
            continue
        h = min(dt, remaining)
        if h != dt:
            Qh, f1h, f2h, f3h, Eh, E2h = _coefs(h)
        else:
            Qh, f1h, f2h, f3h, Eh, E2h = Q, f1, f2, f3, E, E2
        Nv = N_hat(v)
        a = E2h * v + Qh * Nv
        Na = N_hat(a)
        b_ = E2h * v + Qh * Na
        Nb = N_hat(b_)
        c_ = E2h * a + Qh * (2.0 * Nb - Nv)
        Nc = N_hat(c_)
        v = Eh * v + f1h * Nv + 2.0 * f2h * (Na + Nb) + f3h * Nc
        t += h
        if abs(t - target) <= eps:
            out[next_idx] = np.fft.ifft(v).real
            next_idx += 1

    return out


### Test-data generation ###
def generate_one_test_data(
    seed_int, m, nn, s, steps, visc, N_modes, gamma, tau, sigma, dt
):
    """One test trajectory: sample a GRF IC, solve on s points, interpolate, flatten."""
    key = random.PRNGKey(int(seed_int))
    c = sample_GRF_coefs(key, N_modes, gamma, tau, sigma)

    x_sensors = np.linspace(0.0, 1.0, m)
    x_eval = np.linspace(0.0, 1.0, nn)
    x_fine = np.linspace(0.0, 1.0, s, endpoint=False)
    tspan = np.linspace(0.0, 1.0, steps + 1)

    u_sens = np.asarray(eval_GRF_at(c, jnp.asarray(x_sensors)))
    u0_fine = np.asarray(eval_GRF_at(c, jnp.asarray(x_fine)))

    traj_fine = _burgers_etdrk4(u0_fine, tspan, visc=visc, dt=dt)

    # Periodic interpolation to evaluation grid
    x_fine_ext = np.concatenate([x_fine, [1.0]])
    traj_ext = np.concatenate([traj_fine, traj_fine[:, :1]], axis=1)
    traj_eval = np.stack(
        [np.interp(x_eval, x_fine_ext, row) for row in traj_ext], axis=0
    )

    return flatten_test_trajectory(u_sens, traj_eval, x_eval, tspan)


def generate_test_data(
    key,
    N,
    m,
    nn,
    visc=0.01,
    s=4096,
    steps=100,
    gamma=4.0,
    tau=5.0,
    sigma=25.0**2,
    N_modes=None,
    dt=1e-4,
):
    """N test trajectories as (u_test, y_test, s_test), nn * (steps + 1) rows per IC."""
    if N_modes is None:
        N_modes = s // 2

    seeds = np.asarray(random.randint(key, (N,), 0, 2**31 - 1))
    u_list, y_list, s_list = [], [], []
    for j in trange(N, desc="Burgers test trajectories"):
        ut, yt, st = generate_one_test_data(
            int(seeds[j]),
            m,
            nn,
            s,
            steps,
            visc,
            N_modes,
            gamma,
            tau,
            sigma,
            dt,
        )
        u_list.append(ut)
        y_list.append(yt)
        s_list.append(st)

    u_test = np.concatenate(u_list, axis=0)
    y_test = np.concatenate(y_list, axis=0)
    s_test = np.concatenate(s_list, axis=0)
    return jnp.float32(u_test), jnp.float32(y_test), jnp.float32(s_test)


### PDE-spec contract: residual_fn, build_spec ###
def make_residual_fn(visc):
    """Burgers residual: s_t + s*s_x - visc*s_xx."""

    def residual_fn(operator_closure, u, x, t):
        def s_fn(x_, t_):
            return operator_closure(x_, t_)

        s = s_fn(x, t)
        s_t = grad(s_fn, argnums=1)(x, t)
        s_x = grad(s_fn, argnums=0)(x, t)
        s_xx = grad(grad(s_fn, argnums=0), argnums=0)(x, t)
        return s_t + s * s_x - visc * s_xx

    return residual_fn


@register_pde("burgers")
def build_spec(
    visc=0.01,
    gamma=4.0,
    tau=5.0,
    sigma=25.0**2,
    N_train=80,
    P_ic=200,
    P_bc=50,
    Q_train=2000,
    m=80,
    N_test=100,
    nn=101,
    s=4096,
    steps=100,
    dt=1e-4,
):
    """Build the Burgers PDESpec."""
    return PDESpec(
        name="burgers",
        coord_range=((0.0, 1.0), (0.0, 1.0)),
        params={"visc": visc, "gamma": gamma, "tau": tau, "sigma": sigma},
        bcs_structure=BCSStructure(
            kind="periodic_ic",
            n_inputs=4,
            description="(u, y_ic, y_bc_lo, y_bc_hi) -> s_ic",
        ),
        residual_fn=make_residual_fn(visc),
        bcs_loss_fn=bcs_loss_fn,
        generate_training_data=partial(
            generate_training_data,
            N=N_train,
            P_ic=P_ic,
            P_bc=P_bc,
            Q=Q_train,
            m=m,
            gamma=gamma,
            tau=tau,
            sigma=sigma,
        ),
        generate_test_data=partial(
            generate_test_data,
            N=N_test,
            m=m,
            nn=nn,
            visc=visc,
            s=s,
            steps=steps,
            gamma=gamma,
            tau=tau,
            sigma=sigma,
            dt=dt,
        ),
        default_training_kwargs={
            "N": N_train,
            "P_ic": P_ic,
            "P_bc": P_bc,
            "Q": Q_train,
            "m": m,
            "gamma": gamma,
            "tau": tau,
            "sigma": sigma,
        },
        default_test_kwargs={
            "N": N_test,
            "m": m,
            "nn": nn,
            "visc": visc,
            "s": s,
            "steps": steps,
            "gamma": gamma,
            "tau": tau,
            "sigma": sigma,
            "dt": dt,
        },
        default_dataset_classes={
            "bcs": BCSDataGenerator,
            "res": DataGenerator,
        },
        description=(
            "1D viscous Burgers with periodic BCs. ICs sampled from a "
            "periodic GRF. Reference solution via ETDRK4 with 2/3-rule "
            "dealiasing."
        ),
    )
