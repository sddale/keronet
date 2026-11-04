"""
1D linear advection, registered as "advection"; the reference solution is exact.

    u_t + c*u_x = 0 on [0, 1] x [0, 1], periodic in x, c = 1
    Input: u(x, 0) = u0(x), a zero-mean periodic GRF (gamma=4, tau=5, sigma=625)
"""

from __future__ import annotations

import numpy as np
import jax
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


# Test-data generation  (exact: u(x, t) = u0((x - ct) mod 1))
def generate_one_test_data(seed_int, m, nn, steps, c, N_modes, gamma, tau, sigma):
    """One test trajectory: evaluate the GRF IC at (x - c*t) mod 1, then flatten."""
    key = random.PRNGKey(int(seed_int))
    coefs = sample_GRF_coefs(key, N_modes, gamma, tau, sigma)

    x_sensors = np.linspace(0.0, 1.0, m)
    x_eval = np.linspace(0.0, 1.0, nn)
    tspan = np.linspace(0.0, 1.0, steps + 1)

    u_sens = np.asarray(eval_GRF_at(coefs, jnp.asarray(x_sensors)))

    # u(x, t) = u0((x - ct) mod 1)
    # Build the (steps+1) x nn array of back-translated points and evaluate.
    XX = x_eval[None, :]  # (1, nn)
    TT = tspan[:, None]  # (steps+1, 1)
    x_shifted = (XX - c * TT) % 1.0  # (steps+1, nn)
    x_shifted_flat = jnp.asarray(x_shifted.flatten())
    traj_flat = np.asarray(eval_GRF_at(coefs, x_shifted_flat))
    traj_eval = traj_flat.reshape(steps + 1, nn)  # (steps+1, nn)

    return flatten_test_trajectory(u_sens, traj_eval, x_eval, tspan)


def generate_test_data(
    key, N, m, nn, c=1.0, steps=100, gamma=4.0, tau=5.0, sigma=25.0**2, N_modes=None
):
    if N_modes is None:
        N_modes = max(128, m // 2)

    seeds = np.asarray(random.randint(key, (N,), 0, 2**31 - 1))
    u_list, y_list, s_list = [], [], []
    for j in trange(N, desc="Advection test trajectories"):
        ut, yt, st = generate_one_test_data(
            int(seeds[j]),
            m,
            nn,
            steps,
            c,
            N_modes,
            gamma,
            tau,
            sigma,
        )
        u_list.append(ut)
        y_list.append(yt)
        s_list.append(st)

    u_test = np.concatenate(u_list, axis=0)
    y_test = np.concatenate(y_list, axis=0)
    s_test = np.concatenate(s_list, axis=0)
    return jnp.float32(u_test), jnp.float32(y_test), jnp.float32(s_test)


### PDE-spec contract
def make_residual_fn(c):
    """Advection residual: s_t + c*s_x."""

    def residual_fn(operator_closure, u, x, t):
        def s_fn(x_, t_):
            return operator_closure(x_, t_)

        s_t = grad(s_fn, argnums=1)(x, t)
        s_x = grad(s_fn, argnums=0)(x, t)
        return s_t + c * s_x

    return residual_fn


@register_pde("advection")
def build_spec(
    c=1.0,
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
    steps=100,
):
    """Build the advection PDESpec."""
    return PDESpec(
        name="advection",
        coord_range=((0.0, 1.0), (0.0, 1.0)),
        params={"c": c, "gamma": gamma, "tau": tau, "sigma": sigma},
        bcs_structure=BCSStructure(
            kind="periodic_ic",
            n_inputs=4,
            description="(u, y_ic, y_bc_lo, y_bc_hi) -> s_ic",
        ),
        residual_fn=make_residual_fn(c),
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
            c=c,
            steps=steps,
            gamma=gamma,
            tau=tau,
            sigma=sigma,
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
            "c": c,
            "steps": steps,
            "gamma": gamma,
            "tau": tau,
            "sigma": sigma,
        },
        default_dataset_classes={
            "bcs": BCSDataGenerator,
            "res": DataGenerator,
        },
        description=(
            "1D linear advection on a periodic domain. ICs sampled from a "
            "periodic GRF. Reference solution is exact (closed form): "
            "u(x, t) = u0((x - c*t) mod 1)."
        ),
    )
