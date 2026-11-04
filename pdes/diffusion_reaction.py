"""
1D diffusion-reaction, registered as "diffusion_reaction"; reference by method of lines.

    u_t = D*u_xx + k*u^2 + f(x) on [0, 1] x [0, 1], D = k = 0.01
    u(x, 0) = 0, u(0, t) = u(1, t) = 0
    Input: the forcing f, a zero-mean periodic GRF (gamma=4, tau=5, sigma=625)
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from jax import random, grad, vmap, jit
from functools import partial
from torch.utils import data
from tqdm import trange
from scipy.integrate import solve_ivp

from pdes.base import PDESpec, BCSStructure, register_pde
from pdes._periodic_ic import (
    sample_GRF_coefs,
    eval_GRF_at,
    flatten_test_trajectory,
)


### Training-data generation ###
def generate_one_training_data(
    key, P_ic, P_bc, Q, m, N_modes, gamma, tau, sigma, D, k_react
):
    """
    Arrays for one forcing f: u_sens (m,), y_ic at random (x, 0), y_bc at (0 or 1, t)
    (side chosen at random), y_res at random (x, t), s_ic = s_bc = 0, s_res = f(x).
    """
    k_grf, k_ic, k_bc_t, k_bc_side, k_rx, k_rt = random.split(key, 6)
    coefs = sample_GRF_coefs(k_grf, N_modes, gamma, tau, sigma)

    x_sensors = jnp.linspace(0.0, 1.0, m)
    u_sens = eval_GRF_at(coefs, x_sensors)

    # IC points (t = 0); target = 0.
    x_ic = random.uniform(k_ic, (P_ic,), minval=0.0, maxval=1.0)
    t_ic = jnp.zeros((P_ic,))
    y_ic = jnp.stack([x_ic, t_ic], axis=-1)
    s_ic = jnp.zeros((P_ic, 1))

    # BC points: half at x=0, half at x=1, random t.
    t_bc = random.uniform(k_bc_t, (P_bc,), minval=0.0, maxval=1.0)
    side = random.bernoulli(k_bc_side, p=0.5, shape=(P_bc,))  # bool
    x_bc = jnp.where(side, jnp.ones((P_bc,)), jnp.zeros((P_bc,)))
    y_bc = jnp.stack([x_bc, t_bc], axis=-1)
    s_bc = jnp.zeros((P_bc, 1))

    # Residual / collocation: target is f(x_res), the forcing value.
    x_r = random.uniform(k_rx, (Q,), minval=0.0, maxval=1.0)
    t_r = random.uniform(k_rt, (Q,), minval=0.0, maxval=1.0)
    y_res = jnp.stack([x_r, t_r], axis=-1)
    s_res = eval_GRF_at(coefs, x_r)[:, None]

    return u_sens, y_ic, s_ic, y_bc, s_bc, y_res, s_res


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
    D=0.01,
    k_react=0.01,
    N_modes=None,
):
    """generate_one_training_data for N forcings, stacked; D and k_react are unused."""
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
        D=D,
        k_react=k_react,
    )
    u, y_ic, s_ic, y_bc, s_bc, y_res, s_res = vmap(fn)(keys)

    u = jnp.float32(u)
    y_ic = jnp.float32(y_ic)
    s_ic = jnp.float32(s_ic)
    y_bc = jnp.float32(y_bc)
    s_bc = jnp.float32(s_bc)
    y_res = jnp.float32(y_res)
    s_res = jnp.float32(s_res)
    jax.config.update("jax_enable_x64", False)
    return u, y_ic, s_ic, y_bc, s_bc, y_res, s_res


### Test-data generation  (numpy method-of-lines via scipy.solve_ivp) ###
def _diffusion_reaction_mol(
    f_vals, tspan, D, k_react, x_grid=None, rtol=1e-6, atol=1e-8
):
    """
    Solve u_t = D*u_xx + k_react*u^2 + f with zero IC/BCs on x_grid (method of lines,
    central differences, LSODA); returns (len(tspan), len(x_grid)) incl. boundaries.
    """
    if x_grid is None:
        n = f_vals.shape[0]
        x_grid = np.linspace(0.0, 1.0, n)
    n = x_grid.shape[0]
    h = x_grid[1] - x_grid[0]

    # Solve only on interior points; boundaries are pinned to zero.
    f_int = f_vals[1:-1]
    n_int = n - 2
    inv_h2 = 1.0 / h**2

    def rhs(t, u_int):
        # second derivative via centered diff with u(0)=u(end)=0
        u_left = np.concatenate([[0.0], u_int[:-1]])
        u_right = np.concatenate([u_int[1:], [0.0]])
        u_xx = (u_right - 2.0 * u_int + u_left) * inv_h2
        return D * u_xx + k_react * u_int**2 + f_int

    sol = solve_ivp(
        rhs,
        t_span=(tspan[0], tspan[-1]),
        y0=np.zeros(n_int),
        t_eval=tspan,
        method="LSODA",
        rtol=rtol,
        atol=atol,
    )
    if not sol.success:
        raise RuntimeError(f"diffusion-reaction solve_ivp failed: {sol.message}")

    # Re-pad with boundary zeros.
    out = np.zeros((len(tspan), n))
    out[:, 1:-1] = sol.y.T
    return out


def generate_one_test_data(
    seed_int, m, nn, steps, D, k_react, N_modes, gamma, tau, sigma
):
    """One test trajectory: sample f, solve on the nn-point grid, flatten."""
    key = random.PRNGKey(int(seed_int))
    coefs = sample_GRF_coefs(key, N_modes, gamma, tau, sigma)

    x_sensors = np.linspace(0.0, 1.0, m)
    x_eval = np.linspace(0.0, 1.0, nn)
    tspan = np.linspace(0.0, 1.0, steps + 1)

    u_sens = np.asarray(eval_GRF_at(coefs, jnp.asarray(x_sensors)))
    f_eval = np.asarray(eval_GRF_at(coefs, jnp.asarray(x_eval)))

    traj_eval = _diffusion_reaction_mol(
        f_eval,
        tspan,
        D=D,
        k_react=k_react,
        x_grid=x_eval,
    )

    return flatten_test_trajectory(u_sens, traj_eval, x_eval, tspan)


def generate_test_data(
    key,
    N,
    m,
    nn,
    D=0.01,
    k_react=0.01,
    steps=100,
    gamma=4.0,
    tau=5.0,
    sigma=25.0**2,
    N_modes=None,
):
    if N_modes is None:
        N_modes = max(128, m // 2)

    seeds = np.asarray(random.randint(key, (N,), 0, 2**31 - 1))
    u_list, y_list, s_list = [], [], []
    for j in trange(N, desc="DiffReact test trajectories"):
        ut, yt, st = generate_one_test_data(
            int(seeds[j]),
            m,
            nn,
            steps,
            D,
            k_react,
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


### Dataset classes  (Dirichlet-zero variant) ###
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


class DirichletBCSDataGenerator(data.Dataset):
    """BCS batches ((u, y_ic, y_bc), (s_ic, s_bc)) of batch_size random functions."""

    def __init__(
        self, u, y_ic, s_ic, y_bc, s_bc, batch_size=32, rng_key=random.PRNGKey(1234)
    ):
        self.u = u
        self.y_ic, self.s_ic = y_ic, s_ic
        self.y_bc, self.s_bc = y_bc, s_bc
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
            (self.u[idx], self.y_ic[idx], self.y_bc[idx]),
            (self.s_ic[idx], self.s_bc[idx]),
        )


### Dirichlet-zero BCS loss ###
def bcs_loss_fn(model, params, batch):
    """Loss MSE(pred(y_ic) - s_ic) + MSE(pred(y_bc) - s_bc); all targets are zero."""
    inputs, outputs = batch
    u, y_ic, y_bc = inputs
    s_ic, s_bc = outputs

    def single_function_loss(u_i, y_ic_i, y_bc_i, s_ic_i, s_bc_i):
        def eval_point(y_pt):
            return model.operator_net(params, u_i, y_pt[0], y_pt[1])

        s_pred_ic = vmap(eval_point)(y_ic_i)
        s_pred_bc = vmap(eval_point)(y_bc_i)
        loss_ic = jnp.mean((s_ic_i.flatten() - s_pred_ic) ** 2)
        loss_bc = jnp.mean((s_bc_i.flatten() - s_pred_bc) ** 2)
        return loss_ic + loss_bc

    losses = vmap(single_function_loss)(u, y_ic, y_bc, s_ic, s_bc)
    return jnp.mean(losses)


### PDE-spec contract ###
def make_residual_fn(D, k_react):
    """Residual s_t - D*s_xx - k_react*s^2 (no forcing; its target s_res is f(x))."""

    def residual_fn(operator_closure, u, x, t):
        def s_fn(x_, t_):
            return operator_closure(x_, t_)

        s = s_fn(x, t)
        s_t = grad(s_fn, argnums=1)(x, t)
        s_xx = grad(grad(s_fn, argnums=0), argnums=0)(x, t)
        return s_t - D * s_xx - k_react * s**2

    return residual_fn


@register_pde("diffusion_reaction")
def build_spec(
    D=0.01,
    k_react=0.01,
    gamma=4.0,
    tau=5.0,
    sigma=25.0**2,
    N_train=80,
    P_ic=200,
    P_bc=100,
    Q_train=2000,
    m=80,
    N_test=100,
    nn=101,
    steps=100,
):
    """Build the diffusion-reaction PDESpec."""
    return PDESpec(
        name="diffusion_reaction",
        coord_range=((0.0, 1.0), (0.0, 1.0)),
        params={"D": D, "k_react": k_react, "gamma": gamma, "tau": tau, "sigma": sigma},
        bcs_structure=BCSStructure(
            kind="dirichlet_zero_ic",
            n_inputs=3,
            description="(u, y_ic, y_bc) -> (s_ic, s_bc)  both zeros",
        ),
        residual_fn=make_residual_fn(D, k_react),
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
            D=D,
            k_react=k_react,
        ),
        generate_test_data=partial(
            generate_test_data,
            N=N_test,
            m=m,
            nn=nn,
            D=D,
            k_react=k_react,
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
            "D": D,
            "k_react": k_react,
        },
        default_test_kwargs={
            "N": N_test,
            "m": m,
            "nn": nn,
            "D": D,
            "k_react": k_react,
            "steps": steps,
            "gamma": gamma,
            "tau": tau,
            "sigma": sigma,
        },
        default_dataset_classes={
            "bcs": DirichletBCSDataGenerator,
            "res": DataGenerator,
        },
        description=(
            "1D diffusion-reaction with homogeneous Dirichlet BCs and "
            "homogeneous IC. Branch input is the forcing function f(x), "
            "sampled from a GRF. Reference solution via method-of-lines "
            "(scipy LSODA)."
        ),
    )
