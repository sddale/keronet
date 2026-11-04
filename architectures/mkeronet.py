"""
mKerONet: KerONet readout ablation with scalar values (d_v = 1) and no post-readout MLP.

    s(u, y) = sum_l softmax_l(q*K_l/sqrt(d_k)) * v_l + bias
Branch, trunk, Fourier features and training are as in KerONet.
"""

import jax
import jax.numpy as jnp
from jax import random, grad, vmap, jit
import optax
from functools import partial
from tqdm import trange


# ============================================================================
# Primitives  (inlined to keep this file self-contained)
# ============================================================================


def dense_init(key, fan_in, fan_out):
    lim = jnp.sqrt(6.0 / (fan_in + fan_out))
    W = random.uniform(key, (fan_in, fan_out), minval=-lim, maxval=lim)
    b = jnp.zeros(fan_out)
    return W, b


def dense(W, b, x):
    return x @ W + b


def mlp_gelu_init(key, layer_sizes):
    params = []
    for i in range(len(layer_sizes) - 1):
        key, kd = random.split(key)
        params.append(dense_init(kd, layer_sizes[i], layer_sizes[i + 1]))
    return params


def mlp_gelu_apply(params, x):
    """GELU after each hidden layer; final layer linear."""
    for W, b in params[:-1]:
        x = jax.nn.gelu(dense(W, b, x))
    W, b = params[-1]
    return dense(W, b, x)


# -- Branch  (sensor values -> L x (d_k + d_v) key/value dictionary) ----------


def branch_init(key, n_sensors, hidden_dim, num_layers, L, d_k, d_v):
    """Branch GELU MLP: n_sensors -> hidden_dim x num_layers -> L*(d_k + d_v)."""
    out_dim = L * (d_k + d_v)
    layer_sizes = [n_sensors] + [hidden_dim] * num_layers + [out_dim]
    return mlp_gelu_init(key, layer_sizes)


def branch_apply(params, u, L, d_k, d_v):
    """Map sensor values u (m,) to keys (L, d_k) and values (L, d_v)."""
    flat = mlp_gelu_apply(params, u)  # (L*(d_k+d_v),)
    kv = flat.reshape(L, d_k + d_v)
    keys = kv[:, :d_k]
    vals = kv[:, d_k:]
    return keys, vals


# -- Trunk  (Fourier-lift (x, t) -> q in R^{d_q}) ------------------------------


def trunk_init(key, trunk_in, hidden_dim, num_layers, d_q, mapping_size):
    return mlp_gelu_init(key, [2 * mapping_size] + [hidden_dim] * num_layers + [d_q])


def trunk_apply(params, B_fourier, coord_min, coord_max, y):
    y_norm = (y - coord_min) / (coord_max - coord_min + 1e-8)
    proj = 2.0 * jnp.pi * y_norm @ B_fourier
    ff = jnp.concatenate([jnp.sin(proj), jnp.cos(proj)])
    return mlp_gelu_apply(params, ff)


# ============================================================================
# Softmax readout
# ============================================================================


def softmax_readout(q, keys, values, bias):
    """Readout without post-MLP: s = sum_l softmax_l(K_l*q/sqrt(d_k)) * v_l + bias.

    q: (d_k,), keys: (L, d_k), values: (L, 1); returns a scalar.
    """
    d_k = keys.shape[1]
    logits = (keys @ q) / jnp.sqrt(d_k)  # (L,)
    attn = jax.nn.softmax(logits)  # (L,)
    return (attn @ values)[0] + bias  # scalar


# ============================================================================
# mKerONet
# ============================================================================


class MKerONet:
    """mKerONet model: KerONet with scalar values (d_v = 1) and no post-readout MLP."""

    def __init__(
        self,
        pde_spec,
        n_sensors,
        branch_hidden=128,
        branch_layers_n=3,
        trunk_hidden=128,
        trunk_layers_n=3,
        L=8,
        d_k=16,
        mapping_size=64,
        sigma=0.5,
        weight_decay=1e-4,
        init_seed=0,
    ):
        """Initialize params and a placeholder AdamW optimizer (exponential LR decay).

        Args:
            L: number of key/value slots (the value dim is fixed at 1).
            mapping_size, sigma: count and scale of the trunk's random Fourier features.
        """
        trunk_in = len(pde_spec.coord_range)
        d_v = 1  # the ablation: scalar values, not a d_v-dim space

        self.pde_spec = pde_spec
        self._hparams = {
            "n_sensors": n_sensors,
            "branch_hidden": branch_hidden,
            "branch_layers_n": branch_layers_n,
            "trunk_hidden": trunk_hidden,
            "trunk_layers_n": trunk_layers_n,
            "L": L,
            "d_k": d_k,
            "d_v": d_v,
            "mapping_size": mapping_size,
            "sigma": sigma,
            "weight_decay": weight_decay,
            "init_seed": init_seed,
            "pde": pde_spec.name,
        }
        self.L = L
        self.d_k = d_k
        self.d_v = d_v

        # Coord range comes from the PDE spec
        coord_range = pde_spec.coord_range
        self.coord_min = jnp.array([r[0] for r in coord_range])
        self.coord_max = jnp.array([r[1] for r in coord_range])

        key = random.PRNGKey(init_seed)
        key, kb, kt, kf, kp = random.split(key, 5)

        self.B_fourier = random.normal(kf, (trunk_in, mapping_size)) * sigma

        branch_params = branch_init(
            kb, n_sensors, branch_hidden, branch_layers_n, L, d_k, d_v
        )
        trunk_params = trunk_init(
            kt, trunk_in, trunk_hidden, trunk_layers_n, d_k, mapping_size
        )
        bias = jnp.zeros(())

        params = {
            "branch": branch_params,
            "trunk": trunk_params,
            "bias": bias,
        }

        # AdamW with decay mask on 2D+ tensors only (matches our convention)
        def decay_mask(p):
            return jax.tree_util.tree_map(lambda x: x.ndim >= 2, p)

        schedule = optax.exponential_decay(
            init_value=1e-3, transition_steps=2000, decay_rate=0.9
        )
        self.optimizer = optax.adamw(
            learning_rate=schedule,
            weight_decay=weight_decay,
            mask=decay_mask,
        )
        self.opt_state = self.optimizer.init(params)
        self.params = params

        self.loss_log = []
        self.loss_bcs_log = []
        self.loss_res_log = []

    # -- Forward --------------------------------------------------------------

    def operator_net(self, params, u, *coords):
        keys, values = branch_apply(params["branch"], u, self.L, self.d_k, self.d_v)
        y = jnp.array(coords)
        q = trunk_apply(
            params["trunk"], self.B_fourier, self.coord_min, self.coord_max, y
        )
        return softmax_readout(q, keys, values, params["bias"])

    # -- PDE residual (delegated to pde_spec; branch is cached per-call) -----

    def residual_net(self, params, u, *coords):
        # Cache the K/V dictionary once for this (params, u). All derivative
        # calls inside pde_spec.residual_fn reuse this cache.
        keys, values = branch_apply(params["branch"], u, self.L, self.d_k, self.d_v)
        bias = params["bias"]

        def operator_closure(*c_):
            y = jnp.array(c_)
            q = trunk_apply(
                params["trunk"],
                self.B_fourier,
                self.coord_min,
                self.coord_max,
                y,
            )
            return softmax_readout(q, keys, values, bias)

        return self.pde_spec.residual_fn(operator_closure, u, *coords)

    # -- Losses ----------------------------------------------------------------

    def loss_bcs(self, params, batch):
        """Initial/boundary-condition loss, computed by pde_spec.bcs_loss_fn."""
        return self.pde_spec.bcs_loss_fn(self, params, batch)

    def loss_res(self, params, batch):
        """Mean squared residual error over a batch ((u, y), s).

        u: (N, m), y: (N, Q, d), s: (N, Q, 1); one branch evaluation per function.
        """
        inputs, outputs = batch
        u, y = inputs

        def single_function_loss(u_i, y_i, s_i):
            # Cache K/V once per function
            keys, values = branch_apply(
                params["branch"], u_i, self.L, self.d_k, self.d_v
            )
            bias = params["bias"]

            def operator_closure(*c_):
                y_pt = jnp.array(c_)
                q = trunk_apply(
                    params["trunk"],
                    self.B_fourier,
                    self.coord_min,
                    self.coord_max,
                    y_pt,
                )
                return softmax_readout(q, keys, values, bias)

            def residual(*c):
                return self.pde_spec.residual_fn(operator_closure, u_i, *c)

            cols = tuple(y_i[:, i] for i in range(y_i.shape[1]))
            pred = vmap(residual)(*cols)
            return jnp.mean((s_i.flatten() - pred) ** 2)

        losses = vmap(single_function_loss)(u, y, outputs)
        return jnp.mean(losses)

    def loss(self, params, bcs_batch, res_batch):
        return self.loss_bcs(params, bcs_batch) + self.loss_res(params, res_batch)

    # -- Compiled update step --------------------------------------------------

    @partial(jit, static_argnums=(0,))
    def step(self, params, opt_state, bcs_batch, res_batch):
        g = grad(self.loss)(params, bcs_batch, res_batch)
        updates, opt_state = self.optimizer.update(g, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state

    # -- Training loop ---------------------------------------------------------

    def train(
        self,
        bcs_dataset,
        res_dataset,
        nIter=10000,
        init_lr=1e-3,
        end_lr_frac=0.01,
        warmup_frac=0.0,
        rebuild_schedule=True,
        weight_decay=None,
    ):
        """Train for nIter steps; if rebuild_schedule, first call rebuild_optimizer.

        weight_decay=None uses the model's weight_decay hparam.
        """
        if rebuild_schedule:
            from architectures.training_schedules import rebuild_optimizer

            if weight_decay is None:
                weight_decay = self._hparams.get("weight_decay", 1e-4)
            rebuild_optimizer(
                self,
                nIter,
                init_lr=init_lr,
                end_lr_frac=end_lr_frac,
                warmup_frac=warmup_frac,
                weight_decay=weight_decay,
            )

        bcs_data = iter(bcs_dataset)
        res_data = iter(res_dataset)
        pbar = trange(nIter)
        for it in pbar:
            bcs_batch = next(bcs_data)
            res_batch = next(res_data)
            self.params, self.opt_state = self.step(
                self.params, self.opt_state, bcs_batch, res_batch
            )
            if it % 100 == 0:
                lv = self.loss(self.params, bcs_batch, res_batch)
                lbc = self.loss_bcs(self.params, bcs_batch)
                lr = self.loss_res(self.params, res_batch)
                self.loss_log.append(lv)
                self.loss_bcs_log.append(lbc)
                self.loss_res_log.append(lr)
                pbar.set_postfix({"Loss": lv, "loss_bcs": lbc, "loss_physics": lr})

    # -- Prediction ------------------------------------------------------------

    @partial(jit, static_argnums=(0,))
    def predict_s(self, params, U_star, Y_star):
        cols = tuple(Y_star[:, i] for i in range(Y_star.shape[1]))
        in_axes = (None, 0) + (0,) * Y_star.shape[1]
        return vmap(self.operator_net, in_axes)(params, U_star, *cols)

    @partial(jit, static_argnums=(0,))
    def predict_res(self, params, U_star, Y_star):
        cols = tuple(Y_star[:, i] for i in range(Y_star.shape[1]))
        in_axes = (None, 0) + (0,) * Y_star.shape[1]
        return vmap(self.residual_net, in_axes)(params, U_star, *cols)
