"""
JAX port of PINTO (Boya & Subramani, 2025), an iterative cross-attention operator.

tanh MLPs lift the query to a token ct, sensor coordinates to keys and sensor values to
values; each of n_cross_attn blocks does ct = ct + MHA(ct, keys, values) and
ct = ct + FFN(ct), and a tanh MLP plus a linear layer map ct to a scalar.
"""

import jax
import jax.numpy as jnp
from jax import random, grad, vmap, jit
import optax
from functools import partial
from tqdm import trange


# ============================================================================
# Primitives  (Glorot-uniform, tanh MLPs, standard softmax MHA)
# ============================================================================


def dense_init(key, fan_in, fan_out):
    lim = jnp.sqrt(6.0 / (fan_in + fan_out))
    W = random.uniform(key, (fan_in, fan_out), minval=-lim, maxval=lim)
    b = jnp.zeros(fan_out)
    return W, b


def dense(W, b, x):
    return x @ W + b


def mlp_tanh_init(key, layer_sizes):
    """Glorot-uniform dense layers; the activation is chosen by the apply function."""
    params = []
    for i in range(len(layer_sizes) - 1):
        key, kd = random.split(key)
        params.append(dense_init(kd, layer_sizes[i], layer_sizes[i + 1]))
    return params


def mlp_tanh_apply_all(params, x):
    """Apply the layers with tanh after every layer, including the last."""
    for W, b in params:
        x = jnp.tanh(dense(W, b, x))
    return x


def mlp_tanh_apply_last_linear(params, x):
    """Apply the layers with tanh after all but the last (linear) layer."""
    for W, b in params[:-1]:
        x = jnp.tanh(dense(W, b, x))
    W, b = params[-1]
    return dense(W, b, x)


def mha_init(key, d_model, n_heads, key_dim):
    """MHA params; each head has key_dim (Keras style), not d_model / n_heads."""
    keys = random.split(key, 4)
    # Q, K, V project (d_model,) -> (n_heads, key_dim); we store them flat
    # as (d_model, n_heads * key_dim) and reshape in apply.
    Wq, bq = dense_init(keys[0], d_model, n_heads * key_dim)
    Wk, bk = dense_init(keys[1], d_model, n_heads * key_dim)
    Wv, bv = dense_init(keys[2], d_model, n_heads * key_dim)
    # Output projection  (n_heads * key_dim) -> d_model
    Wo, bo = dense_init(keys[3], n_heads * key_dim, d_model)
    return {
        "Wq": Wq,
        "bq": bq,
        "Wk": Wk,
        "bk": bk,
        "Wv": Wv,
        "bv": bv,
        "Wo": Wo,
        "bo": bo,
    }


def mha_apply(params, q, k, v, n_heads, key_dim):
    """Softmax multi-head cross-attention: q (Sq, d), k and v (Sk, d) -> (Sq, d)."""
    Sq = q.shape[0]
    Sk = k.shape[0]

    Q = (
        dense(params["Wq"], params["bq"], q)
        .reshape(Sq, n_heads, key_dim)
        .transpose(1, 0, 2)
    )
    K = (
        dense(params["Wk"], params["bk"], k)
        .reshape(Sk, n_heads, key_dim)
        .transpose(1, 0, 2)
    )
    V = (
        dense(params["Wv"], params["bv"], v)
        .reshape(Sk, n_heads, key_dim)
        .transpose(1, 0, 2)
    )
    # Now Q: (h, Sq, key_dim), K, V: (h, Sk, key_dim)

    scale = jnp.sqrt(jnp.asarray(key_dim, dtype=Q.dtype))
    scores = jnp.einsum("hqd,hkd->hqk", Q, K) / scale
    attn = jax.nn.softmax(scores, axis=-1)
    out = jnp.einsum("hqk,hkd->hqd", attn, V)  # (h, Sq, key_dim)

    out = out.transpose(1, 0, 2).reshape(Sq, n_heads * key_dim)
    return dense(params["Wo"], params["bo"], out)  # (Sq, d_model)


# ============================================================================
# PINTO
# ============================================================================


def pinto_init(key, d_model=64, n_heads=2, key_dim=64, n_cross_attn=3, n_qc=2):
    """Initialize query/key/value lifts, n_cross_attn blocks and the output head.

    n_qc is the number of query coordinates.
    """
    keys = random.split(key, 10)
    # Three lift MLPs (query, key, value)
    sp_trans = mlp_tanh_init(
        keys[0], [n_qc, d_model, d_model]
    )  # query coords -> d_model
    bpe = mlp_tanh_init(keys[1], [n_qc, d_model, d_model])  # sensor coords -> d_model
    bve = mlp_tanh_init(keys[2], [1, d_model, d_model])  # u_s -> d_model

    # All cross-attention blocks have identical structure per the paper's
    # Figure 1: each block has MHA + attention-skip + FFN + FFN-skip.
    blocks = []
    block_keys = random.split(keys[3], n_cross_attn)
    for k in block_keys:
        kk = random.split(k, 2)
        block = {
            "attn": mha_init(kk[0], d_model, n_heads, key_dim),
            "ffn": mlp_tanh_init(kk[1], [d_model, d_model, d_model]),
        }
        blocks.append(block)

    # Output head: flatten(ct) -> MLP [64, 64, tanh] -> Dense(1)
    # Since query is a single token, flatten just reshapes (1, d_model) -> (d_model,)
    out_mlp = mlp_tanh_init(keys[4], [d_model, d_model, d_model])
    out_final = dense_init(keys[5], d_model, 1)

    return {
        "sp_trans": sp_trans,
        "bpe": bpe,
        "bve": bve,
        "blocks": blocks,
        "out_mlp": out_mlp,
        "out_final": out_final,
    }


def pinto_apply(
    params,
    query_coords,
    sensor_coords,
    u_sensors,
    coord_min,
    coord_max,
    n_heads,
    key_dim,
):
    """PINTO forward pass for one query point; returns a scalar.

    query_coords: (n_qc,); sensor_coords: (m, n_qc) in the same space; u_sensors: (m,).
    """
    # Rescale all coords to [-1, 1]
    q_r = 2.0 * (query_coords - coord_min) / (coord_max - coord_min) - 1.0
    s_r = 2.0 * (sensor_coords - coord_min) / (coord_max - coord_min) - 1.0
    # sensor values are NOT rescaled (no corresponding bound known a priori)

    # Query lift:  (n_qc,) -> (1, d_model)
    spq = mlp_tanh_apply_all(params["sp_trans"], q_r)
    spq = spq[None, :]  # (1, d_model)

    # Key lift:  (m, n_qc) -> (m, d_model)
    pe_in = s_r  # (m, n_qc)
    pe = vmap(lambda row: mlp_tanh_apply_all(params["bpe"], row))(pe_in)

    # Value lift:  (m, 1) -> (m, d_model)
    ce_in = u_sensors[:, None]  # (m, 1)
    ce = vmap(lambda row: mlp_tanh_apply_all(params["bve"], row))(ce_in)

    # Iterative cross-attention: every block has identical structure
    # (MHA + attention-skip + FFN + FFN-skip), per paper Figure 1.
    ct = spq
    for block in params["blocks"]:
        # Cross-attention with skip connection
        attn_out = mha_apply(block["attn"], ct, pe, ce, n_heads, key_dim)
        ct = ct + attn_out
        # FFN with skip connection
        ffn_out = mlp_tanh_apply_all(block["ffn"], ct)
        ct = ct + ffn_out

    # Output head: flatten -> MLP -> Dense(1)
    flat = ct.reshape(-1)  # (d_model,)
    h = mlp_tanh_apply_all(params["out_mlp"], flat)
    W, b = params["out_final"]
    return dense(W, b, h)[0]  # scalar


# ============================================================================
# PINTO_DeepONet  (class wrapper matching our interface)
# ============================================================================


class PINTO:
    """PINTO model, trained with AdamW (weight decay on 2D+ tensors only)."""

    def __init__(
        self,
        pde_spec,
        n_sensors,
        d_model=64,
        n_heads=2,
        key_dim=64,
        n_cross_attn=3,
        sensor_t=0.0,  # all sensors at t=0 (initial condition)
        weight_decay=1e-4,
        init_seed=0,
        sensor_coords=None,
    ):
        """Initialize params and a placeholder AdamW optimizer (exponential LR decay).

        Args:
            sensor_t: value of the time coordinate appended to sensor locations.
            sensor_coords: (m, d_s) locations; default from the spec, else uniform in x.
        """
        self.pde_spec = pde_spec
        if sensor_coords is None:
            from pdes._periodic2d import resolve_sensor_coords

            sensor_coords = resolve_sensor_coords(pde_spec, n_sensors)
        self._hparams = {
            "n_sensors": n_sensors,
            "d_model": d_model,
            "n_heads": n_heads,
            "key_dim": key_dim,
            "n_cross_attn": n_cross_attn,
            "sensor_t": sensor_t,
            "weight_decay": weight_decay,
            "pde": pde_spec.name,
        }
        self.n_heads = n_heads
        self.key_dim = key_dim
        trunk_in = 2

        # Coord range comes from the PDE spec
        coord_range = pde_spec.coord_range
        self.coord_min = jnp.array([r[0] for r in coord_range])
        self.coord_max = jnp.array([r[1] for r in coord_range])

        # Sensor coordinates as a full (m, n_qc) array in the query space.
        # The final (time) channel is pinned to 0: sensors sample the initial
        # condition. Spatial channels come from sensor_coords when provided
        # (scattered 2D sensors), else a uniform grid in x.
        n_qc = len(coord_range)
        if sensor_coords is not None:
            sc = jnp.asarray(sensor_coords)  # (m, d_spatial)
        else:
            x_min, x_max = coord_range[0]
            sc = jnp.linspace(x_min, x_max, n_sensors)[:, None]
        d_spatial = sc.shape[1]
        pad = n_qc - d_spatial
        if pad > 0:
            sc = jnp.concatenate([sc, jnp.full((n_sensors, pad), sensor_t)], axis=1)
        self.sensor_coords_full = sc  # (m, n_qc)

        key = random.PRNGKey(init_seed)
        key, kp = random.split(key)
        params = pinto_init(
            kp, d_model, n_heads, key_dim, n_cross_attn, n_qc=len(coord_range)
        )

        # AdamW with decay mask on 2D+ tensors only (matches other classes)
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
        query_coords = jnp.array(coords)
        return pinto_apply(
            params,
            query_coords,
            self.sensor_coords_full,
            u,
            self.coord_min,
            self.coord_max,
            self.n_heads,
            self.key_dim,
        )

    # -- PDE residual (delegated to pde_spec) --------------------------------
    # The PDESpec's residual_fn knows which derivatives it needs and how to
    # assemble the residual. We just hand it a scalar (x, t) -> u closure.

    def residual_net(self, params, u, *coords):
        def operator_closure(*c_):
            return self.operator_net(params, u, *c_)

        return self.pde_spec.residual_fn(operator_closure, u, *coords)

    # -- Losses ----------------------------------------------------------------

    def loss_bcs(self, params, batch):
        """Initial/boundary-condition loss, computed by pde_spec.bcs_loss_fn."""
        return self.pde_spec.bcs_loss_fn(self, params, batch)

    def loss_res(self, params, batch):
        inputs, outputs = batch
        u, y = inputs

        def single_function_loss(u_i, y_i, s_i):
            def residual(*c):
                return self.residual_net(params, u_i, *c)

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

    def predict_res(self, params, U_star, Y_star):
        cols = tuple(Y_star[:, i] for i in range(Y_star.shape[1]))
        in_axes = (None, 0) + (0,) * Y_star.shape[1]
        return vmap(self.residual_net, in_axes)(params, U_star, *cols)


# ============================================================================
# Smoke test  (run as: python pinto.py)
# ============================================================================
if __name__ == "__main__":
    from pdes import get_pde

    spec = get_pde("burgers")
    m = 80

    model = PINTO(
        pde_spec=spec,
        n_sensors=m,
        d_model=64,
        n_heads=2,
        key_dim=64,
        n_cross_attn=3,
    )

    key = random.PRNGKey(99)
    params = model.params
    u_test = random.normal(key, (m,))
    out = model.operator_net(params, u_test, 0.5, 0.3)
    print("operator_net output:", out)

    res = model.residual_net(params, u_test, 0.5, 0.3)
    print("residual_net output:", res)

    # Smoke-test BCS loss with IC+periodic BC structure
    N = 4
    P_ic, P_bc, Q = 200, 50, 2000
    U_star = random.normal(key, (N, m))
    k_i, k_l, k_s, k_r = random.split(key, 4)
    y_ic = jnp.stack([random.uniform(k_i, (N, P_ic)), jnp.zeros((N, P_ic))], axis=-1)
    s_ic = random.normal(k_s, (N, P_ic, 1))
    t_bc = random.uniform(k_l, (N, P_bc))
    y_bc_lo = jnp.stack([jnp.zeros((N, P_bc)), t_bc], axis=-1)
    y_bc_hi = jnp.stack([jnp.ones((N, P_bc)), t_bc], axis=-1)
    bcs_batch = ((U_star, y_ic, y_bc_lo, y_bc_hi), s_ic)
    print("loss_bcs:", model.loss_bcs(params, bcs_batch))

    # Smoke-test residual loss
    y_res_test = random.uniform(k_r, (N, Q, 2))
    s_res_test = jnp.zeros((N, Q, 1))
    res_batch = ((U_star, y_res_test), s_res_test)
    print("loss_res:", model.loss_res(params, res_batch))

    # Prediction sanity
    N_pts = 32
    U_p = random.normal(key, (N_pts, m))
    Y_p = random.uniform(key, (N_pts, 2))
    s_pred = model.predict_s(params, U_p, Y_p)
    print("predict_s shape:", s_pred.shape)
    s_r = model.predict_res(params, U_p, Y_p)
    print("predict_res shape:", s_r.shape)

    from jax.flatten_util import ravel_pytree

    for k, sub in params.items():
        flat, _ = ravel_pytree(sub)
        print(f"{k:>14}: {len(flat):>10,}")
    flat, _ = ravel_pytree(params)
    print(f"{'total':>14}: {len(flat):>10,}")
