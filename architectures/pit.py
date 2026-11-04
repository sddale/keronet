"""
JAX port of PIT (Physics-Informed Transformer; dos Santos et al., 2023).

GELU MLPs embed sinusoidal encodings of the query (z) and of each sensor coordinate
with its value (y); each of n_blocks blocks computes z = z + MLP(LN(z) + MHA(LN(z), y)),
and a GELU MLP decodes z to a scalar.
"""

import jax
import jax.numpy as jnp
from jax import random, grad, vmap, jit
import optax
from functools import partial
from tqdm import trange


### Primitives ###
def dense_init(key, fan_in, fan_out):
    lim = jnp.sqrt(6.0 / (fan_in + fan_out))
    W = random.uniform(key, (fan_in, fan_out), minval=-lim, maxval=lim)
    b = jnp.zeros(fan_out)
    return W, b


def dense(W, b, x):
    return x @ W + b


def mlp_init(key, layer_sizes):
    params = []
    for i in range(len(layer_sizes) - 1):
        key, kd = random.split(key)
        params.append(dense_init(kd, layer_sizes[i], layer_sizes[i + 1]))
    return params


def mlp_apply_gelu(params, x):
    """GELU after each hidden layer; final layer linear."""
    for W, b in params[:-1]:
        x = jax.nn.gelu(dense(W, b, x))
    W, b = params[-1]
    return dense(W, b, x)


def layer_norm_init(d):
    return {"gamma": jnp.ones(d), "beta": jnp.zeros(d)}


def layer_norm_apply(params, x, eps=1e-5):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
    return params["gamma"] * (x - mean) / jnp.sqrt(var + eps) + params["beta"]


###  Sinusoidal positional encoding ###
def sinusoidal_pe(coords, pe_dim):
    """Sinusoidal encoding per coordinate: coords (d_in,) -> (d_in * pe_dim,).

    Each c maps to [sin(c*f_i), cos(c*f_i)], f_i = 10000^(-2i/pe_dim); pe_dim even.
    """
    half = pe_dim // 2
    # Frequencies follow Vaswani et al.: 1/10000^(2i/pe_dim)
    freqs = jnp.exp(jnp.arange(half) * (-jnp.log(10000.0) / half))
    # coords: (d_in,), scaled: (d_in, half)
    scaled = coords[:, None] * freqs[None, :]
    pe = jnp.concatenate([jnp.sin(scaled), jnp.cos(scaled)], axis=-1)
    return pe.flatten()


###  Multi-head cross-attention ###
def mha_init(key, d_model, n_heads, key_dim):
    keys = random.split(key, 4)
    Wq, bq = dense_init(keys[0], d_model, n_heads * key_dim)
    Wk, bk = dense_init(keys[1], d_model, n_heads * key_dim)
    Wv, bv = dense_init(keys[2], d_model, n_heads * key_dim)
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

    scale = jnp.sqrt(jnp.asarray(key_dim, dtype=Q.dtype))
    scores = jnp.einsum("hqd,hkd->hqk", Q, K) / scale
    attn = jax.nn.softmax(scores, axis=-1)
    out = jnp.einsum("hqk,hkd->hqd", attn, V)
    out = out.transpose(1, 0, 2).reshape(Sq, n_heads * key_dim)
    return dense(params["Wo"], params["bo"], out)


### PIT ###
def pit_init(
    key,
    d_model=64,
    n_heads=2,
    key_dim=32,
    n_blocks=1,
    pe_dim=32,
    query_dim=2,
    sensor_dim=1,
):
    """Initialize encoders phi and psi, n_blocks attention blocks and decoder gamma.

    query_dim and sensor_dim are the numbers of query and sensor coordinates.
    """
    keys = random.split(key, 10)

    # Encoder MLPs:
    # \phi: pe(q) -> d_model      (pe_dim per dim, query_dim dims -> query_dim*pe_dim)
    # \psi: [pe(p), f(p)] -> d_model    (sensor_dim*pe_dim + 1)
    phi_in_dim = query_dim * pe_dim
    psi_in_dim = sensor_dim * pe_dim + 1  # +1 for f(p) value
    phi = mlp_init(keys[0], [phi_in_dim, d_model, d_model])
    psi = mlp_init(keys[1], [psi_in_dim, d_model, d_model])

    # Cross-attention blocks (paper equation 6):
    #   \hat{z} = LN(z)
    #   z' = z + MLP(\hat{z} + Cross-attention(\hat{z}, y))
    # One LN, one MLP, single residual per block.
    blocks = []
    block_keys = random.split(keys[2], n_blocks)
    for k in block_keys:
        kk = random.split(k, 3)
        block = {
            "ln": layer_norm_init(d_model),
            "attn": mha_init(kk[0], d_model, n_heads, key_dim),
            "mlp": mlp_init(kk[1], [d_model, d_model, d_model]),
        }
        blocks.append(block)

    # Decoder \gamma: d_model -> scalar (single-output for Burgers)
    gamma = mlp_init(keys[3], [d_model, d_model, 1])

    return {
        "phi": phi,
        "psi": psi,
        "blocks": blocks,
        "gamma": gamma,
    }


def pit_apply(
    params,
    query_coords,
    sensor_coords,
    sensor_vals,
    n_heads,
    key_dim,
    pe_dim,
    coord_min,
    coord_max,
):
    """PIT forward pass for one query point; returns a scalar.

    query_coords: (n_q,); sensor_coords: (m, d_s) spatial; sensor_vals: (m,).
    """
    # Rescale coords to [-1, 1] (matches PINTO convention; PIT paper
    # doesn't specify but normalization is standard practice)
    # Query coordinate is (x, t) in the 1D benchmarks and (x, y) in 2D, in
    # both cases matching the column order of sensor_coords. That agreement
    # matters: sinusoidal_pe assigns frequency bands per component, so if the
    # query were ordered (t, x) while sensors are (x, y), the query's first
    # band would be compared against the sensors' second.
    xt_scaled = 2 * (query_coords - coord_min) / (coord_max - coord_min) - 1

    # Rescale each sensor column against its own coordinate range. Slicing to
    # d_s keeps the 1D path working, where sensor_coords is (m, 1) holding x
    # only and coord_min/coord_max are 2-vectors.
    d_s = sensor_coords.shape[1]
    sensor_scaled = (
        2 * (sensor_coords - coord_min[:d_s]) / (coord_max[:d_s] - coord_min[:d_s]) - 1
    )

    #  Positional encoding
    pe_q = sinusoidal_pe(xt_scaled, pe_dim)  # (query_dim * pe_dim,)
    pe_p = vmap(lambda c: sinusoidal_pe(c, pe_dim))(sensor_scaled)
    # pe_p: (m, d_s * pe_dim)

    #  Encoder embeddings
    # Query: (1, d_model)
    U = mlp_apply_gelu(params["phi"], pe_q)[None, :]

    # Input: concatenate pe(p_i) with f(p_i), then \psi
    psi_input = jnp.concatenate([pe_p, sensor_vals[:, None]], axis=-1)
    K_in = vmap(lambda v: mlp_apply_gelu(params["psi"], v))(psi_input)
    # K_in: (m, d_model)

    #  Cross-attention blocks (paper equation 6)
    # \hat{z} = LN(z)
    # z' = z + MLP(\hat{z} + Cross-attention(\hat{z}, y))
    Z = U
    for block in params["blocks"]:
        # Pre-norm: \hat{z} = LN(z)
        Z_hat = vmap(lambda v: layer_norm_apply(block["ln"], v))(Z)
        # Cross-attention with normalized query
        attn_out = mha_apply(block["attn"], Z_hat, K_in, K_in, n_heads, key_dim)
        # MLP input is the sum of normalized query and attention output
        # (this is the paper's "accentuate the influence of the query points":
        # \hat{z} is added inside the MLP arg, equivalent to a residual through MLP)
        mlp_out = vmap(lambda v: mlp_apply_gelu(block["mlp"], v))(Z_hat + attn_out)
        # Outer residual on the original (un-normalized) z
        Z = Z + mlp_out

    #  Decoder \gamma
    out = mlp_apply_gelu(params["gamma"], Z[0])  # (1,)
    return out[0]


# PIT class --- interface matches our other physics-informed models
class PIT:
    """PIT model; the PDE residual and BCS loss are delegated to the PDESpec."""

    def __init__(
        self,
        pde_spec,
        n_sensors,
        d_model=64,
        n_heads=2,
        key_dim=32,
        n_blocks=1,  # L=1 by default (single cross-attention)
        pe_dim=32,
        sensor_t=0.0,  # all sensors at t=0 (initial condition)
        weight_decay=1e-4,
        init_seed=0,
        sensor_coords=None,
    ):
        """Initialize params and a placeholder AdamW optimizer (exponential LR decay).

        Args:
            pe_dim: sinusoidal size per coordinate; sensor_t: unused, kept in hparams.
            sensor_coords: (m, d_s) locations; default from the spec, else uniform in x.
        """
        self.pde_spec = pde_spec
        # build_model_from_config calls cls(pde_spec=..., **hparams) with
        # hparams coming from YAML, so an (m, d_s) array cannot be threaded
        # through a config file. Specs instead advertise a layout name, which
        # this resolves. Returns None for every 1D spec.
        if sensor_coords is None:
            try:
                from pdes._periodic2d import resolve_sensor_coords

                sensor_coords = resolve_sensor_coords(pde_spec, n_sensors)
            except ImportError:
                sensor_coords = None

        self._hparams = {
            "n_sensors": n_sensors,
            "d_model": d_model,
            "n_heads": n_heads,
            "key_dim": key_dim,
            "n_blocks": n_blocks,
            "pe_dim": pe_dim,
            "sensor_t": sensor_t,
            "weight_decay": weight_decay,
            "pde": pde_spec.name,
        }
        self.n_heads = n_heads
        self.key_dim = key_dim
        self.pe_dim = pe_dim

        # Coord range comes from the PDE spec
        coord_range = pde_spec.coord_range
        self.coord_min = jnp.array([r[0] for r in coord_range])
        self.coord_max = jnp.array([r[1] for r in coord_range])

        # Sensor positions.
        if sensor_coords is not None:
            # Explicit scattered locations, (m, d_s). Their positions carry
            # real information, so every column is encoded.
            self.sensor_coords = jnp.asarray(sensor_coords)
        else:
            # 1D default: uniform in x at fixed t = sensor_t. Only the x column
            # is kept, so sensor_dim = 1. Every sensor shares the same t, so a
            # second column would be a constant channel -- it would add
            # parameters and change results without adding information, and
            # would break continuity with the published 1D configurations.
            x_min, x_max = coord_range[0]
            self.sensor_coords = jnp.linspace(x_min, x_max, n_sensors)[:, None]
        self._hparams["sensor_dim"] = int(self.sensor_coords.shape[1])

        key = random.PRNGKey(init_seed)
        params = pit_init(
            key,
            d_model=d_model,
            n_heads=n_heads,
            key_dim=key_dim,
            n_blocks=n_blocks,
            pe_dim=pe_dim,
            query_dim=len(coord_range),
            sensor_dim=self.sensor_coords.shape[1],
        )

        # AdamW with decay mask on 2D+ tensors only (matches our convention)
        def decay_mask(p):
            return jax.tree_util.tree_map(lambda x: x.ndim >= 2, p)

        # Default schedule (will be overridden by train() via rebuild_optimizer)
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

    ### Forward ###
    def operator_net(self, params, u, *coords):
        query_coords = jnp.array(coords)
        return pit_apply(
            params,
            query_coords,
            self.sensor_coords,
            u,
            self.n_heads,
            self.key_dim,
            self.pe_dim,
            self.coord_min,
            self.coord_max,
        )

    ###  PDE residual (delegated to pde_spec) ###
    def residual_net(self, params, u, *coords):
        def operator_closure(*c_):
            return self.operator_net(params, u, *c_)

        return self.pde_spec.residual_fn(operator_closure, u, *coords)

    ###  Losses ###
    def loss_bcs(self, params, batch):
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

    ### Compiled update step ###
    @partial(jit, static_argnums=(0,))
    def step(self, params, opt_state, bcs_batch, res_batch):
        g = grad(self.loss)(params, bcs_batch, res_batch)
        updates, opt_state = self.optimizer.update(g, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state

    ### Training loop ###
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
                pbar.set_postfix(
                    {
                        "loss": f"{float(lv):.3e}",
                        "bcs": f"{float(lbc):.3e}",
                        "res": f"{float(lr):.3e}",
                    }
                )

    ### Prediction ###
    @partial(jit, static_argnums=(0,))
    def predict_s(self, params, U_star, Y_star):
        cols = tuple(Y_star[:, i] for i in range(Y_star.shape[1]))
        in_axes = (None, 0) + (0,) * Y_star.shape[1]
        return vmap(self.operator_net, in_axes)(params, U_star, *cols)

    def predict_res(self, params, U_star, Y_star):
        cols = tuple(Y_star[:, i] for i in range(Y_star.shape[1]))
        in_axes = (None, 0) + (0,) * Y_star.shape[1]
        return vmap(self.residual_net, in_axes)(params, U_star, *cols)
