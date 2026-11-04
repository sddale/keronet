"""
PI-DeepONet (Wang, Wang & Perdikaris, 2021): DeepONet with a bilinear readout.

tanh branch and Fourier-feature tanh trunk MLPs give b(u) and g(y) in R^p.
    s(u, y) = <b, g> + bias
"""

import jax
import jax.numpy as jnp
from jax import random, grad, vmap, jit
import optax
from functools import partial
from tqdm import trange


# ============================================================================
# Primitives
# ============================================================================


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


def mlp_tanh_apply(params, x):
    """tanh after every hidden layer; final layer linear."""
    for W, b in params[:-1]:
        x = jnp.tanh(dense(W, b, x))
    W, b = params[-1]
    return dense(W, b, x)


# -- Branch and Trunk ---------------------------------------------------------


def branch_init(key, n_sensors, hidden, layers_n, p):
    """Branch MLP: n_sensors -> hidden x layers_n -> p."""
    sizes = [n_sensors] + [hidden] * layers_n + [p]
    return mlp_init(key, sizes)


def branch_apply(branch_params, u):
    """Branch coefficients b (p,) for sensor values u (m,)."""
    return mlp_tanh_apply(branch_params, u)


def trunk_init(key, trunk_in, hidden, layers_n, p, mapping_size, use_fourier):
    """Trunk MLP -> p; input dim is 2*mapping_size if use_fourier else trunk_in."""
    in_dim = 2 * mapping_size if use_fourier else trunk_in
    sizes = [in_dim] + [hidden] * layers_n + [p]
    return mlp_init(key, sizes)


def trunk_apply(trunk_params, B_fourier, coord_min, coord_max, y, use_fourier):
    """Trunk basis g (p,) at coordinate y (trunk_in,).

    Fourier input: [cos, sin](2*pi*y_n @ B_fourier), y_n = y rescaled to [0, 1].
    """
    if use_fourier:
        # Same [0, 1] normalization as KerONet's trunk lift, so a given sigma
        # gives both architectures the same physical frequencies.
        y_norm = (y - coord_min) / (coord_max - coord_min + 1e-8)
        angles = 2.0 * jnp.pi * (y_norm @ B_fourier)
        ff = jnp.concatenate([jnp.cos(angles), jnp.sin(angles)], axis=-1)
        return mlp_tanh_apply(trunk_params, ff)
    else:
        # Raw (x, t) into MLP -- matches PINTO paper's PI-DeepONet baseline
        return mlp_tanh_apply(trunk_params, y)


# ============================================================================
# PIDeepONet
# ============================================================================


class PIDeepONet:
    """PI-DeepONet model with readout s = <branch(u), trunk(y)> + bias."""

    def __init__(
        self,
        pde_spec,
        n_sensors,
        branch_hidden=128,
        branch_layers_n=3,
        trunk_hidden=128,
        trunk_layers_n=3,
        p=128,
        mapping_size=64,
        sigma=0.5,
        weight_decay=1e-4,
        init_seed=0,
    ):
        """Initialize params and a placeholder AdamW optimizer (exponential LR decay).

        Args:
            p: number of basis functions (branch and trunk output dim).
            mapping_size, sigma: count and scale of the trunk's random Fourier features.
        """
        trunk_in = len(pde_spec.coord_range)

        self.pde_spec = pde_spec
        self._hparams = {
            "n_sensors": n_sensors,
            "branch_hidden": branch_hidden,
            "branch_layers_n": branch_layers_n,
            "trunk_hidden": trunk_hidden,
            "trunk_layers_n": trunk_layers_n,
            "p": p,
            "mapping_size": mapping_size,
            "sigma": sigma,
            "weight_decay": weight_decay,
            "init_seed": init_seed,
            "pde": pde_spec.name,
        }
        self.p = p

        # Coord range from PDE spec
        coord_range = pde_spec.coord_range
        self.coord_min = jnp.array([r[0] for r in coord_range])
        self.coord_max = jnp.array([r[1] for r in coord_range])

        key = random.PRNGKey(init_seed)
        key, kb, kt, kf = random.split(key, 4)
        self.B_fourier = random.normal(kf, (trunk_in, mapping_size)) * sigma

        branch_params = branch_init(kb, n_sensors, branch_hidden, branch_layers_n, p)
        trunk_params = trunk_init(
            kt,
            trunk_in,
            trunk_hidden,
            trunk_layers_n,
            p,
            mapping_size,
            use_fourier=True,
        )
        bias = jnp.zeros(())
        params = {
            "branch": branch_params,
            "trunk": trunk_params,
            "bias": bias,
        }

        # AdamW with decay mask on 2D+ tensors only
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
        b = branch_apply(params["branch"], u)
        y = jnp.array(coords)
        g = trunk_apply(
            params["trunk"],
            self.B_fourier,
            self.coord_min,
            self.coord_max,
            y,
            use_fourier=True,
        )
        return jnp.dot(b, g) + params["bias"]

    # -- PDE residual (delegated; with branch caching) -----------------------

    def residual_net(self, params, u, *coords):
        # Cache branch coefficients
        b = branch_apply(params["branch"], u)
        bias = params["bias"]

        def operator_closure(*c_):
            y = jnp.array(c_)
            g = trunk_apply(
                params["trunk"],
                self.B_fourier,
                self.coord_min,
                self.coord_max,
                y,
                use_fourier=True,
            )
            return jnp.dot(b, g) + bias

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
            b = branch_apply(params["branch"], u_i)
            bias = params["bias"]

            def operator_closure(*c_):
                y_pt = jnp.array(c_)
                g = trunk_apply(
                    params["trunk"],
                    self.B_fourier,
                    self.coord_min,
                    self.coord_max,
                    y_pt,
                    use_fourier=True,
                )
                return jnp.dot(b, g) + bias

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


# ============================================================================
# Smoke test
# ============================================================================
if __name__ == "__main__":
    from pdes import get_pde

    spec = get_pde("burgers")
    m = 80

    model = PIDeepONet(
        pde_spec=spec,
        n_sensors=m,
        branch_hidden=128,
        branch_layers_n=3,
        trunk_hidden=128,
        trunk_layers_n=3,
        p=128,
        mapping_size=64,
        sigma=0.5,
    )

    key = random.PRNGKey(99)
    params = model.params
    u_test = random.normal(key, (m,))
    out = model.operator_net(params, u_test, 0.5, 0.3)
    print("operator_net output:", out)
    res = model.residual_net(params, u_test, 0.5, 0.3)
    print("residual_net output:", res)

    from jax.flatten_util import ravel_pytree

    flat, _ = ravel_pytree(params)
    print(f"total params: {len(flat):,}")
