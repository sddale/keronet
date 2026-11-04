"""
2D helpers on the periodic unit square: GRF, uniform interior points, and the sensor grid.

GRF: g(x) = sum_k a_k cos(2*pi*k.x) + b_k sin(2*pi*k.x) over half-plane modes k != 0,
a_k, b_k ~ N(0, lam_k^2), lam_k = sqrt(2)*|sigma|*((2*pi)^2*|k|^2 + tau^2)^(-gamma/2).
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
from jax import lax, random


# ==========================================================================
# 2D GRF
# ==========================================================================
#
# The 1D code stores conjugate-symmetric complex coefficients; summing each
# +/-k pair gives the real form
#     s(x) = sum_{k>0} a_k cos(2 pi k x) + b_k sin(2 pi k x),
#     a_k, b_k ~ N(0, lambda_k^2),
#     lambda_k = sqrt(2) sigma ((2 pi k)^2 + tau^2)^(-gamma/2).
# We use that real form directly in 2D with (2 pi k)^2 -> (2 pi)^2 |k|^2,
# drawing modes from a half-plane so the field is real by construction.
# The (0,0) mode is excluded, so the field is zero-mean as in 1D.


def half_plane_modes(n_modes: int) -> jnp.ndarray:
    """Float32 (n, 2) integer modes, |k|_inf <= n_modes, one per +/- pair, no (0, 0)."""
    ks = []
    for k1 in range(0, n_modes + 1):
        for k2 in range(-n_modes, n_modes + 1):
            if k1 == 0 and k2 <= 0:
                continue
            ks.append((k1, k2))
    return jnp.asarray(np.asarray(ks, dtype=np.float32))


def sample_GRF2d_coefs(key, modes, gamma=4.0, tau=5.0, sigma=25.0**2):
    """Draw the cosine and sine amplitudes (a, b), each (len(modes),), of one GRF."""
    k2 = jnp.sum(modes**2, axis=1)
    lam = (
        jnp.sqrt(2.0)
        * jnp.abs(sigma)
        * ((2.0 * jnp.pi) ** 2 * k2 + tau**2) ** (-gamma / 2.0)
    )
    key_a, key_b = random.split(key)
    a = lam * random.normal(key_a, (modes.shape[0],))
    b = lam * random.normal(key_b, (modes.shape[0],))
    return a, b


def eval_GRF2d_at(modes, a, b, pts, block=2048):
    """Evaluate the GRF at pts (n, 2) -> (n,), scanning blocks of `block` points."""
    n = pts.shape[0]
    pad = (-n) % block
    P = jnp.pad(pts, ((0, pad), (0, 0))).reshape(-1, block, 2)

    def body(carry, blk):
        theta = 2.0 * jnp.pi * (blk @ modes.T)
        return carry, jnp.cos(theta) @ a + jnp.sin(theta) @ b

    _, out = lax.scan(body, None, P)
    return out.reshape(-1)[:n]


# ==========================================================================
# Interior points and sensors
# ==========================================================================


def sample_interior_square(key, n):
    """Uniform on the unit square. Static shapes; safe under vmap."""
    return random.uniform(key, (n, 2))


def resolve_sensor_coords(pde_spec, n_sensors):
    """Sensor coordinates (n_sensors, 2) for sensor_layout "square_grid_2d", else None (every 1D spec)."""
    if pde_spec.params.get("sensor_layout") != "square_grid_2d":
        return None
    # 2D Allen-Cahn: sqrt(m) x sqrt(m) periodic spatial grid, (x, y) only.
    # PIT keeps sensor_dim < query_dim (sensors carry no time channel), so
    # this is 2D even though the query is (x, y, t).
    r = int(round(np.sqrt(n_sensors)))
    g = np.linspace(0.0, 1.0, r, endpoint=False)
    X, Y = np.meshgrid(g, g, indexing="ij")
    return np.stack([X.ravel(), Y.ravel()], axis=-1).astype(np.float32)
