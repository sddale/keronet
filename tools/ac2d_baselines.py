"""Benchmark checks for the 2D Allen-Cahn test set (pdes/allen_cahn_2d.py) against a 256^2, dt = 1e-3 solve.

Prints mean/median/max of the test set's reference error, the linearized-PDE and frozen-IC baselines,
a 9x9-sensor kriging floor, and saturation fractions.
Usage: uv run tools/ac2d_baselines.py [--workers 8] [--n 100] [--out FILE.json] [--cache FILE.npz]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the repository root

TV = np.linspace(0.0, 1.0, 101)


def grid(n):
    g = np.linspace(0.0, 1.0, n, endpoint=False)
    X, Y = np.meshgrid(g, g, indexing="ij")
    return np.stack([X.ravel(), Y.ravel()], -1)


def field(modes, a, b, pts):
    th = 2 * np.pi * pts @ modes.T
    return np.cos(th) @ a + np.sin(th) @ b


def etdrk4(u0, tspan, eps2, dt):
    """numpy port of pdes/allen_cahn_2d.py's ETDRK4 solver (same scheme and contour quadrature)."""
    n = u0.shape[0]
    k1 = 2.0 * np.pi * np.fft.fftfreq(n, d=1.0 / n)
    KX, KY = np.meshgrid(k1, k1, indexing="ij")
    L = (-eps2 * (KX ** 2 + KY ** 2)).ravel()
    M = 16
    r = np.exp(1j * np.pi * (np.arange(1, M + 1) - 0.5) / M)
    LR = dt * L[:, None] + r[None, :]
    Q = dt * ((np.exp(LR / 2.0) - 1.0) / LR).mean(axis=1).real
    f1 = dt * ((-4.0 - LR + np.exp(LR) * (4.0 - 3.0 * LR + LR ** 2)) / LR ** 3).mean(axis=1).real
    f2 = dt * ((2.0 + LR + np.exp(LR) * (-2.0 + LR)) / LR ** 3).mean(axis=1).real
    f3 = dt * ((-4.0 - 3.0 * LR - LR ** 2 + np.exp(LR) * (4.0 - LR)) / LR ** 3).mean(axis=1).real
    E, E2 = np.exp(dt * L), np.exp(dt * L / 2.0)

    def N_hat(v_hat):
        v = np.fft.ifft2(v_hat.reshape(n, n)).real
        return np.fft.fft2(v - v ** 3).ravel()

    v = np.fft.fft2(u0).ravel()
    out = np.zeros((len(tspan), n, n))
    out[0] = u0
    steps_per = int(round((tspan[1] - tspan[0]) / dt))
    for j in range(1, len(tspan)):
        for _ in range(steps_per):
            Nv = N_hat(v)
            a = E2 * v + Q * Nv
            Na = N_hat(a)
            b_ = E2 * v + Q * Na
            Nb = N_hat(b_)
            c_ = E2 * a + Q * (2.0 * Nb - Nv)
            Nc = N_hat(c_)
            v = E * v + f1 * Nv + 2.0 * f2 * (Na + Nb) + f3 * Nc
        out[j] = np.fft.ifft2(v.reshape(n, n)).real
    return out


def linear_evolve(u0, eps2):
    n = u0.shape[0]
    k = 2 * np.pi * np.fft.fftfreq(n, d=1.0 / n)
    K2 = k[:, None] ** 2 + k[None, :] ** 2
    U = np.fft.fft2(u0)
    return np.stack([np.fft.ifft2(U * np.exp((1 - eps2 * K2) * t)).real for t in TV])


def bilinear_to_64(traj):
    """Periodic bilinear interpolation onto the 64^2 evaluation grid (as the test set)."""
    s = traj.shape[-1]
    pts = grid(64).astype(np.float32)
    xi, yi = (pts[:, 0] * s) % s, (pts[:, 1] * s) % s
    x0, y0 = np.floor(xi).astype(int) % s, np.floor(yi).astype(int) % s
    x1, y1 = (x0 + 1) % s, (y0 + 1) % s
    fx, fy = (xi - np.floor(xi))[None, :], (yi - np.floor(yi))[None, :]
    F = traj.reshape(len(traj), -1)
    i = lambda a, b: a * s + b
    return (F[:, i(x0, y0)] * (1 - fx) * (1 - fy) + F[:, i(x1, y0)] * fx * (1 - fy)
            + F[:, i(x0, y1)] * (1 - fx) * fy + F[:, i(x1, y1)] * fx * fy)


def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def work(job):
    i, modes, a, b, scale, ah, bh, eps2, s_fine, dt = job
    ic = lambda s, aa, bb: np.tanh(scale * field(modes, aa, bb, grid(s)).reshape(s, s))
    fine = etdrk4(ic(256, a, b), TV, eps2, 1e-3)[:, ::4, ::4].reshape(len(TV), -1)
    bench = bilinear_to_64(etdrk4(ic(s_fine, a, b), TV, eps2, dt))
    lin = linear_evolve(ic(256, a, b), eps2)[:, ::4, ::4].reshape(len(TV), -1)
    krig = etdrk4(ic(256, ah, bh), TV, eps2, 1e-3)[:, ::4, ::4].reshape(len(TV), -1)
    r = {"reference": rel(bench, fine), "linear": rel(lin, fine),
         "freeze": rel(np.repeat(fine[:1], len(TV), 0), fine),
         "sign": rel(np.repeat(np.sign(fine[:1]), len(TV), 0), fine),
         "sat0": float(np.mean(np.abs(fine[0]) > 0.9)), "sat1": float(np.mean(np.abs(fine[-1]) > 0.9)),
         "floor_9x9": rel(krig, fine), "ic_krig": rel(krig[0], fine[0])}
    return i, r, bench.astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--seed", type=int, default=7654321)
    p.add_argument("--cache", default=None, help="cached test set .npz to compare the reference against")
    p.add_argument("--out", default=None)
    A = p.parse_args()

    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    from jax import random

    from pdes import get_pde
    from pdes._periodic2d import half_plane_modes, sample_GRF2d_coefs
    from pdes.allen_cahn_2d import _grf_field_std_np, _sensor_grid

    spec = get_pde("allen_cahn_2d")
    pp, tk = spec.params, spec.default_test_kwargs
    gamma, tau, sigma, eps2 = pp["gamma"], pp["tau"], pp["sigma"], pp["eps2"]
    modes = half_plane_modes(int(pp["n_modes"]))
    mf = np.asarray(modes).astype(float)
    scale = pp["ic_amp"] / _grf_field_std_np(np.asarray(modes), gamma, tau, sigma)
    # GRF prior variance per cosine/sine coefficient, and the 9x9 kriging operator
    lam2 = (np.sqrt(2.0) * abs(sigma) * ((2 * np.pi) ** 2 * np.sum(mf ** 2, 1) + tau ** 2) ** (-gamma / 2.0)) ** 2
    Lam = np.concatenate([lam2, lam2])
    th = 2 * np.pi * _sensor_grid(81).astype(float) @ mf.T
    Phi = np.concatenate([np.cos(th), np.sin(th)], 1)
    G = (Phi * Lam) @ Phi.T
    K = (Lam[:, None] * Phi.T) @ np.linalg.inv(G + 1e-12 * np.trace(G) / 81 * np.eye(81))
    keys = random.split(random.PRNGKey(A.seed), 100)
    jobs = []
    for i in range(A.n):
        a, b = (np.asarray(v).astype(float) for v in sample_GRF2d_coefs(keys[i], modes, gamma, tau, sigma))
        ch = K @ (Phi @ np.concatenate([a, b]))
        jobs.append((i, mf, a, b, scale, ch[: len(a)], ch[len(a):], eps2, int(tk["s_fine"]), float(tk["dt"])))

    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=A.workers, mp_context=mp.get_context("spawn")) as ex:
        res = sorted(ex.map(work, jobs, chunksize=1), key=lambda r: r[0])
    print(f"{len(res)} test functions in {time.time() - t0:.0f}s", flush=True)
    recs = [r[1] for r in res]
    if A.cache:
        s = np.load(A.cache)["arr2"].reshape(100, -1)[: A.n]
        d = np.abs(s - np.stack([r[2] for r in res]).reshape(A.n, -1))
        print(f"benchmark reference recomputed vs cached test set: max |diff| = {d.max():.2e}")
    for k in recs[0]:
        v = np.array([r[k] for r in recs])
        f, u = (1, "") if k.startswith("sat") else (100, "%")
        print(f"  {k:12s} mean {f * v.mean():8.3f}{u}  median {f * np.median(v):8.3f}{u}  max {f * v.max():8.3f}{u}")
    if A.out:
        json.dump(recs, open(A.out, "w"), indent=1)


if __name__ == "__main__":
    main()
