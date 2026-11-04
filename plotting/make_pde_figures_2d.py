"""Solution figures for 2D Allen-Cahn from evaluate.py output directories.

Rows: the reference at --n_times times, then each architecture's absolute error (the
--seed run) with the reference s = 0 contour. Labels give rel-L2 mean +- std (ddof=1)
over seeds. Writes solutions2d_<pde>_seed<N>[_rank<R>_ic<I>].pdf.
Usage: uv run plotting/make_pde_figures_2d.py --results_root results --pde_dir_template '{pde}' --seed 0 --out_dir figures
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the repository root
from make_pde_figures import (  # noqa: E402
    ARCH_DISPLAY_NAMES,
    ARCH_ORDER,
    aggregate_across_seeds,
    aggregate_per_ic_across_seeds,
    load_keronet_per_ic_errors,
    resolve_cell_dir,
    resolve_pde_metadata_dir,
    resolve_version_dirs,
    select_ic_index,
    select_ic_indices_evenly_spaced,
)

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "lines.linewidth": 1.0,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

PDE_DISPLAY_NAMES_2D = {
    "allen_cahn_2d": "2D Allen-Cahn ($\\varepsilon^2 = 10^{-2}$)",
}


def infer_grid(pde_name: str, pts_per_traj: int) -> Tuple[int, int]:
    """
    Return (nn, n_time) of the spec's test grid; exit if inconsistent with pts_per_traj.
    """
    from pdes import get_pde

    spec = get_pde(pde_name)
    tk = spec.default_test_kwargs
    nn, n_time = int(tk["nn"]), int(tk["steps"]) + 1
    if nn * nn * n_time != pts_per_traj:
        sys.exit(
            f"ABORT: spec grid nn={nn}, steps={n_time - 1} gives "
            f"{nn * nn * n_time} points per trajectory, but "
            f"evaluation_meta.json records pts_per_traj={pts_per_traj}. "
            f"The spec has changed since this evaluation ran; re-evaluate or "
            f"check out the matching spec before plotting."
        )
    return nn, n_time


def load_cell_2d(cell_dir: Path, n_test: int, nn: int, n_time: int) -> np.ndarray:
    """
    Return predictions as (n_test, n_time, nn, nn); [k, t, i, j] is at (x_i, y_j, t).
    """
    if not (cell_dir / "done").exists():
        raise FileNotFoundError(f"Cell {cell_dir} not marked done")
    flat = np.load(cell_dir / "predictions.npy")
    expected = n_test * n_time * nn * nn
    if flat.size != expected:
        raise ValueError(
            f"{cell_dir}: predictions size {flat.size} != "
            f"n_test * n_time * nn^2 = {expected}"
        )
    return flat.reshape(n_test, n_time, nn, nn)


def plot_pde_2d(
    pde_name: str,
    version_dirs: List[Path],
    seed: int,
    ic_mode: str,
    ic_user_index: int | None,
    out_path: Path,
    n_times: int = 4,
    metric_scope: str = "aggregate",
    precomputed_ic: int | None = None,
) -> Tuple[int, Dict[str, float]]:
    """Plot one PDE to out_path; return (ic_index, {arch: rel-L2 mean in its label})."""
    if not version_dirs:
        raise FileNotFoundError(f"No version directories found for {pde_name}")

    meta_dir = resolve_pde_metadata_dir(version_dirs)
    if meta_dir is None:
        raise FileNotFoundError(
            f"No version dir for {pde_name} has both evaluation_meta.json and "
            f"ground_truth.npy (searched: {[v.name for v in version_dirs]})"
        )
    with open(meta_dir / "evaluation_meta.json") as f:
        eval_meta = json.load(f)
    n_test = int(eval_meta["N_test"])
    ppt = int(eval_meta["pts_per_traj"])
    nn, n_time = infer_grid(pde_name, ppt)

    gt_flat = np.load(meta_dir / "ground_truth.npy")
    if gt_flat.size != n_test * ppt:
        raise ValueError(f"ground_truth size {gt_flat.size} != n_test * ppt")
    gt = gt_flat.reshape(n_test, n_time, nn, nn)

    # per-cell predictions + multi-seed aggregates
    cell_data: Dict[str, np.ndarray] = {}
    cell_metrics: Dict[str, Tuple[float, float, int]] = {}
    cell_dir_used: Dict[str, Path] = {}
    for arch in ARCH_ORDER:
        cell_dir = resolve_cell_dir(version_dirs, f"{arch}_seed{seed}")
        if cell_dir is None:
            print(f"  [warn] {arch}_seed{seed} not found; skipping {arch}")
            continue
        cell_data[arch] = load_cell_2d(cell_dir, n_test, nn, n_time)
        cell_dir_used[arch] = cell_dir
        cell_metrics[arch] = aggregate_across_seeds(version_dirs, arch)
        am, asd, ns_ = cell_metrics[arch]
        print(
            f"        {arch}: seed {seed} from {cell_dir.parent.name}; "
            f"{ns_} seeds -> {100 * am:.2f}% +- {100 * asd:.2f}%"
        )
    if not cell_data:
        raise FileNotFoundError(f"No architecture cells found for {pde_name}")

    # IC selection (KerONet's per-IC errors, as in the 1D script)
    kero_per_ic = None
    if "keronet" in cell_dir_used:
        p = cell_dir_used["keronet"] / "per_ic_errors.npy"
        if p.exists():
            kero_per_ic = np.load(p)
    if kero_per_ic is None:
        any_pred = next(iter(cell_data.values()))
        d = (any_pred - gt).reshape(n_test, -1)
        kero_per_ic = np.linalg.norm(d, axis=1) / (
            np.linalg.norm(gt.reshape(n_test, -1), axis=1) + 1e-12
        )
    ic = (
        int(precomputed_ic)
        if precomputed_ic is not None
        else select_ic_index(kero_per_ic, n_test, ic_mode, ic_user_index)
    )
    if not (0 <= ic < n_test):
        raise ValueError(f"IC index {ic} out of [0, {n_test})")

    if metric_scope == "per_ic":
        for arch in list(cell_data):
            mi, si, ni = aggregate_per_ic_across_seeds(version_dirs, arch, ic)
            if ni:
                cell_metrics[arch] = (mi, si, ni)

    archs = [a for a in ARCH_ORDER if a in cell_data]
    n_arch = len(archs)

    # time slices: always include t=0 and t=1
    t_idx = np.linspace(0, n_time - 1, n_times).round().astype(int)
    t_idx = sorted(set(int(k) for k in t_idx))
    tvals = np.linspace(0.0, 1.0, n_time)

    gt_ic = gt[ic]  # (n_time, nn, nn)
    err = {a: np.abs(cell_data[a][ic] - gt_ic) for a in archs}

    # Shared norms. The error norm MUST span all architectures and all times:
    # per-panel rescaling would erase the temporal-growth signal the figure
    # exists to show.
    sol_max = float(np.abs(gt_ic).max())
    sol_norm = Normalize(vmin=-sol_max, vmax=sol_max)
    all_err = np.concatenate([err[a][t_idx].ravel() for a in archs])
    pos = all_err[all_err > 1e-10]
    err_floor = float(np.percentile(pos, 1)) if pos.size else 1e-8
    err_ceil = max(err_floor * 10, float(all_err.max()))
    err_norm = LogNorm(vmin=err_floor, vmax=err_ceil)

    n_cols = len(t_idx)
    fig_w = 1.35 * n_cols + 1.55
    fig_h = 1.35 * (1 + n_arch) + 0.55
    fig, axes = plt.subplots(
        1 + n_arch,
        n_cols,
        figsize=(fig_w, fig_h),
        squeeze=False,
        gridspec_kw={
            "wspace": 0.08,
            "hspace": 0.10,
            "left": 0.105,
            "right": 0.875,
            "top": 0.90,
            "bottom": 0.065,
        },
    )
    extent = [0.0, 1.0, 0.0, 1.0]
    g = np.linspace(0.0, 1.0, nn, endpoint=False)  # periodic grid

    # -- reference row ----------------------------------------------------
    for c, k in enumerate(t_idx):
        ax = axes[0, c]
        # .T because imshow puts the first array index on the vertical axis,
        # and F[t, i, j] has i = x, j = y.
        im_sol = ax.imshow(
            gt_ic[k].T,
            origin="lower",
            extent=extent,
            aspect="equal",
            norm=sol_norm,
            cmap="RdBu_r",
            interpolation="nearest",
        )
        ax.set_title(f"$t = {tvals[k]:.2f}$", pad=3)
        ax.set_xticks([])
        ax.set_yticks([])
        if c == 0:
            ax.set_ylabel("Reference", labelpad=4)
    axes[0, 0].text(
        0.03,
        0.03,
        "$= u_0(x,y)$",
        transform=axes[0, 0].transAxes,
        fontsize=6,
        color="0.15",
        bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.0),
    )

    # -- one error row per architecture -----------------------------------
    for r, arch in enumerate(archs, start=1):
        m_mean, m_std, _ = cell_metrics[arch]
        for c, k in enumerate(t_idx):
            ax = axes[r, c]
            im_err = ax.imshow(
                np.maximum(err[arch][k], err_floor).T,
                origin="lower",
                extent=extent,
                aspect="equal",
                norm=err_norm,
                cmap="magma",
                interpolation="nearest",
            )
            # reference interface: where the error lives, if it is
            # interface-localized
            ax.contour(
                g, g, gt_ic[k].T, levels=[0.0], colors="w", linewidths=0.45, alpha=0.85
            )
            ax.set_xticks([])
            ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(
                    f"{ARCH_DISPLAY_NAMES.get(arch, arch)}\n"
                    f"${100 * m_mean:.2f}\\pm{100 * m_std:.2f}\\%$",
                    labelpad=4,
                    fontsize=7,
                )
    for c in range(n_cols):
        axes[-1, c].set_xlabel("$x$", labelpad=1)
    for r in range(1 + n_arch):
        axes[r, 0].set_yticks([])

    # -- colorbars --------------------------------------------------------
    top = axes[0, -1].get_position()
    bot = axes[-1, -1].get_position()
    cax_sol = fig.add_axes([0.89, top.y0, 0.012, top.height])
    cb = fig.colorbar(im_sol, cax=cax_sol)
    cb.ax.tick_params(labelsize=6, pad=1)
    cb.set_label("$s$", fontsize=7, labelpad=2)

    cax_err = fig.add_axes([0.89, bot.y0, 0.012, top.y0 - bot.y0 - 0.035])
    cb2 = fig.colorbar(im_err, cax=cax_err)
    cb2.ax.tick_params(labelsize=6, pad=1)
    cb2.set_label("$|s_\\mathrm{pred} - s|$", fontsize=7, labelpad=2)

    label = PDE_DISPLAY_NAMES_2D.get(pde_name, pde_name)
    n_seeds = cell_metrics[archs[0]][2]
    scope = (
        f"IC #{ic} across {n_seeds} seeds"
        if metric_scope == "per_ic"
        else f"aggregated over {n_seeds} seeds"
    )
    fig.suptitle(
        f"{label} \u2014 test IC #{ic}, seed {seed} shown; {scope}. "
        f"White contour: reference interface $s=0$.",
        fontsize=8.5,
        y=0.975,
    )
    fig.savefig(out_path)
    plt.close(fig)
    return ic, {a: cell_metrics[a][0] for a in archs}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_root", type=Path, required=True)
    p.add_argument("--pde_dir_template", type=str, default="{pde}_v1")
    p.add_argument("--pdes", nargs="+", default=["allen_cahn_2d"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--ic_selection",
        type=str,
        default="median_keronet",
        choices=["fixed", "median_keronet", "worst_keronet", "best_keronet"],
    )
    p.add_argument("--ic_index", type=int, default=None)
    p.add_argument(
        "--n_ics",
        type=int,
        default=1,
        help="If > 1, emit this many figures per PDE with ICs "
        "sampled evenly along KerONet's error rank.",
    )
    p.add_argument(
        "--n_times",
        type=int,
        default=4,
        help="Time slices (columns). Always includes t=0 and t=1.",
    )
    p.add_argument(
        "--metric_scope", type=str, default="aggregate", choices=["aggregate", "per_ic"]
    )
    p.add_argument("--out_dir", type=Path, default=Path("figures"))
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for pde in args.pdes:
        vdirs = resolve_version_dirs(args.results_root, pde, args.pde_dir_template)
        if not vdirs:
            print(f"[skip] {pde}: no version directories found")
            continue
        print(f"[plot] {pde}  (versions: {', '.join(v.name for v in vdirs)})")

        if args.n_ics > 1:
            kero = load_keronet_per_ic_errors(vdirs, args.seed)
            if kero is None:
                print(f"  [skip] {pde}: KerONet per_ic_errors.npy missing")
                continue
            picks = select_ic_indices_evenly_spaced(kero, args.n_ics)
            w = len(str(kero.shape[0] - 1))
            for rank, ic in picks:
                out = (
                    args.out_dir / f"solutions2d_{pde}_seed{args.seed}"
                    f"_rank{rank:0{w}d}_ic{ic:0{w}d}.pdf"
                )
                plot_pde_2d(
                    pde,
                    vdirs,
                    args.seed,
                    args.ic_selection,
                    args.ic_index,
                    out,
                    args.n_times,
                    args.metric_scope,
                    precomputed_ic=ic,
                )
                print(
                    f"        rank {rank}  IC #{ic}  "
                    f"(KerONet {100 * kero[ic]:.2f}%)  ->  {out.name}"
                )
        else:
            out = args.out_dir / f"solutions2d_{pde}_seed{args.seed}.pdf"
            ic, _ = plot_pde_2d(
                pde,
                vdirs,
                args.seed,
                args.ic_selection,
                args.ic_index,
                out,
                args.n_times,
                args.metric_scope,
            )
            print(f"        IC #{ic}  ->  {out}")


if __name__ == "__main__":
    main()
