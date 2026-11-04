"""Solution figures for the 1D PDEs from evaluate.py output directories.

Rows: reference and predictions (the --seed run); input function and absolute errors.
Titles give rel-L2 mean +- std (ddof=1) over seeds. Writes solutions_<pde>_seed<N>.pdf,
or with --n_ics > 1 one solutions_<pde>_seed<N>_rank<R>_ic<I>.pdf per input.
Usage: python plotting/make_pde_figures.py --results_root results --pde_dir_template '{pde}' --pdes burgers --seed 1 --n_ics 1 --ic_selection fixed --ic_index 72 --metric_scope per_ic --out_dir figures
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm

#  Style: NeurIPS-ish
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
        "pdf.fonttype": 42,  # embed TrueType, not Type 3 (NeurIPS requirement)
        "ps.fonttype": 42,
    }
)


PDE_DISPLAY_NAMES = {
    "burgers": "Burgers ($\\nu = 0.01$)",
    "advection": "Advection ($c = 1$)",
    "diffusion_reaction": "Diffusion-reaction ($D = k = 0.01$)",
    "allen_cahn": "1D Allen-Cahn ($\\varepsilon^2 = 10^{-3}$)",
}

# What the input function represents physically per PDE --- used only as
# the y-axis label of the input-function subplot. All other PDEs treat
# the input function as the initial condition u_0(x); diffusion-reaction
# treats it as the forcing f(x).
PDE_INPUT_LABELS = {
    "burgers": "$u_0(x)$",
    "advection": "$u_0(x)$",
    "diffusion_reaction": "$f(x)$",
    "allen_cahn": "$u_0(x)$",
}

ARCH_DISPLAY_NAMES = {
    "keronet": "KerONet",
    "pi_deeponet": "PI-DeepONet",
    "pinto": "PINTO",
    "pit": "PIT",
    "mkeronet": "mKerONet",
}

# Architecture order = winner-first. Adjust if needed for narrative.
ARCH_ORDER = ["keronet", "mkeronet", "pi_deeponet", "pinto", "pit"]


def load_input_function(
    pde_name: str,
    n_test: int,
    test_seed: int,
    ic_index: int,
    data_cache_dir: Path,
) -> np.ndarray | None:
    """Return one test input's sensor values from the cached test-set .npz, or None."""
    cache_path = data_cache_dir / f"test_{pde_name}_{n_test}_{test_seed}.npz"
    if not cache_path.exists():
        return None
    with np.load(cache_path) as f:
        u_test = np.asarray(f["arr0"])  # (N_test * pts_per_traj, m)
    # u_test was tiled across query points so all rows for one IC are equal.
    pts_per_traj = u_test.shape[0] // n_test
    return np.asarray(u_test[ic_index * pts_per_traj])


#  Version-aware directory resolution


def resolve_version_dirs(
    results_root: Path,
    pde: str,
    pde_dir_template: str,
) -> List[Path]:
    """Return the PDE's result dir, or all its _v<N> versions highest first."""
    # Resolve the user-supplied template against this PDE first to seed the
    # discovery. If the template doesn't end in _v\d+, fall back to the
    # single resolved path.
    base = pde_dir_template.format(pde=pde)
    m = re.match(r"^(?P<stem>.+)_v(?P<ver>\d+)$", base)
    if m is None:
        # Template doesn't follow the _v\d+ convention; treat as single dir.
        single = results_root / base
        return [single] if single.exists() else []

    stem = m.group("stem")
    pattern = re.compile(rf"^{re.escape(stem)}_v(\d+)$")
    candidates = []
    for child in results_root.iterdir():
        if not child.is_dir():
            continue
        match = pattern.match(child.name)
        if match is None:
            continue
        candidates.append((int(match.group(1)), child))
    if not candidates:
        return []
    # Sort descending by version
    candidates.sort(key=lambda x: x[0], reverse=True)
    return [path for _, path in candidates]


def resolve_cell_dir(
    version_dirs: List[Path],
    cell_id: str,
) -> Path | None:
    """Return the first done <vdir>/<cell_id> directory in version_dirs, or None."""
    for vdir in version_dirs:
        cell_dir = vdir / cell_id
        if cell_dir.is_dir() and (cell_dir / "done").exists():
            return cell_dir
    return None


def resolve_pde_metadata_dir(version_dirs: List[Path]) -> Path | None:
    """Return the first dir with evaluation_meta.json and ground_truth.npy, or None."""
    for vdir in version_dirs:
        if (vdir / "evaluation_meta.json").exists() and (
            vdir / "ground_truth.npy"
        ).exists():
            return vdir
    return None


def load_cell(
    cell_dir: Path,
    nt: int,
    nn: int,
    n_test: int,
) -> Tuple[np.ndarray, float, float]:
    """Return a run's predictions (n_test, nt, nn) and meta.json rel-L2 mean and std."""
    if not (cell_dir / "done").exists():
        raise FileNotFoundError(f"Cell {cell_dir} not marked done")

    preds_flat = np.load(cell_dir / "predictions.npy")  # (N_test * nt * nn,)
    expected = n_test * nt * nn
    if preds_flat.size != expected:
        raise ValueError(
            f"{cell_dir}: predictions size {preds_flat.size} != "
            f"N_test * nt * nn = {expected}"
        )
    preds = preds_flat.reshape(n_test, nt, nn)

    with open(cell_dir / "meta.json") as f:
        meta = json.load(f)

    return preds, float(meta["rel_l2_mean"]), float(meta["rel_l2_std"])


def aggregate_across_seeds(
    version_dirs: List[Path],
    arch: str,
) -> Tuple[float, float, int]:
    """Return (mean, std with ddof=1, n_seeds) over seeds of each run's rel_l2_mean."""
    # Collect seed numbers seen across all version dirs.
    seed_to_dir: Dict[int, Path] = {}
    for vdir in version_dirs:
        for cell_dir in sorted(vdir.glob(f"{arch}_seed*")):
            if not (cell_dir / "done").exists():
                continue
            try:
                seed = int(cell_dir.name.split("_seed")[-1])
            except ValueError:
                continue
            # Earlier in version_dirs == higher version. Don't override.
            if seed not in seed_to_dir:
                seed_to_dir[seed] = cell_dir

    per_seed_means = []
    for _, cell_dir in sorted(seed_to_dir.items()):
        meta_path = cell_dir / "meta.json"
        if not meta_path.exists():
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        per_seed_means.append(float(meta["rel_l2_mean"]))
    if not per_seed_means:
        return float("nan"), float("nan"), 0
    arr = np.array(per_seed_means)
    return float(arr.mean()), (float(arr.std(ddof=1)) if arr.size > 1 else 0.0), len(arr)


def aggregate_per_ic_across_seeds(
    version_dirs: List[Path],
    arch: str,
    ic_index: int,
) -> Tuple[float, float, int]:
    """Return (mean, std with ddof=1, n_seeds) over seeds of input ic_index's rel-L2."""
    seed_to_dir: Dict[int, Path] = {}
    for vdir in version_dirs:
        for cell_dir in sorted(vdir.glob(f"{arch}_seed*")):
            if not (cell_dir / "done").exists():
                continue
            try:
                seed = int(cell_dir.name.split("_seed")[-1])
            except ValueError:
                continue
            if seed not in seed_to_dir:
                seed_to_dir[seed] = cell_dir

    per_seed_vals = []
    for _, cell_dir in sorted(seed_to_dir.items()):
        errs_path = cell_dir / "per_ic_errors.npy"
        if not errs_path.exists():
            continue
        errs = np.load(errs_path)
        if ic_index >= errs.shape[0]:
            continue
        per_seed_vals.append(float(errs[ic_index]))
    if not per_seed_vals:
        return float("nan"), float("nan"), 0
    arr = np.array(per_seed_vals)
    return float(arr.mean()), (float(arr.std(ddof=1)) if arr.size > 1 else 0.0), len(arr)


def select_ic_index(
    keronet_errors: np.ndarray,
    n_test: int,
    mode: str,
    user_index: int | None,
) -> int:
    """Return user_index (mode "fixed") or KerONet's median, worst or best input."""
    if mode == "fixed":
        idx = user_index if user_index is not None else 0
        if not (0 <= idx < n_test):
            raise ValueError(f"--ic_index {idx} out of [0, {n_test})")
        return idx
    if mode == "median_keronet":
        # IC closest to KerONet's median per-IC error --- a "typical" case.
        order = np.argsort(keronet_errors)
        return int(order[n_test // 2])
    if mode == "worst_keronet":
        return int(np.argmax(keronet_errors))
    if mode == "best_keronet":
        return int(np.argmin(keronet_errors))
    raise ValueError(f"Unknown ic_selection mode: {mode!r}")


def load_keronet_per_ic_errors(
    version_dirs: List[Path],
    seed: int,
) -> np.ndarray | None:
    """Return per_ic_errors.npy of the first done keronet_seed<seed> run, or None."""
    cell_id = f"keronet_seed{seed}"
    cell_dir = resolve_cell_dir(version_dirs, cell_id)
    if cell_dir is None:
        return None
    errs_path = cell_dir / "per_ic_errors.npy"
    if not errs_path.exists():
        return None
    return np.load(errs_path)


def select_ic_indices_evenly_spaced(
    keronet_errors: np.ndarray,
    n_ics: int,
) -> List[Tuple[int, int]]:
    """Return (rank, ic_index) at n_ics evenly spaced KerONet error ranks (0 = best)."""
    if n_ics < 1:
        raise ValueError(f"n_ics must be >= 1, got {n_ics}")
    n_test = keronet_errors.shape[0]
    if n_ics > n_test:
        raise ValueError(f"n_ics={n_ics} > n_test={n_test}")
    order = np.argsort(keronet_errors)  # ascending: best first
    if n_ics == 1:
        ranks = [n_test // 2]
    else:
        # Evenly spaced ranks from 0 to n_test-1, inclusive of both ends.
        ranks = np.linspace(0, n_test - 1, n_ics).round().astype(int).tolist()
        # Deduplicate while preserving order (in case rounding collisions
        # happen for very small n_test).
        seen = set()
        deduped = []
        for r in ranks:
            if r not in seen:
                seen.add(r)
                deduped.append(r)
        ranks = deduped
    return [(int(rank), int(order[rank])) for rank in ranks]


def plot_pde_row(
    pde_name: str,
    version_dirs: List[Path],
    seed: int,
    ic_mode: str,
    ic_user_index: int | None,
    out_path: Path,
    data_cache_dir: Path,
    metric_scope: str = "aggregate",
    precomputed_ic: int | None = None,
) -> Tuple[int, Dict[str, float]]:
    """Plot one PDE to out_path; return (ic_index, {arch: rel-L2 mean in its title})."""
    if not version_dirs:
        raise FileNotFoundError(f"No version directories found for {pde_name}")

    #  PDE metadata (highest-version dir that has it)
    meta_dir = resolve_pde_metadata_dir(version_dirs)
    if meta_dir is None:
        raise FileNotFoundError(
            f"No version dir for {pde_name} contains both "
            f"evaluation_meta.json and ground_truth.npy "
            f"(searched: {[v.name for v in version_dirs]})"
        )

    with open(meta_dir / "evaluation_meta.json") as f:
        eval_meta = json.load(f)
    n_test = int(eval_meta["N_test"])
    pts_per_traj = int(eval_meta["pts_per_traj"])
    # nt = steps + 1, nn = spatial grid; both 101 in our setup
    nn = nt = int(np.sqrt(pts_per_traj))
    if nn * nt != pts_per_traj:
        # Fallback: assume nn=101, nt=101
        nn, nt = 101, 101

    #  Ground truth (from the same dir as evaluation_meta.json)
    gt_flat = np.load(meta_dir / "ground_truth.npy")
    if gt_flat.size != n_test * pts_per_traj:
        raise ValueError(f"ground_truth size {gt_flat.size} != n_test * pts_per_traj")
    gt = gt_flat.reshape(n_test, nt, nn)

    #  Per-cell predictions (this seed) + multi-seed aggregates (for titles)
    cell_data: Dict[str, np.ndarray] = {}
    cell_metrics: Dict[str, Tuple[float, float, int]] = {}
    cell_dir_used: Dict[str, Path] = {}
    for arch in ARCH_ORDER:
        cell_id = f"{arch}_seed{seed}"
        cell_dir = resolve_cell_dir(version_dirs, cell_id)
        if cell_dir is None:
            print(
                f"  [warn] {cell_id} not found in any version of {pde_name}; "
                f"skipping {arch}"
            )
            continue
        preds, _, _ = load_cell(cell_dir, nt, nn, n_test)
        cell_data[arch] = preds
        cell_dir_used[arch] = cell_dir
        agg_mean, agg_std, n_seeds = aggregate_across_seeds(version_dirs, arch)
        cell_metrics[arch] = (agg_mean, agg_std, n_seeds)
        # Show which version dir this seed came from for transparency
        version_tag = cell_dir.parent.name
        print(
            f"        {arch}: shown seed {seed} from {version_tag}; "
            f"aggregated over {n_seeds} seeds  "
            f"-> {100 * agg_mean:.2f}% ± {100 * agg_std:.2f}%"
        )

    if not cell_data:
        raise FileNotFoundError(
            f"No architecture cells found for {pde_name} in any of "
            f"{[v.name for v in version_dirs]}"
        )

    #  IC selection (uses KerONet's per-IC errors if available)
    keronet_per_ic = None
    if "keronet" in cell_data:
        kero_errs_path = cell_dir_used["keronet"] / "per_ic_errors.npy"
        if kero_errs_path.exists():
            keronet_per_ic = np.load(kero_errs_path)
    if keronet_per_ic is None:
        # Fallback: compute from predictions
        kero_pred = next(iter(cell_data.values()))
        diff = (kero_pred - gt).reshape(n_test, -1)
        gtn = gt.reshape(n_test, -1)
        keronet_per_ic = np.linalg.norm(diff, axis=1) / (
            np.linalg.norm(gtn, axis=1) + 1e-12
        )

    if precomputed_ic is not None:
        if not (0 <= precomputed_ic < n_test):
            raise ValueError(f"precomputed_ic={precomputed_ic} out of [0, {n_test})")
        ic = int(precomputed_ic)
    else:
        ic = select_ic_index(keronet_per_ic, n_test, ic_mode, ic_user_index)

    #  Build the figure
    archs_present = [a for a in ARCH_ORDER if a in cell_data]
    n_arch = len(archs_present)

    #  Optionally recompute metrics for the specific IC
    # The aggregate metrics computed earlier are mean +/- std across all
    # 100 ICs and all seeds. If the user asked for per-IC scope, replace
    # them with mean +/- std across seeds at this single IC.
    if metric_scope == "per_ic":
        for arch in archs_present:
            mean_ic, std_ic, n_seeds_ic = aggregate_per_ic_across_seeds(
                version_dirs, arch, ic
            )
            if n_seeds_ic == 0:
                # Fall back to the aggregate if we can't load per_ic_errors.npy
                print(
                    f"  [warn] {arch}: no per_ic_errors.npy found across seeds; "
                    f"keeping aggregate metric in title"
                )
                continue
            cell_metrics[arch] = (mean_ic, std_ic, n_seeds_ic)
            print(
                f"        {arch}: per-IC at #{ic} across {n_seeds_ic} seeds "
                f"-> {100 * mean_ic:.2f}% ± {100 * std_ic:.2f}%"
            )

    # Layout: 2 rows x (1 + n_arch) columns
    #   Row 0: ground truth (col 0), then predictions per arch
    #   Row 1: input function (col 0), then absolute error per arch
    # The bottom-left cell is a 1D line plot; the rest are imshow panels.
    fig_w = 1.55 * (1 + n_arch) + 0.9  # +cbar gutter
    fig_h = 3.1
    fig, axes = plt.subplots(
        2,
        1 + n_arch,
        figsize=(fig_w, fig_h),
        gridspec_kw={
            "wspace": 0.20,
            "hspace": 0.30,
            "left": 0.055,
            "right": 0.895,
            "top": 0.86,
            "bottom": 0.13,
        },
    )

    # Domain extents
    extent = [0.0, 1.0, 0.0, 1.0]  # [xmin, xmax, tmin, tmax]
    aspect = "auto"

    # Colormaps
    cmap_sol = "RdBu_r"
    cmap_err = "magma"

    # Solution colour-norm: shared across GT + all predictions for this PDE
    gt_ic = gt[ic]
    pred_stack = np.stack([cell_data[a][ic] for a in archs_present], axis=0)
    sol_max = max(np.abs(gt_ic).max(), np.abs(pred_stack).max())
    sol_norm = Normalize(vmin=-sol_max, vmax=sol_max)

    # Error norm: log scale, common across architectures for this PDE
    err_stack = np.abs(pred_stack - gt_ic[None])
    eps = 1e-8
    err_floor = max(eps, np.percentile(err_stack[err_stack > eps], 1))
    err_ceiling = max(err_floor * 10, err_stack.max())
    err_norm = LogNorm(vmin=err_floor, vmax=err_ceiling)

    #  Row 0, col 0: ground truth
    ax = axes[0, 0]
    im_sol = ax.imshow(
        gt_ic,
        origin="lower",
        extent=extent,
        aspect=aspect,
        norm=sol_norm,
        cmap=cmap_sol,
        interpolation="nearest",
    )
    ax.set_title("Reference", pad=2)
    ax.set_ylabel("$t$", labelpad=1)
    ax.tick_params(axis="x", labelbottom=False)

    #  Row 1, col 0: input function $u_0(x)$ or $f(x)$
    test_seed = int(eval_meta.get("test_seed", 7_654_321))
    sensor_vals = load_input_function(
        pde_name=pde_name,
        n_test=n_test,
        test_seed=test_seed,
        ic_index=ic,
        data_cache_dir=data_cache_dir,
    )
    ax_input = axes[1, 0]
    if sensor_vals is not None:
        m = sensor_vals.shape[0]
        x_sensors = np.linspace(0.0, 1.0, m)
        ax_input.plot(x_sensors, sensor_vals, color="black", lw=1.0)
        ax_input.axhline(0.0, color="0.7", lw=0.4, zorder=0)
        ax_input.set_xlim(0.0, 1.0)
        ax_input.set_xlabel("$x$", labelpad=1)
        ax_input.set_ylabel(PDE_INPUT_LABELS.get(pde_name, "$u_0(x)$"), labelpad=1)
        ax_input.set_title("Input function", pad=2)
        # Modest tick density to match the imshow row visually
        ax_input.tick_params(labelsize=6)
        for spine in ("top", "right"):
            ax_input.spines[spine].set_visible(False)
    else:
        # Test-set cache not found: leave the input-function panel blank.
        ax_input.axis("off")
        print(
            f"  [warn] test-set cache not found at "
            f"{data_cache_dir}/test_{pde_name}_{n_test}_{test_seed}.npz; "
            f"input-function panel left blank"
        )

    #  Architecture columns
    for j, arch in enumerate(archs_present, start=1):
        pred = cell_data[arch][ic]
        m_mean, m_std, _ = cell_metrics[arch]

        # Solution prediction
        ax = axes[0, j]
        ax.imshow(
            pred,
            origin="lower",
            extent=extent,
            aspect=aspect,
            norm=sol_norm,
            cmap=cmap_sol,
            interpolation="nearest",
        )
        title = (
            f"{ARCH_DISPLAY_NAMES[arch]}\n${100 * m_mean:.2f}\\pm{100 * m_std:.2f}\\%$"
        )
        ax.set_title(title, pad=2)
        ax.tick_params(axis="x", labelbottom=False)
        ax.tick_params(axis="y", labelleft=False)

        # Absolute error
        ax = axes[1, j]
        err = np.abs(pred - gt_ic)
        im_err = ax.imshow(
            np.maximum(err, err_floor),
            origin="lower",
            extent=extent,
            aspect=aspect,
            norm=err_norm,
            cmap=cmap_err,
            interpolation="nearest",
        )
        if j == 1:
            # First error column keeps its y-tick labels (so the reader
            # can read off t values) and labels the x-axis. We don't
            # repeat the "$t$" axis label here because the reference
            # panel directly above already labels its y-axis as t and
            # the input-function panel to the left labels its x-axis,
            # so an extra "$t$" here just collides with the f(x) label.
            ax.set_xlabel("$x$", labelpad=1)
        else:
            ax.tick_params(axis="x", labelbottom=False)
            ax.tick_params(axis="y", labelleft=False)

    #  Colorbars
    # Solution colorbar: spans the right edge of row 0
    cax_sol = fig.add_axes([0.91, 0.51, 0.011, 0.34])
    cb_sol = fig.colorbar(im_sol, cax=cax_sol)
    cb_sol.ax.tick_params(labelsize=6, pad=1)
    cb_sol.set_label("$s(x, t)$", fontsize=7, labelpad=2)

    # Error colorbar: spans the right edge of row 1
    cax_err = fig.add_axes([0.91, 0.13, 0.011, 0.34])
    cb_err = fig.colorbar(im_err, cax=cax_err)
    cb_err.ax.tick_params(labelsize=6, pad=1)
    cb_err.set_label("$|s_\\mathrm{pred} - s|$", fontsize=7, labelpad=2)

    #  Suptitle
    pde_label = PDE_DISPLAY_NAMES.get(pde_name, pde_name)
    # Use the first arch present to learn the seed count
    first_arch = archs_present[0]
    n_seeds = cell_metrics[first_arch][2]
    if metric_scope == "per_ic":
        scope_str = f"per-arch numbers for IC #{ic} across {n_seeds} seeds"
    else:
        scope_str = f"per-arch numbers aggregated over {n_seeds} seeds"
    fig.suptitle(
        pde_label,
        fontsize=9,
        y=0.985,
    )

    fig.savefig(out_path)
    plt.close(fig)
    return ic, {a: cell_metrics[a][0] for a in archs_present}


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--results_root",
        type=Path,
        required=True,
        help="Directory containing per-PDE evaluation outputs.",
    )
    p.add_argument(
        "--pde_dir_template",
        type=str,
        default="{pde}_v1",
        help="Subdir name pattern for each PDE's evaluation "
        "output. Default '{pde}_v1'. The trailing '_v\\d+' "
        "suffix is treated as a version marker: the script "
        "discovers all sibling directories matching the same "
        "stem with any integer version suffix, and for each "
        "(architecture, seed) cell uses the highest-version "
        "directory containing it. PDE-level metadata "
        "(evaluation_meta.json, ground_truth.npy) is read "
        "from the highest-version directory containing both.",
    )
    p.add_argument(
        "--pdes",
        nargs="+",
        default=["burgers", "advection", "diffusion_reaction", "allen_cahn"],
        help="PDE names to include. Default: all four "
        "(burgers, advection, diffusion_reaction, allen_cahn).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Which seed's checkpoint to visualize (default 0).",
    )
    p.add_argument(
        "--ic_selection",
        type=str,
        default="median_keronet",
        choices=["fixed", "median_keronet", "worst_keronet", "best_keronet"],
        help="Strategy for picking the test IC to plot.",
    )
    p.add_argument(
        "--ic_index",
        type=int,
        default=None,
        help="If --ic_selection=fixed, which IC index to use.",
    )
    p.add_argument(
        "--n_ics",
        type=int,
        default=11,
        help="If > 1, ignore --ic_selection and emit n_ics "
        "separate figures per PDE, with ICs sampled "
        "evenly along KerONet's per-IC error percentile "
        "(rank order). Always includes the best (rank 0) "
        "and worst (rank N-1) as endpoints. n_ics=1 falls "
        "back to the single-IC mode controlled by "
        "--ic_selection. Default 11.",
    )
    p.add_argument(
        "--metric_scope",
        type=str,
        default="aggregate",
        choices=["aggregate", "per_ic"],
        help="What error number to display in each architecture's "
        "panel title. 'aggregate' (default) shows the mean "
        "over all 100 test ICs and all seeds (matches the "
        "headline-table convention). 'per_ic' shows the "
        "mean over seeds at the specific IC being plotted, "
        "± std across seeds at that IC.",
    )
    p.add_argument(
        "--out_dir",
        type=Path,
        default=Path("figures"),
        help="Where to write solutions_{pde}_seed{N}.pdf files.",
    )
    p.add_argument(
        "--data_cache_dir",
        type=Path,
        default=Path("data_cache"),
        help="Directory holding the test-set cache files written "
        "by tune_common (test_{pde}_{N_test}_{seed}.npz). "
        "Used to source the input-function $u_0(x)$ for the "
        "bottom-left subplot. If the cache is missing, the "
        "subplot is left blank.",
    )
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Per-PDE figures
    per_pde_paths: List[Path] = []
    for pde in args.pdes:
        version_dirs = resolve_version_dirs(
            args.results_root,
            pde,
            args.pde_dir_template,
        )
        if not version_dirs:
            print(
                f"[skip] {pde}: no version directories found "
                f"(template: {args.pde_dir_template.format(pde=pde)})"
            )
            continue
        if len(version_dirs) == 1:
            print(f"[plot] {pde}  ({version_dirs[0].name})")
        else:
            print(
                f"[plot] {pde}  (versions: "
                f"{', '.join(v.name for v in version_dirs)}, "
                f"highest first)"
            )

        if args.n_ics > 1:
            # Multi-IC mode: pick evenly-spaced ICs by KerONet rank, emit
            # one figure per IC.
            kero_errs = load_keronet_per_ic_errors(version_dirs, args.seed)
            if kero_errs is None:
                print(
                    f"  [skip] {pde}: KerONet per_ic_errors.npy not found "
                    f"for seed {args.seed}; cannot rank ICs for multi-IC mode"
                )
                continue
            picks = select_ic_indices_evenly_spaced(kero_errs, args.n_ics)
            n_test = kero_errs.shape[0]
            # Width for zero-padding the IC index in filenames
            ic_width = len(str(n_test - 1))
            print(
                f"        emitting {len(picks)} figures, IC ranks: "
                f"{[r for r, _ in picks]}"
            )
            for rank, ic in picks:
                err_pct = 100.0 * float(kero_errs[ic])
                # Filename includes both the rank (so files sort by
                # difficulty) and the absolute IC index (so the reader can
                # cross-reference the test-set IC).
                out_path = (
                    args.out_dir / f"solutions_{pde}_seed{args.seed}"
                    f"_rank{rank:0{ic_width}d}_ic{ic:0{ic_width}d}.pdf"
                )
                ic_used, _ = plot_pde_row(
                    pde_name=pde,
                    version_dirs=version_dirs,
                    seed=args.seed,
                    ic_mode=args.ic_selection,  # ignored when precomputed_ic set
                    ic_user_index=args.ic_index,
                    out_path=out_path,
                    data_cache_dir=args.data_cache_dir,
                    metric_scope=args.metric_scope,
                    precomputed_ic=ic,
                )
                print(
                    f"        rank {rank}/{n_test - 1}  IC #{ic_used}  "
                    f"(KerONet rel-L2 = {err_pct:.2f}%)  ->  {out_path.name}"
                )
                per_pde_paths.append(out_path)
        else:
            # Single-IC mode (selected by --ic_selection)
            out_path = args.out_dir / f"solutions_{pde}_seed{args.seed}.pdf"
            ic_used, _ = plot_pde_row(
                pde_name=pde,
                version_dirs=version_dirs,
                seed=args.seed,
                ic_mode=args.ic_selection,
                ic_user_index=args.ic_index,
                out_path=out_path,
                data_cache_dir=args.data_cache_dir,
                metric_scope=args.metric_scope,
            )
            print(f"        IC #{ic_used}  ->  {out_path}")
            per_pde_paths.append(out_path)

    if not per_pde_paths:
        print("No PDEs were plotted.")
        return

    print(f"\nDone. Per-PDE PDFs in {args.out_dir}/")
    print("To concatenate into one file:")
    print(
        f"    pdfunite {args.out_dir}/solutions_*_seed{args.seed}.pdf "
        f"{args.out_dir}/solutions_seed{args.seed}_combined.pdf"
    )


if __name__ == "__main__":
    main()
