"""Error-map diagnostics for one advection test function and one trained model.

Reads <results_root>/<pde_dir> (ground truth, <arch>_seed<seed> predictions) and writes
raw_png.png, raw_svg.svg, raw_pdf.pdf, slice_t_half.pdf and error_fft.pdf to --out_dir,
to tell whether a checkerboard in the error map is real or a rendering artifact.
Usage: uv run plotting/inspect_advection.py --results_root results --pde_dir advection --arch keronet --seed 0 --ic_index 28 --out_dir figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm

from paper_style import (
    ARCH_DISPLAY,
    apply_paper_style,
    fig_size,
    save_fig,
)

apply_paper_style()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_root", type=Path, required=True)
    p.add_argument("--pde_dir", type=str, required=True, help="e.g. 'advection_v1'")
    p.add_argument("--arch", type=str, default="keronet")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--ic_index",
        type=int,
        required=True,
        help="Which test IC's error map to diagnose.",
    )
    p.add_argument("--out_dir", type=Path, default=Path("checkerboard_diagnostic"))
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    pde_dir = args.results_root / args.pde_dir
    cell_dir = pde_dir / f"{args.arch}_seed{args.seed}"

    #  Load arrays
    with open(pde_dir / "evaluation_meta.json") as f:
        eval_meta = json.load(f)
    n_test = int(eval_meta["N_test"])
    pts_per_traj = int(eval_meta["pts_per_traj"])
    nn = nt = int(np.sqrt(pts_per_traj))
    if nn * nt != pts_per_traj:
        nn, nt = 101, 101

    gt = np.load(pde_dir / "ground_truth.npy").reshape(n_test, nt, nn)
    preds = np.load(cell_dir / "predictions.npy").reshape(n_test, nt, nn)

    pred = preds[args.ic_index]
    gt_ic = gt[args.ic_index]
    err = np.abs(pred - gt_ic)

    arch_label = ARCH_DISPLAY.get(args.arch, args.arch)

    #  (1) Stats
    print(f"Cell: {cell_dir}")
    print(
        f"IC {args.ic_index}, ground truth shape: {gt_ic.shape}, "
        f"prediction shape: {pred.shape}"
    )
    print(f"GT amplitude:        [{gt_ic.min():+.4f}, {gt_ic.max():+.4f}]")
    print(f"Prediction amplitude: [{pred.min():+.4f}, {pred.max():+.4f}]")
    print(
        f"Error magnitude --- median: {np.median(err):.4e}, "
        f"95th pct: {np.percentile(err, 95):.4e}, max: {err.max():.4e}"
    )

    #  (2) Same data, three output paths
    sol_max = max(np.abs(gt_ic).max(), np.abs(pred).max())
    err_floor = max(1e-8, np.percentile(err[err > 1e-8], 1))
    err_ceil = err.max()

    for fmt, dpi in [("png", 600), ("svg", None), ("pdf", None)]:
        fig, axes = plt.subplots(
            1,
            3,
            figsize=fig_size("col", aspect=0.35),
            gridspec_kw={"wspace": 0.25},
        )
        axes[0].imshow(
            gt_ic,
            origin="lower",
            extent=[0, 1, 0, 1],
            cmap="RdBu_r",
            norm=Normalize(-sol_max, sol_max),
            interpolation="nearest",
        )
        axes[0].set_title("Reference")
        axes[1].imshow(
            pred,
            origin="lower",
            extent=[0, 1, 0, 1],
            cmap="RdBu_r",
            norm=Normalize(-sol_max, sol_max),
            interpolation="nearest",
        )
        axes[1].set_title(f"{arch_label} prediction")
        im = axes[2].imshow(
            np.maximum(err, err_floor),
            origin="lower",
            extent=[0, 1, 0, 1],
            cmap="magma",
            norm=LogNorm(err_floor, err_ceil),
            interpolation="nearest",
        )
        axes[2].set_title("Absolute error")
        for ax in axes:
            ax.set_xlabel("$x$")
            ax.set_ylabel("$t$")
        plt.colorbar(im, ax=axes[2], shrink=0.85)

        # rcParams already sets savefig.bbox = "tight"; only override dpi
        # when we explicitly want the high-res raster comparison.
        out_path = args.out_dir / f"raw_{fmt}.{fmt}"
        if fmt == "png":
            fig.savefig(out_path, dpi=dpi)
        else:
            fig.savefig(out_path)
        plt.close(fig)
        print(f"Wrote {out_path}")

    #  (3) 1D slice through the error map
    fig, axes = plt.subplots(2, 1, figsize=fig_size("col", aspect=0.7), sharex=True)
    t_idx_mid = nt // 2
    axes[0].plot(np.linspace(0, 1, nn), gt_ic[t_idx_mid], label="reference", lw=1.0)
    axes[0].plot(
        np.linspace(0, 1, nn),
        pred[t_idx_mid],
        label=f"{arch_label} prediction",
        lw=1.0,
        ls="--",
    )
    axes[0].set_ylabel("$u(x, t = 0.5)$")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(np.linspace(0, 1, nn), err[t_idx_mid], lw=1.0, color="C3")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("$x$")
    axes[1].set_ylabel("$|u_\\mathrm{pred} - u|$ at $t = 0.5$")
    axes[1].grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    slice_path = save_fig(fig, args.out_dir / "slice_t_half.pdf")
    print(f"Wrote {slice_path}")

    #  (4) FFT of the error map: spectral signature of a checkerboard
    # A true grid-aligned checkerboard concentrates power at the highest
    # spatial/temporal frequencies (k_x, k_t) ~ (Nyquist, Nyquist).
    err_fft = np.fft.fft2(err)
    err_pwr = np.abs(np.fft.fftshift(err_fft)) ** 2
    fig, ax = plt.subplots(figsize=fig_size("col", aspect=0.55))
    im = ax.imshow(np.log10(err_pwr + 1e-20), cmap="magma", interpolation="nearest")
    ax.set_xlabel("$k_x$ index")
    ax.set_ylabel("$k_t$ index")
    ax.set_title(r"$\log_{10}\,|\mathcal{F}(|s_\mathrm{pred} - s|)|^2$ (centered)")
    plt.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fft_path = save_fig(fig, args.out_dir / "error_fft.pdf")
    print(f"Wrote {fft_path}")

    print()
    print("Interpretation:")
    print("  - If the checkerboard is real in the data, error_fft.pdf will")
    print("    show concentrated power at the Nyquist corners (highest k_x, k_t).")
    print("  - If the checkerboard appears in PDF but not in raw_png.png at")
    print("    high dpi or in raw_svg.svg, it's a rasterization moiré artifact.")
    print("  - The 1D slice at t = 0.5 should show whether the prediction")
    print("    has visible high-frequency oscillations that aren't in the")
    print("    reference.")


if __name__ == "__main__":
    main()
