"""Shared matplotlib style and figure helpers for the paper's figures.

apply_paper_style() sets the rcParams, fig_size() sizes figures to the NeurIPS text
width, and ARCH_DISPLAY, ARCH_STYLE and PDE_DISPLAY give display names and each
architecture's color, line style and marker.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt


# Figure-size helpers (NeurIPS sizing)


# NeurIPS 2026 textwidth: 397.48499pt with 1.5-inch margins on a US Letter
# page. Converted at 72.27 pt/in (TeX's pt) for figure sizing.
NEURIPS_TEXTWIDTH_IN = 397.48499 / 72.27  # ~ 5.500 in
# Single-column subfigure width when using a 0.48\linewidth subfigure block:
NEURIPS_SUBCOL_IN = 0.48 * NEURIPS_TEXTWIDTH_IN

# Keys map to common figure widths used in the paper.
_WIDTHS = {
    "col": NEURIPS_TEXTWIDTH_IN,  # one full-width figure
    "subcol": NEURIPS_SUBCOL_IN,  # one subfigure in a 2-up row
    "wide": NEURIPS_TEXTWIDTH_IN,  # alias of "col"; kept for clarity
}


def fig_size(
    kind: Literal["col", "subcol", "wide"] = "col", aspect: float = 0.62
) -> Tuple[float, float]:
    """Return (width, height) in inches for the given kind.

    "col" and "wide" are the text width, "subcol" 0.48 of it; aspect is height / width.
    """
    if kind not in _WIDTHS:
        raise ValueError(f"Unknown fig kind {kind!r}; choose from {list(_WIDTHS)}")
    w = _WIDTHS[kind]
    return (w, w * aspect)


# Architecture and PDE display conventions
ARCH_DISPLAY = {
    "pi_deeponet": "PI-DeepONet",
    "pit": "PIT",
    "pinto": "PINTO",
    "keronet": "KerONet",
    "mkeronet": "mKerONet",
}

# Shared color palette --- one color per architecture, used everywhere the
# architecture appears in a figure. Colorblind-friendly (Wong 2011) with
# KerONet (the proposed method) given the most distinct/saturated hue.
#
# Linestyles let figures stay legible if printed greyscale.
ARCH_STYLE = {
    "pi_deeponet": {"color": "#0072B2", "ls": "-", "marker": "o"},  # blue
    "pit": {"color": "#009E73", "ls": "-", "marker": "s"},  # green
    "pinto": {"color": "#E69F00", "ls": "-", "marker": "^"},  # orange
    "keronet": {"color": "#D55E00", "ls": "-", "marker": "D"},  # vermilion (highlight)
    "mkeronet": {"color": "#CC79A7", "ls": "--", "marker": "d"},  # reddish purple
}

PDE_DISPLAY = {
    "burgers": "Burgers",
    "advection": "Advection",
    "diffusion_reaction": "Diffusion-reaction",
    "allen_cahn": "Allen-Cahn",
    "allen_cahn_2d": "2D Allen-Cahn",
}


# rcParams


# Font fallback order: Times-equivalent serif first, with sane fallbacks for
# environments where the preferred font isn't installed. STIX is matched to
# Computer Modern's metrics for math; combining it with a Times-like serif
# for body text gives a NeurIPS/LaTeX-consistent appearance without
# requiring a working LaTeX install on the user's machine.
_PAPER_RCPARAMS = {
    # Fonts
    "font.family": "serif",
    "font.serif": [
        "Times New Roman",
        "Times",
        "Liberation Serif",
        "Nimbus Roman",
        "STIX Two Text",
        "STIXGeneral",
        "DejaVu Serif",
        "serif",
    ],
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial", "sans-serif"],
    "mathtext.fontset": "stix",
    "mathtext.default": "regular",
    # Sizes (column-width figures need small text)
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "legend.title_fontsize": 7,
    "figure.titlesize": 9,
    # Lines and markers
    "lines.linewidth": 1.1,
    "lines.markersize": 3.5,
    "patch.linewidth": 0.6,
    # Axes / spines
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelpad": 2.5,
    "axes.titlepad": 4.0,
    # Ticks
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
    "xtick.minor.size": 1.4,
    "ytick.minor.size": 1.4,
    "xtick.direction": "out",
    "ytick.direction": "out",
    # Grid (subtle, only when the script asks for it)
    "grid.linewidth": 0.4,
    "grid.color": "0.80",
    "grid.alpha": 0.6,
    # Legend
    "legend.frameon": False,
    "legend.handlelength": 1.6,
    "legend.handletextpad": 0.5,
    "legend.columnspacing": 1.2,
    "legend.borderaxespad": 0.4,
    # Saving
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,  # TrueType embedded; NeurIPS requirement
    "ps.fonttype": 42,
    "pdf.compression": 9,
    "figure.dpi": 150,  # on-screen preview only; savefig overrides
    # Image colormaps
    "image.cmap": "viridis",
}


def apply_paper_style() -> None:
    """Apply the shared paper rcParams (idempotent)."""
    mpl.rcParams.update(_PAPER_RCPARAMS)


# Convenience: save with consistent settings


def save_fig(fig: "plt.Figure", path: str | Path, *, close: bool = True) -> Path:
    """Save fig to path (format from its extension), closing it unless close=False."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    if close:
        plt.close(fig)
    return path


# Self-test (run as a script to render a sanity-check figure)


if __name__ == "__main__":
    import numpy as np

    apply_paper_style()
    fig, ax = plt.subplots(figsize=fig_size("col", aspect=0.5))
    x = np.linspace(0, 10, 200)
    for arch in ("pi_deeponet", "pit", "pinto", "keronet"):
        s = ARCH_STYLE[arch]
        ax.plot(
            x,
            np.sin(x + hash(arch) % 7) * 0.5,
            label=ARCH_DISPLAY[arch],
            color=s["color"],
            linestyle=s["ls"],
            lw=1.2,
        )
    ax.set_xlabel(r"Iteration $t$")
    ax.set_ylabel(r"Loss $\mathcal{L}(\theta)$")
    ax.legend(loc="upper right", ncol=2)
    ax.grid(True, alpha=0.3)
    save_fig(fig, "/tmp/paper_style_test.pdf")
    print("Wrote /tmp/paper_style_test.pdf")
