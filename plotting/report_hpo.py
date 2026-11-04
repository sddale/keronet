"""HPO tables and figures from Optuna studies <studies_dir>/optuna_<arch>_<pde>.db.

Writes search_space_<arch>, best_hparams_<arch> and gpu_time tables (.tex, .md),
convergence_<pde>.pdf (best validation rel-L2 so far, whole study) and, with
--include_importance, fANOVA importance plots and importance_summary.{tex,md}.
Usage: uv run plotting/report_hpo.py --studies_dir DIR --output_dir figures
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the repository root
from plotting.paper_style import (  # noqa: E402
    apply_paper_style,
    fig_size,
    ARCH_STYLE,
    ARCH_DISPLAY,
    PDE_DISPLAY,
    save_fig,
)

apply_paper_style()


# Defaults


DEFAULT_ARCHS = ["keronet", "pinto", "pit", "pi_deeponet", "mkeronet"]
DEFAULT_PDES = [
    "burgers",
    "advection",
    "diffusion_reaction",
    "allen_cahn",
    "allen_cahn_2d",
]

# Pretty hparam names --- fall back to raw name if not listed
HPARAM_DISPLAY = {
    "branch_hidden": "branch hidden",
    "branch_layers_n": "branch layers",
    "trunk_hidden": "trunk hidden",
    "trunk_layers_n": "trunk layers",
    "L": r"$L$",
    "d_k": r"$d_k$",
    "d_v": r"$d_v$",
    "post_mlp_hidden": "post-MLP hidden",
    "mapping_size": "Fourier $m_f$",
    "sigma": r"Fourier $\sigma$",
    "p": r"basis count $p$",
    "d_model": r"$d_{\mathrm{model}}$",
    "n_heads": "heads",
    "key_dim": "key dim",
    "n_cross_attn": "CAUs",
    "n_blocks": "blocks",
    "pe_dim": "PE dim",
    "weight_decay": "weight decay",
    "init_lr": "init LR",
    "end_lr_frac": "LR decay frac",
    "warmup_frac": "warmup frac",
    "bcs_batch_size": "BCS batch",
    "res_batch_size": "residual batch",
}


def display_hparam(name: str) -> str:
    return HPARAM_DISPLAY.get(name, name.replace("_", " "))


# Optuna study loading
def study_path(studies_dir: Path, arch: str, pde: str) -> Path:
    return studies_dir / f"optuna_{arch}_{pde}.db"


def load_study(studies_dir: Path, arch: str, pde: str):
    """Load study <arch>_<pde> (or the file's only study) from optuna_<arch>_<pde>.db, or None."""
    import optuna

    p = study_path(studies_dir, arch, pde)
    if not p.exists():
        print(f"  [skip] {p} not found")
        return None
    storage = f"sqlite:///{p}"
    name = f"{arch}_{pde}"
    names = optuna.get_all_study_names(storage=storage)
    if name not in names and len(names) == 1:
        name = names[0]  # e.g. a released study stored under another name
    return optuna.load_study(study_name=name, storage=storage)


def completed_trials(study) -> list:
    """Return only successfully completed trials, sorted by trial number."""
    import optuna

    return sorted(
        [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE],
        key=lambda t: t.number,
    )


# Search space extraction (Tier 1)
def search_space(study) -> Dict[str, Dict[str, Any]]:
    """Return {hparam: {type, range or choices, log}} from the first completed trial."""
    import optuna

    ct = completed_trials(study)
    if not ct:
        return {}
    distributions = ct[0].distributions
    out = {}
    for name, dist in distributions.items():
        entry: Dict[str, Any] = {}
        if isinstance(dist, optuna.distributions.CategoricalDistribution):
            entry["type"] = "categorical"
            entry["choices"] = list(dist.choices)
        elif isinstance(dist, optuna.distributions.IntDistribution):
            entry["type"] = "int"
            entry["range"] = (dist.low, dist.high)
            entry["log"] = bool(getattr(dist, "log", False))
        elif isinstance(dist, optuna.distributions.FloatDistribution):
            entry["type"] = "float"
            entry["range"] = (dist.low, dist.high)
            entry["log"] = bool(getattr(dist, "log", False))
        else:
            entry["type"] = type(dist).__name__
            entry["raw"] = repr(dist)
        out[name] = entry
    return out


def fmt_value(v: Any) -> str:
    """Format a numeric value for tables."""
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if v == 0:
            return "0"
        if abs(v) < 1e-2 or abs(v) >= 1e4:
            return f"{v:.2e}"
        return f"{v:.4g}"
    return str(v)


def fmt_search_space_entry(entry: Dict[str, Any]) -> str:
    if entry["type"] == "categorical":
        return "{" + ", ".join(fmt_value(c) for c in entry["choices"]) + "}"
    if entry["type"] in ("int", "float"):
        lo, hi = entry["range"]
        scale = " (log)" if entry.get("log") else ""
        return f"[{fmt_value(lo)}, {fmt_value(hi)}]{scale}"
    return entry.get("raw", "?")


def emit_search_space_tables(
    studies: Dict[Tuple[str, str], Any],
    output_dir: Path,
    archs: List[str],
    pdes: List[str],
) -> None:
    """Write search_space_<arch>.{tex,md} from the first PDE that has a study."""
    for arch in archs:
        space = None
        used_pde = None
        for pde in pdes:
            study = studies.get((arch, pde))
            if study is None:
                continue
            sp = search_space(study)
            if not sp:
                continue
            if space is None:
                space = sp
                used_pde = pde
            else:
                if set(sp.keys()) != set(space.keys()):
                    print(
                        f"  [warn] {arch}/{pde}: search space differs from "
                        f"{arch}/{used_pde}; using {used_pde}"
                    )
        if space is None:
            print(f"  [skip] no search space found for {arch}")
            continue

        # Sort hparams for stable ordering
        names = sorted(space.keys())

        # LaTeX
        tex = []
        tex.append(r"\begin{tabular}{lll}")
        tex.append(r"\toprule")
        tex.append(r"Hyperparameter & Type & Range / Choices \\")
        tex.append(r"\midrule")
        for name in names:
            entry = space[name]
            tex.append(
                f"{display_hparam(name)} & {entry['type']} & "
                f"{fmt_search_space_entry(entry)} \\\\"
            )
        tex.append(r"\bottomrule")
        tex.append(r"\end{tabular}")
        tex_path = output_dir / f"search_space_{arch}.tex"
        tex_path.write_text("\n".join(tex) + "\n")
        print(f"  wrote {tex_path}")

        # Markdown
        md = ["| Hyperparameter | Type | Range / Choices |", "|---|---|---|"]
        for name in names:
            entry = space[name]
            md.append(
                f"| {display_hparam(name)} | {entry['type']} | "
                f"{fmt_search_space_entry(entry)} |"
            )
        md_path = output_dir / f"search_space_{arch}.md"
        md_path.write_text("\n".join(md) + "\n")
        print(f"  wrote {md_path}")


# Best hyperparameters table (Tier 1)
def emit_best_hparams_tables(
    studies: Dict[Tuple[str, str], Any],
    output_dir: Path,
    archs: List[str],
    pdes: List[str],
) -> None:
    """Write best_hparams_<arch>.{tex,md}: best trial and completed trials per PDE."""
    for arch in archs:
        cell_data: Dict[str, Dict[str, Any]] = {}
        all_hparams: set = set()
        for pde in pdes:
            study = studies.get((arch, pde))
            if study is None:
                continue
            try:
                best = study.best_trial
            except ValueError:
                continue
            cell_data[pde] = dict(best.params)
            cell_data[pde]["__value__"] = best.value
            cell_data[pde]["__n_trials__"] = len(completed_trials(study))
            all_hparams.update(best.params.keys())

        if not cell_data:
            print(f"  [skip] no completed trials for {arch}")
            continue

        # Stable hparam order
        names = sorted(all_hparams)

        # LaTeX
        tex = []
        col_spec = "l" + "c" * (len(names) + 2)
        tex.append(rf"\begin{{tabular}}{{{col_spec}}}")
        tex.append(r"\toprule")
        header = (
            ["PDE"]
            + [display_hparam(n) for n in names]
            + [r"val.\ rel-$L^2$", "trials"]
        )
        tex.append(" & ".join(header) + r" \\")
        tex.append(r"\midrule")
        for pde in pdes:
            if pde not in cell_data:
                continue
            row = [pde.replace("_", r"\_")]
            for n in names:
                v = cell_data[pde].get(n, "---")
                row.append(fmt_value(v) if v != "---" else "---")
            row.append(fmt_value(cell_data[pde]["__value__"]))
            row.append(str(cell_data[pde]["__n_trials__"]))
            tex.append(" & ".join(row) + r" \\")
        tex.append(r"\bottomrule")
        tex.append(r"\end{tabular}")
        tex_path = output_dir / f"best_hparams_{arch}.tex"
        tex_path.write_text("\n".join(tex) + "\n")
        print(f"  wrote {tex_path}")

        # Markdown
        md = [
            "| PDE | "
            + " | ".join(display_hparam(n) for n in names)
            + " | val. rel-L2 | trials |",
            "|" + "|".join("---" for _ in range(len(names) + 3)) + "|",
        ]
        for pde in pdes:
            if pde not in cell_data:
                continue
            row = [pde]
            for n in names:
                v = cell_data[pde].get(n, "---")
                row.append(fmt_value(v) if v != "---" else "---")
            row.append(fmt_value(cell_data[pde]["__value__"]))
            row.append(str(cell_data[pde]["__n_trials__"]))
            md.append("| " + " | ".join(row) + " |")
        md_path = output_dir / f"best_hparams_{arch}.md"
        md_path.write_text("\n".join(md) + "\n")
        print(f"  wrote {md_path}")


# Convergence traces (Tier 2)
def emit_convergence_plots(
    studies: Dict[Tuple[str, str], Any],
    output_dir: Path,
    archs: List[str],
    pdes: List[str],
) -> None:
    """Write convergence_<pde>.pdf: best validation rel-L2 so far by completed trial."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    for pde in pdes:
        any_data = False
        fig, ax = plt.subplots(figsize=fig_size("col", aspect=0.6))
        for arch in archs:
            study = studies.get((arch, pde))
            if study is None:
                continue
            ct = completed_trials(study)
            if not ct:
                continue
            values = np.array([t.value for t in ct], dtype=float)
            best_so_far = np.minimum.accumulate(values)
            x = np.arange(1, len(best_so_far) + 1)
            s = ARCH_STYLE.get(arch, {"color": None, "ls": "-", "marker": "o"})
            ax.plot(
                x,
                best_so_far,
                label=ARCH_DISPLAY.get(arch, arch),
                color=s["color"],
                linestyle=s["ls"],
                linewidth=1.2,
                marker=s.get("marker", "o"),
                markersize=2.8,
                markevery=max(1, len(x) // 30),
            )
            any_data = True

        if not any_data:
            plt.close(fig)
            continue

        ax.set_yscale("log")
        ax.set_xlabel("Completed trials")
        ax.set_ylabel(r"Best validation rel-$L^2$ so far")
        ax.set_title(PDE_DISPLAY.get(pde, pde))
        ax.grid(True, which="major", alpha=0.3, linewidth=0.4)
        ax.grid(True, which="minor", alpha=0.12, linewidth=0.3)
        ax.xaxis.set_major_locator(MultipleLocator(2))
        ax.set_axisbelow(True)
        ax.legend(loc="upper right")
        fig.tight_layout()
        save_fig(fig, output_dir / f"convergence_{pde}.pdf")
        print(f"  wrote {output_dir / f'convergence_{pde}.pdf'}")


# Hyperparameter importance (Tier 3, opt-in)
def compute_importance(study) -> Dict[str, float]:
    """Return Optuna's default (fANOVA) hyperparameter importances, or {} on failure."""
    import optuna

    try:
        return optuna.importance.get_param_importances(study)
    except Exception as e:
        print(f"  [warn] importance computation failed: {e}")
        return {}


def emit_importance_plots(
    studies: Dict[Tuple[str, str], Any],
    output_dir: Path,
    archs: List[str],
    pdes: List[str],
) -> None:
    """Write per-study importance charts and importance_summary.{tex,md} (top three)."""
    import matplotlib.pyplot as plt

    summary: Dict[Tuple[str, str], List[Tuple[str, float]]] = {}

    for arch in archs:
        for pde in pdes:
            study = studies.get((arch, pde))
            if study is None:
                continue
            ct = completed_trials(study)
            if len(ct) < 5:
                print(
                    f"  [skip] {arch}/{pde}: only {len(ct)} completed trials, "
                    f"importance computation needs more"
                )
                continue

            importances = compute_importance(study)
            if not importances:
                continue

            # Bar chart, sized to fit one figure column with one bar per item.
            items = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)
            bar_color = ARCH_STYLE.get(arch, {"color": "0.5"})["color"]
            row_h = 0.22  # inches per bar
            height = max(1.4, row_h * len(items) + 0.7)
            fig, ax = plt.subplots(figsize=(fig_size("col")[0], height))
            names = [display_hparam(k) for k, _ in items]
            vals = [v for _, v in items]
            ax.barh(
                names[::-1],
                vals[::-1],
                color=bar_color,
                edgecolor=bar_color,
                alpha=0.65,
                linewidth=0.4,
                height=0.7,
            )
            ax.set_xlabel("fANOVA importance")
            ax.set_title(
                f"{ARCH_DISPLAY.get(arch, arch)} --- {PDE_DISPLAY.get(pde, pde)}"
            )
            ax.grid(True, axis="x", which="major", alpha=0.3, linewidth=0.4)
            ax.set_axisbelow(True)
            fig.tight_layout()
            save_fig(fig, output_dir / f"importance_{arch}_{pde}.pdf")
            print(f"  wrote {output_dir / f'importance_{arch}_{pde}.pdf'}")

            summary[(arch, pde)] = items[:3]

    if not summary:
        return

    # Top-3 importance summary table
    tex = []
    tex.append(r"\begin{tabular}{llll}")
    tex.append(r"\toprule")
    tex.append(r"Architecture & PDE & Top hparam (importance) & 2nd / 3rd \\")
    tex.append(r"\midrule")
    md = [
        "| Architecture | PDE | Top hparam (importance) | 2nd / 3rd |",
        "|---|---|---|---|",
    ]
    for (arch, pde), items in sorted(summary.items()):
        first = f"{display_hparam(items[0][0])} ({items[0][1]:.2f})"
        rest = " / ".join(f"{display_hparam(k)} ({v:.2f})" for k, v in items[1:3])
        tex.append(
            f"{ARCH_DISPLAY.get(arch, arch)} & "
            f"{pde.replace('_', chr(92) + '_')} & {first} & {rest} \\\\"
        )
        md.append(f"| {ARCH_DISPLAY.get(arch, arch)} | {pde} | {first} | {rest} |")
    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    (output_dir / "importance_summary.tex").write_text("\n".join(tex) + "\n")
    (output_dir / "importance_summary.md").write_text("\n".join(md) + "\n")
    print(f"  wrote importance_summary.tex / .md")


# GPU time table
def emit_gpu_time_table(
    studies: Dict[Tuple[str, str], Any],
    output_dir: Path,
    archs: List[str],
    pdes: List[str],
    include_over_budget: bool = False,
) -> None:
    """Write gpu_time.{tex,md}: HPO trial counts and time per architecture, all PDEs."""
    import optuna

    rows: List[Dict[str, Any]] = []
    for arch in archs:
        n_trials = 0
        n_completed = 0
        n_pruned_median = 0
        n_pruned_budget = 0
        total_seconds = 0.0
        train_seconds = 0.0  # only completed + median-pruned
        train_count = 0  # completed + median-pruned (for mean)
        any_data = False
        for pde in pdes:
            study = studies.get((arch, pde))
            if study is None:
                continue
            for t in study.trials:
                if t.duration is None:
                    continue
                dur_s = t.duration.total_seconds()
                total_seconds += dur_s
                n_trials += 1
                any_data = True
                if t.state == optuna.trial.TrialState.COMPLETE:
                    n_completed += 1
                    train_seconds += dur_s
                    train_count += 1
                elif t.state == optuna.trial.TrialState.PRUNED:
                    # Distinguish median-pruner kills (reported intermediate
                    # values) from parameter-budget rejections at construction
                    # (no intermediate values).
                    if t.intermediate_values:
                        n_pruned_median += 1
                        train_seconds += dur_s
                        train_count += 1
                    else:
                        n_pruned_budget += 1
        if not any_data:
            continue
        rows.append(
            {
                "arch": arch,
                "n_trials": n_trials,
                "n_completed": n_completed,
                "n_pruned_median": n_pruned_median,
                "n_pruned_budget": n_pruned_budget,
                "total_seconds": total_seconds,
                "train_seconds": train_seconds,
                "train_count": train_count,
                "mean_trained_seconds": (
                    train_seconds / train_count if train_count else 0.0
                ),
            }
        )

    if not rows:
        print("  [skip] no trial-duration data available for GPU time table")
        return

    grand_completed = sum(r["n_completed"] for r in rows)
    grand_pruned_med = sum(r["n_pruned_median"] for r in rows)
    grand_pruned_bud = sum(r["n_pruned_budget"] for r in rows)
    grand_total_s = sum(r["total_seconds"] for r in rows)

    # When over-budget is excluded, the "Trials" column reports trials that
    # actually entered the search space (completed + median-pruned). The
    # Total GPU-hours figure stays the same because over-budget rejections
    # cost negligible compute.
    if include_over_budget:
        for r in rows:
            r["display_trials"] = r["n_trials"]
        grand_display_trials = sum(r["n_trials"] for r in rows)
    else:
        for r in rows:
            r["display_trials"] = r["n_completed"] + r["n_pruned_median"]
        grand_display_trials = grand_completed + grand_pruned_med

    def _hours(s: float) -> str:
        return f"{s / 3600.0:.2f}"

    def _minutes(s: float) -> str:
        return f"{s / 60.0:.1f}"

    #  LaTeX
    col_spec = "lrrrrrr" if include_over_budget else "lrrrrr"
    tex = [r"\begin{tabular}{" + col_spec + "}"]
    tex.append(r"\toprule")
    if include_over_budget:
        tex.append(
            r"Architecture & Trials & Completed & Pruned & Over budget "
            r"& Mean trained trial (min) & Total GPU-hours \\"
        )
    else:
        tex.append(
            r"Architecture & Trials & Completed & Pruned "
            r"& Mean trained trial (min) & Total GPU-hours \\"
        )
    tex.append(r"\midrule")
    for r in rows:
        cells = [
            ARCH_DISPLAY.get(r["arch"], r["arch"]),
            str(r["display_trials"]),
            str(r["n_completed"]),
            str(r["n_pruned_median"]),
        ]
        if include_over_budget:
            cells.append(str(r["n_pruned_budget"]))
        cells.append(_minutes(r["mean_trained_seconds"]))
        cells.append(_hours(r["total_seconds"]))
        tex.append(" & ".join(cells) + r" \\")
    tex.append(r"\midrule")
    total_cells = [
        r"\textbf{Total}",
        rf"\textbf{{{grand_display_trials}}}",
        rf"\textbf{{{grand_completed}}}",
        rf"\textbf{{{grand_pruned_med}}}",
    ]
    if include_over_budget:
        total_cells.append(rf"\textbf{{{grand_pruned_bud}}}")
    total_cells.append("---")
    total_cells.append(rf"\textbf{{{_hours(grand_total_s)}}}")
    tex.append(" & ".join(total_cells) + r" \\")
    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    (output_dir / "gpu_time.tex").write_text("\n".join(tex) + "\n")
    print(f"  wrote {output_dir / 'gpu_time.tex'}")

    #  Markdown
    if include_over_budget:
        md_header = (
            "| Architecture | Trials | Completed | Pruned | Over budget "
            "| Mean trained trial (min) | Total GPU-hours |"
        )
        md_sep = "|---|---|---|---|---|---|---|"
    else:
        md_header = (
            "| Architecture | Trials | Completed | Pruned "
            "| Mean trained trial (min) | Total GPU-hours |"
        )
        md_sep = "|---|---|---|---|---|---|"

    md = [md_header, md_sep]
    for r in rows:
        cells = [
            ARCH_DISPLAY.get(r["arch"], r["arch"]),
            str(r["display_trials"]),
            str(r["n_completed"]),
            str(r["n_pruned_median"]),
        ]
        if include_over_budget:
            cells.append(str(r["n_pruned_budget"]))
        cells.append(_minutes(r["mean_trained_seconds"]))
        cells.append(_hours(r["total_seconds"]))
        md.append("| " + " | ".join(cells) + " |")

    total_md = [
        "**Total**",
        f"**{grand_display_trials}**",
        f"**{grand_completed}**",
        f"**{grand_pruned_med}**",
    ]
    if include_over_budget:
        total_md.append(f"**{grand_pruned_bud}**")
    total_md.append("---")
    total_md.append(f"**{_hours(grand_total_s)}**")
    md.append("| " + " | ".join(total_md) + " |")
    (output_dir / "gpu_time.md").write_text("\n".join(md) + "\n")
    print(f"  wrote {output_dir / 'gpu_time.md'}")


# Top-level runner
def run_hpo_report(
    studies_dir: Path,
    output_dir: Path,
    archs: List[str],
    pdes: List[str],
    include_importance: bool = False,
    include_over_budget: bool = False,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading studies from {studies_dir}")
    studies: Dict[Tuple[str, str], Any] = {}
    for arch in archs:
        for pde in pdes:
            s = load_study(studies_dir, arch, pde)
            if s is not None:
                studies[(arch, pde)] = s

    if not studies:
        print("No studies loaded; nothing to do")
        return

    print(f"\nLoaded {len(studies)} studies. Generating artifacts...")

    print("\nTier 1: search space tables")
    emit_search_space_tables(studies, output_dir, archs, pdes)

    print("\nTier 1: best hyperparameters tables")
    emit_best_hparams_tables(studies, output_dir, archs, pdes)

    print("\nTier 2: convergence plots")
    emit_convergence_plots(studies, output_dir, archs, pdes)

    print("\nGPU time table")
    emit_gpu_time_table(
        studies,
        output_dir,
        archs,
        pdes,
        include_over_budget=include_over_budget,
    )

    if include_importance:
        print("\nTier 3: hyperparameter importance")
        emit_importance_plots(studies, output_dir, archs, pdes)

    print(f"\nDone. Output in: {output_dir}")


# CLI
def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--studies_dir",
        type=str,
        default=".",
        help="Directory containing optuna_{arch}_{pde}.db files",
    )
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--archs", type=str, nargs="+", default=DEFAULT_ARCHS)
    p.add_argument("--pdes", type=str, nargs="+", default=DEFAULT_PDES)
    p.add_argument(
        "--include_importance",
        action="store_true",
        help="Also compute fANOVA importance plots (Tier 3). "
        "Slow for studies with many trials and hparams.",
    )
    p.add_argument(
        "--include_over_budget",
        action="store_true",
        help="Show parameter-budget rejections as a separate column in the "
        "GPU time table. Default omits them, since they are an artifact of "
        "how the parameter cap is enforced rather than meaningful "
        "search-space exploration.",
    )
    args = p.parse_args()

    run_hpo_report(
        studies_dir=Path(args.studies_dir),
        output_dir=Path(args.output_dir),
        archs=args.archs,
        pdes=args.pdes,
        include_importance=args.include_importance,
        include_over_budget=args.include_over_budget,
    )


if __name__ == "__main__":
    main()
