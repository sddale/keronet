"""Result tables, paired statistics and figures from evaluate.py output directories.

Writes main_table.{tex,md}, stats.json, training_time.{tex,md}, summary.{md,tex} and
per-PDE boxplot/stripplot/loss_curves PDFs to --output_dir. Seed std uses ddof=1; paired
tests and bootstrap CIs use per-seed mean rel-L2, with every architecture as baseline.
Usage: python plotting/report.py --results DIR [DIR ...] --output_dir figures --baseline keronet
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from paper_style import (
    apply_paper_style,
    fig_size,
    ARCH_STYLE,
    ARCH_DISPLAY,
    PDE_DISPLAY,
    save_fig,
)


# Apply publication style globally for any figures emitted by this module.
apply_paper_style()


# Loading artifacts
def load_results_dir(results_dir: Path) -> Dict[str, Any]:
    summary_path = results_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"No summary.json in {results_dir}; was evaluate.py ever run?"
        )
    with open(summary_path) as f:
        summary = json.load(f)

    pde = summary["pde"]
    cells: Dict[str, Dict[str, Any]] = {}
    for cell_id, meta in summary["by_cell"].items():
        cell_dir = results_dir / cell_id
        errs_path = cell_dir / "per_ic_errors.npy"
        loss_path = cell_dir / "loss_log.npz"
        cells[cell_id] = {
            "meta": meta,
            "errs": np.load(errs_path) if errs_path.exists() else None,
            "loss_log": dict(np.load(loss_path)) if loss_path.exists() else None,
        }

    by_arch: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for cell_id, c in cells.items():
        if c["meta"]:
            by_arch[c["meta"]["arch"]].append(c["meta"])
    for arch in by_arch:
        by_arch[arch].sort(key=lambda m: m["seed"])

    return {
        "pde": pde,
        "summary": summary,
        "cells": cells,
        "by_arch": dict(by_arch),
        "dir": results_dir,
    }


def load_all_results(results_dirs: List[Path]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for d in results_dirs:
        loaded = load_results_dir(d)
        pde = loaded["pde"]
        if pde not in out:
            out[pde] = loaded
            continue
        existing = out[pde]
        existing["cells"].update(loaded["cells"])
        by_arch: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for cell in existing["cells"].values():
            if cell["meta"]:
                by_arch[cell["meta"]["arch"]].append(cell["meta"])
        for arch in by_arch:
            by_arch[arch].sort(key=lambda m: m["seed"])
        existing["by_arch"] = dict(by_arch)
    return out


# Aggregation helpers
def per_seed_means(loaded: Dict[str, Any], arch: str) -> np.ndarray:
    """Return each seed's mean rel-L2 for one architecture, ordered by seed."""
    metas = loaded["by_arch"].get(arch, [])
    return np.array([m["rel_l2_mean"] for m in metas])


def all_per_ic_errs(loaded: Dict[str, Any], arch: str) -> np.ndarray:
    """Return the per-IC errors of all of one architecture's seeds, concatenated."""
    chunks = []
    for cell_id, c in loaded["cells"].items():
        if c["meta"] and c["meta"]["arch"] == arch and c["errs"] is not None:
            chunks.append(c["errs"])
    if not chunks:
        return np.array([])
    return np.concatenate(chunks)


# Statistical tests
def paired_wilcoxon(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    """Two-sided paired Wilcoxon test; returns p_value and the median and mean of a - b."""
    from scipy.stats import wilcoxon

    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: a {a.shape} vs b {b.shape}")
    if a.size < 1:
        return {"p_value": float("nan"), "median_diff": float("nan"), "n_pairs": 0}
    if a.size == 1:
        return {
            "p_value": float("nan"),
            "median_diff": float(a[0] - b[0]),
            "mean_diff": float(a[0] - b[0]),
            "n_pairs": 1,
        }
    try:
        stat = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
        p = float(stat.pvalue)
    except Exception:
        p = float("nan")
    return {
        "p_value": p,
        "median_diff": float(np.median(a - b)),
        "mean_diff": float(np.mean(a - b)),
        "n_pairs": int(a.size),
    }


def bootstrap_diff_ci(
    a: np.ndarray,
    b: np.ndarray,
    *,
    n_resamples: int = 10_000,
    ci: float = 0.95,
    seed: int = 0,
) -> Dict[str, float]:
    """Percentile bootstrap CI for the mean of the paired differences a - b."""
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape or a.size == 0:
        return {
            "diff_mean": float("nan"),
            "ci_lo": float("nan"),
            "ci_hi": float("nan"),
            "n_pairs": int(a.size),
        }
    rng = np.random.default_rng(seed)
    n = a.size
    diffs = a - b
    boot = rng.choice(diffs, size=(n_resamples, n), replace=True).mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(boot, [alpha, 1.0 - alpha])
    return {
        "diff_mean": float(diffs.mean()),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "n_pairs": int(n),
        "ci_level": float(ci),
    }


def compute_pairwise_stats(
    all_results: Dict[str, Dict[str, Any]],
    baseline: str,
) -> Dict[str, Any]:
    """Per PDE, compare each other architecture with baseline on per-seed mean rel-L2.

    Returns {"baseline", "by_pde": {pde: {arch: stats}}}; diffs are arch minus baseline.
    """
    out: Dict[str, Any] = {"baseline": baseline, "by_pde": {}}
    for pde, loaded in all_results.items():
        per_pde: Dict[str, Any] = {}
        if baseline not in loaded["by_arch"]:
            per_pde["error"] = (
                f"baseline {baseline!r} has no completed cells for this PDE"
            )
            out["by_pde"][pde] = per_pde
            continue

        seeds_baseline = sorted(m["seed"] for m in loaded["by_arch"][baseline])

        for arch in loaded["by_arch"]:
            if arch == baseline:
                continue
            seeds_other = sorted(m["seed"] for m in loaded["by_arch"][arch])
            common = sorted(set(seeds_baseline) & set(seeds_other))
            if not common:
                per_pde[arch] = {"error": "no common seeds"}
                continue

            def _by_seed(metas, seeds):
                return np.array(
                    [
                        next(m["rel_l2_mean"] for m in metas if m["seed"] == s)
                        for s in seeds
                    ]
                )

            a = _by_seed(loaded["by_arch"][baseline], common)  # baseline
            b = _by_seed(loaded["by_arch"][arch], common)  # other

            # Direction-of-effect: count seeds where baseline beats other.
            n_baseline_wins = int(np.sum(a < b))
            n_ties = int(np.sum(a == b))
            n_other_wins = int(np.sum(a > b))

            wcx = paired_wilcoxon(b, a)
            boot = bootstrap_diff_ci(b, a)
            per_pde[arch] = {
                "n_seeds_compared": len(common),
                "mean_diff_arch_minus_baseline": wcx["mean_diff"],
                "median_diff_arch_minus_baseline": wcx["median_diff"],
                "wilcoxon_p": wcx["p_value"],
                "bootstrap_diff_mean": boot["diff_mean"],
                "bootstrap_ci_lo": boot["ci_lo"],
                "bootstrap_ci_hi": boot["ci_hi"],
                "bootstrap_ci_level": boot["ci_level"],
                "n_seeds_favoring_baseline": n_baseline_wins,
                "n_seeds_favoring_other": n_other_wins,
                "n_seeds_tied": n_ties,
            }
        out["by_pde"][pde] = per_pde
    return out


def compute_all_pairwise_stats(
    all_results: Dict[str, Dict[str, Any]],
    arch_order: List[str],
    primary_baseline: str,
) -> List[Dict[str, Any]]:
    """Run compute_pairwise_stats with each architecture as baseline, primary first."""
    ordered = [primary_baseline] + [a for a in arch_order if a != primary_baseline]
    return [compute_pairwise_stats(all_results, b) for b in ordered]


# Tables
def _format_mean_std(mean: float, std: float) -> str:
    """Format a rel-L2 mean and std in percent for LaTeX (escaped percent sign)."""
    return f"{100 * mean:.2f} ± {100 * std:.2f} \\%"


def _format_mean_std_md(mean: float, std: float) -> str:
    return f"{100 * mean:.2f} ± {100 * std:.2f}%"


def main_table_rows(
    all_results: Dict[str, Dict[str, Any]], arch_order: List[str], pde_order: List[str]
):
    rows = []
    for arch in arch_order:
        row = {"arch": arch}
        for pde in pde_order:
            loaded = all_results.get(pde)
            if not loaded or arch not in loaded["by_arch"]:
                row[pde] = None
                continue
            ms = per_seed_means(loaded, arch)
            param_count = loaded["by_arch"][arch][0]["param_count"]
            train_time = float(
                np.mean([m["train_time_s"] for m in loaded["by_arch"][arch]])
            )
            row[pde] = {
                "mean_of_seed_means": float(ms.mean()),
                "std_of_seed_means": float(ms.std(ddof=1)) if ms.size > 1 else 0.0,
                "n_seeds": int(ms.size),
                "param_count": int(param_count),
                "train_time_mean_s": float(train_time),
            }
        rows.append(row)
    return rows


def emit_main_table_latex(rows, pde_order, output_dir: Path) -> Path:
    best_idx = {}
    for pde in pde_order:
        vals = [
            (i, r[pde]["mean_of_seed_means"])
            for i, r in enumerate(rows)
            if r[pde] is not None
        ]
        if vals:
            best_idx[pde] = min(vals, key=lambda x: x[1])[0]

    lines = []
    lines.append(r"\begin{tabular}{l" + "c" * len(pde_order) + "}")
    lines.append(r"\toprule")
    header = (
        "Method & "
        + " & ".join(PDE_DISPLAY.get(p, p.replace("_", r"\_")) for p in pde_order)
        + r" \\"
    )
    lines.append(header)
    lines.append(r"\midrule")
    for i, r in enumerate(rows):
        cells = []
        for pde in pde_order:
            entry = r[pde]
            if entry is None:
                cells.append("---")
            else:
                txt = _format_mean_std(
                    entry["mean_of_seed_means"],
                    entry["std_of_seed_means"],
                )
                if i == best_idx.get(pde):
                    txt = r"\textbf{" + txt + "}"
                cells.append(txt)
        arch_disp = ARCH_DISPLAY.get(r["arch"], r["arch"])
        lines.append(arch_disp + " & " + " & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    path = output_dir / "main_table.tex"
    path.write_text("\n".join(lines) + "\n")
    return path


def emit_main_table_md(rows, pde_order, output_dir: Path) -> Path:
    headers = ["Method"] + [PDE_DISPLAY.get(p, p) for p in pde_order]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    best_idx = {}
    for pde in pde_order:
        vals = [
            (i, r[pde]["mean_of_seed_means"])
            for i, r in enumerate(rows)
            if r[pde] is not None
        ]
        if vals:
            best_idx[pde] = min(vals, key=lambda x: x[1])[0]

    for i, r in enumerate(rows):
        cells = [ARCH_DISPLAY.get(r["arch"], r["arch"])]
        for pde in pde_order:
            entry = r[pde]
            if entry is None:
                cells.append("---")
            else:
                txt = _format_mean_std_md(
                    entry["mean_of_seed_means"],
                    entry["std_of_seed_means"],
                )
                if i == best_idx.get(pde):
                    txt = "**" + txt + "**"
                cells.append(txt)
        lines.append("| " + " | ".join(cells) + " |")

    path = output_dir / "main_table.md"
    path.write_text("\n".join(lines) + "\n")
    return path


# Pairwise table rendering
def _pairwise_md_table(stats: Dict[str, Any], pde_order: List[str]) -> List[str]:
    """Render one pairwise table as a list of Markdown lines."""
    base = stats["baseline"]
    base_disp = ARCH_DISPLAY.get(base, base)
    lines = []
    lines.append(f"### Baseline: {base_disp}")
    lines.append("")
    lines.append(
        f"Positive `mean_diff` means the other architecture has higher error "
        f"than {base_disp} (i.e., {base_disp} is better). "
        f"`Seeds favoring base.` is the count of paired seeds where "
        f"{base_disp} achieved lower rel-L2 than the comparison."
    )
    lines.append("")
    lines.append(
        "| PDE | Other arch | n | mean_diff (other - baseline) | "
        "Seeds favoring base. | Wilcoxon p | bootstrap 95% CI |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for pde in pde_order:
        per_pde = stats["by_pde"].get(pde, {})
        if "error" in per_pde:
            lines.append(
                f"| {PDE_DISPLAY.get(pde, pde)} | --- | --- | --- | --- | --- | "
                f"(skipped: {per_pde['error']}) |"
            )
            continue
        for arch, s in per_pde.items():
            if "error" in s:
                continue
            arch_disp = ARCH_DISPLAY.get(arch, arch)
            p_str = (
                f"{s['wilcoxon_p']:.3f}"
                if s["wilcoxon_p"] == s["wilcoxon_p"]
                else "n/a"
            )
            doe = f"{s['n_seeds_favoring_baseline']}/{s['n_seeds_compared']}"
            lines.append(
                f"| {PDE_DISPLAY.get(pde, pde)} | {arch_disp} | "
                f"{s['n_seeds_compared']} | "
                f"{100 * s['mean_diff_arch_minus_baseline']:+.2f}pp | "
                f"{doe} | "
                f"{p_str} | "
                f"[{100 * s['bootstrap_ci_lo']:+.2f}, "
                f"{100 * s['bootstrap_ci_hi']:+.2f}]pp |"
            )
    return lines


def _pairwise_tex_table(stats: Dict[str, Any], pde_order: List[str]) -> List[str]:
    """Render one pairwise table as LaTeX lines (tabular in a captioned table)."""
    base = stats["baseline"]
    base_disp = ARCH_DISPLAY.get(base, base)
    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{llrrll}")
    lines.append(r"\toprule")
    lines.append(
        r"PDE & vs.\ baseline & $n$ & mean diff.\ (pp) "
        r"& Seeds fav.\ base. & 95\% CI (pp) \\"
    )
    lines.append(r"\midrule")
    for pde in pde_order:
        per_pde = stats["by_pde"].get(pde, {})
        if "error" in per_pde:
            continue
        pde_disp = PDE_DISPLAY.get(pde, pde.replace("_", r"\_"))
        for arch, s in per_pde.items():
            if "error" in s:
                continue
            arch_disp = ARCH_DISPLAY.get(arch, arch)
            ci_str = (
                f"$[{100 * s['bootstrap_ci_lo']:+.2f},\\,"
                f"{100 * s['bootstrap_ci_hi']:+.2f}]$"
            )
            doe = f"{s['n_seeds_favoring_baseline']}/{s['n_seeds_compared']}"
            lines.append(
                f"{pde_disp} & {arch_disp} & {s['n_seeds_compared']} "
                f"& ${100 * s['mean_diff_arch_minus_baseline']:+.2f}$ "
                f"& {doe} & {ci_str} \\\\"
            )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    safe_base = base.replace("-", "_")
    lines.append(
        rf"\caption{{Pairwise comparisons against {base_disp}. "
        rf"Positive mean difference means the comparison architecture has "
        rf"higher error than {base_disp}. "
        rf"\textit{{Seeds fav.\ base.}} counts paired seeds on which "
        rf"{base_disp} achieved lower rel-$L^2$ than the comparison "
        rf"architecture. CIs are 10{{,}}000-replicate paired bootstrap on "
        rf"per-seed means.}}"
    )
    lines.append(rf"\label{{tab:pairwise_{safe_base}}}")
    lines.append(r"\end{table}")
    return lines


# Per-seed tables (one per PDE)
def _per_seed_rows_for_pde(
    loaded: Dict[str, Any], arch_order: List[str]
) -> List[Tuple[str, int, float, float, float]]:
    """Return (arch, seed, rel_l2_mean, rel_l2_std, train_time_s) rows for one PDE."""
    rows = []
    for arch in arch_order:
        for meta in loaded["by_arch"].get(arch, []):
            rows.append(
                (
                    arch,
                    int(meta["seed"]),
                    float(meta["rel_l2_mean"]),
                    float(meta.get("rel_l2_std", float("nan"))),
                    float(meta.get("train_time_s", float("nan"))),
                )
            )
    return rows


def _per_seed_md_table(loaded: Dict[str, Any], arch_order: List[str]) -> List[str]:
    """Render per-seed cells for one PDE as Markdown lines."""
    pde = loaded["pde"]
    pde_disp = PDE_DISPLAY.get(pde, pde)
    rows = _per_seed_rows_for_pde(loaded, arch_order)

    lines = [f"### {pde_disp}", ""]
    if not rows:
        lines.append("(no completed cells)")
        lines.append("")
        return lines

    lines.append("| Method | Seed | rel-L2 mean | rel-L2 std (per-IC) | Train time |")
    lines.append("|---|---|---|---|---|")
    for arch, seed, mean, std, t in rows:
        arch_disp = ARCH_DISPLAY.get(arch, arch)
        lines.append(
            f"| {arch_disp} | {seed} | "
            f"{100 * mean:.2f}% | "
            f"{100 * std:.2f}% | "
            f"{t:.1f}s |"
        )
    return lines


def _per_seed_tex_table(loaded: Dict[str, Any], arch_order: List[str]) -> List[str]:
    """Render per-seed cells for one PDE as a LaTeX table."""
    pde = loaded["pde"]
    pde_disp = PDE_DISPLAY.get(pde, pde.replace("_", r"\_"))
    safe_pde = pde.replace("-", "_")
    rows = _per_seed_rows_for_pde(loaded, arch_order)

    lines = [r"\begin{table}[h]", r"\centering", r"\small"]
    if not rows:
        lines.append(rf"\caption*{{No completed cells for {pde_disp}.}}")
        lines.append(r"\end{table}")
        return lines

    lines.append(r"\begin{tabular}{lcrrr}")
    lines.append(r"\toprule")
    lines.append(
        r"Method & Seed & rel-$L^2$ mean & rel-$L^2$ std (per-IC) "
        r"& Train time \\"
    )
    lines.append(r"\midrule")

    prev_arch = None
    for arch, seed, mean, std, t in rows:
        if prev_arch is not None and arch != prev_arch:
            lines.append(r"\midrule")
        prev_arch = arch
        arch_disp = ARCH_DISPLAY.get(arch, arch)
        lines.append(
            f"{arch_disp} & {seed} "
            f"& ${100 * mean:.2f}\\%$ "
            f"& ${100 * std:.2f}\\%$ "
            f"& ${t:.1f}$s \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(
        rf"\caption{{Per-seed results for {pde_disp}. \emph{{rel-$L^2$ std}}"
        rf" is the standard deviation of per-IC rel-$L^2$ within a single "
        rf"seed (across the $100$ test input functions); it is not a "
        rf"cross-seed statistic.}}"
    )
    lines.append(rf"\label{{tab:per_seed_{safe_pde}}}")
    lines.append(r"\end{table}")
    return lines


# Training-time table (one row per architecture, total across all PDEs)
def _training_time_rows(
    all_results: Dict[str, Dict[str, Any]], arch_order: List[str]
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """Sum each architecture's train_time_s over all PDEs and seeds; return (rows, totals).

    rows: {arch, n_runs, total_seconds, mean_run_seconds}; totals: {n_runs, total_seconds}.
    """
    rows: List[Dict[str, Any]] = []
    grand_runs = 0
    grand_seconds = 0.0
    for arch in arch_order:
        n_runs = 0
        total_seconds = 0.0
        any_data = False
        for loaded in all_results.values():
            for meta in loaded["by_arch"].get(arch, []):
                t = meta.get("train_time_s")
                if t is None:
                    continue
                total_seconds += float(t)
                n_runs += 1
                any_data = True
        if not any_data:
            continue
        rows.append(
            {
                "arch": arch,
                "n_runs": n_runs,
                "total_seconds": total_seconds,
                "mean_run_seconds": total_seconds / max(n_runs, 1),
            }
        )
        grand_runs += n_runs
        grand_seconds += total_seconds
    return rows, {"n_runs": grand_runs, "total_seconds": grand_seconds}


def _training_time_md(rows, totals) -> List[str]:
    """Markdown rendering of the training-time table."""
    if not rows:
        return ["(no training-time data available)"]

    def _hours(s: float) -> str:
        return f"{s / 3600.0:.2f}"

    def _trial_min(s: float) -> str:
        return f"{s / 60.0:.1f}"

    out = [
        "| Architecture | Runs | Mean run (min) | Total GPU-hours |",
        "|---|---|---|---|",
    ]
    for r in rows:
        out.append(
            f"| {ARCH_DISPLAY.get(r['arch'], r['arch'])} "
            f"| {r['n_runs']} "
            f"| {_trial_min(r['mean_run_seconds'])} "
            f"| {_hours(r['total_seconds'])} |"
        )
    out.append(
        f"| **Total** | **{totals['n_runs']}** | --- "
        f"| **{_hours(totals['total_seconds'])}** |"
    )
    return out


def _training_time_tex(rows, totals) -> List[str]:
    """LaTeX rendering of the training-time table, in a captioned table environment."""
    out = [r"\begin{table}[h]", r"\centering", r"\small"]
    if not rows:
        out.append(r"\caption*{No training-time data available.}")
        out.append(r"\end{table}")
        return out

    def _hours(s: float) -> str:
        return f"{s / 3600.0:.2f}"

    def _trial_min(s: float) -> str:
        return f"{s / 60.0:.1f}"

    out.append(r"\begin{tabular}{lrrr}")
    out.append(r"\toprule")
    out.append(r"Architecture & Runs & Mean run (min) & Total GPU-hours \\")
    out.append(r"\midrule")
    for r in rows:
        out.append(
            f"{ARCH_DISPLAY.get(r['arch'], r['arch'])} "
            f"& {r['n_runs']} "
            f"& {_trial_min(r['mean_run_seconds'])} "
            f"& {_hours(r['total_seconds'])} \\\\"
        )
    out.append(r"\midrule")
    out.append(
        rf"\textbf{{Total}} & \textbf{{{totals['n_runs']}}} & --- "
        rf"& \textbf{{{_hours(totals['total_seconds'])}}} \\"
    )
    out.append(r"\bottomrule")
    out.append(r"\end{tabular}")
    out.append(
        r"\caption{Final-evaluation training compute summed across all "
        r"PDEs and seeds. \emph{Runs} counts every "
        r"(PDE, seed) cell at the architecture's HPO-selected configuration. "
        r"HPO compute is reported separately in Appendix~\ref{app:hpo}.}"
    )
    out.append(r"\label{tab:training_time}")
    out.append(r"\end{table}")
    return out


def emit_training_time_table(
    all_results: Dict[str, Dict[str, Any]],
    arch_order: List[str],
    output_dir: Path,
) -> Tuple[Path, Path] | None:
    """Write training_time.tex/.md; return their paths, or None if there is no data."""
    rows, totals = _training_time_rows(all_results, arch_order)
    if not rows:
        return None
    tex_lines = _training_time_tex(rows, totals)
    md_lines = _training_time_md(rows, totals)
    tex_path = output_dir / "training_time.tex"
    md_path = output_dir / "training_time.md"
    tex_path.write_text("\n".join(tex_lines) + "\n")
    md_path.write_text("\n".join(md_lines) + "\n")
    return tex_path, md_path


# Figures
def emit_boxplot(
    loaded: Dict[str, Any], arch_order: List[str], output_dir: Path
) -> Path | None:
    import matplotlib.pyplot as plt

    pde = loaded["pde"]
    archs = [a for a in arch_order if a in loaded["by_arch"]]
    if not archs:
        return None

    data = [all_per_ic_errs(loaded, a) for a in archs]
    if all(len(d) == 0 for d in data):
        return None

    fig, ax = plt.subplots(figsize=fig_size("col", aspect=0.5))
    box_colors = [ARCH_STYLE[a]["color"] if a in ARCH_STYLE else "0.5" for a in archs]
    bp = ax.boxplot(
        data,
        tick_labels=[ARCH_DISPLAY.get(a, a) for a in archs],
        showmeans=True,
        meanline=False,
        patch_artist=True,
        widths=0.55,
        flierprops={
            "marker": ".",
            "markersize": 2.5,
            "markerfacecolor": "0.4",
            "markeredgecolor": "none",
            "alpha": 0.6,
        },
        whiskerprops={"linewidth": 0.7, "color": "0.25"},
        capprops={"linewidth": 0.7, "color": "0.25"},
        medianprops={"linewidth": 1.0, "color": "black"},
        meanprops={
            "marker": "D",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
            "markersize": 3.5,
        },
    )
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)
        patch.set_edgecolor(color)
        patch.set_linewidth(0.7)

    ax.set_yscale("log")
    ax.set_ylabel(r"Per-IC relative $L^2$ error")
    ax.set_title(PDE_DISPLAY.get(pde, pde))
    ax.grid(True, which="both", axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return save_fig(fig, output_dir / f"boxplot_{pde}.pdf")


def emit_stripplot(
    loaded: Dict[str, Any], arch_order: List[str], output_dir: Path
) -> Path | None:
    """Write stripplot_<pde>.pdf: per-IC errors of all seeds, jittered, over a box plot."""
    import matplotlib.pyplot as plt

    pde = loaded["pde"]
    archs = [a for a in arch_order if a in loaded["by_arch"]]
    if not archs:
        return None

    # Per-IC errors, organized one list per arch (preserving seed grouping
    # so we can color points by seed if useful).
    per_arch_errs: List[np.ndarray] = []
    per_arch_seed_idx: List[np.ndarray] = []
    for arch in archs:
        chunks = []
        seed_idx_chunks = []
        # Iterate seeds in the order they appear in by_arch (already sorted by seed)
        for s_i, meta in enumerate(loaded["by_arch"][arch]):
            cell_id = f"{arch}_seed{meta['seed']}"
            errs = loaded["cells"].get(cell_id, {}).get("errs")
            if errs is None:
                continue
            chunks.append(np.asarray(errs))
            seed_idx_chunks.append(np.full(len(errs), s_i))
        if not chunks:
            per_arch_errs.append(np.array([]))
            per_arch_seed_idx.append(np.array([]))
        else:
            per_arch_errs.append(np.concatenate(chunks))
            per_arch_seed_idx.append(np.concatenate(seed_idx_chunks))

    if all(len(e) == 0 for e in per_arch_errs):
        return None

    fig, ax = plt.subplots(figsize=fig_size("col", aspect=0.5))

    # Reproducible jitter per cell (independent of seed-ordering)
    rng = np.random.default_rng(0)
    jitter_width = 0.28

    for i, (arch, errs, seed_idx) in enumerate(
        zip(archs, per_arch_errs, per_arch_seed_idx)
    ):
        x_center = i + 1
        if len(errs) == 0:
            continue

        color = ARCH_STYLE.get(arch, {"color": "0.5"})["color"]

        #  Box outline (Q1/median/Q3 + whiskers) sitting behind points
        q1, med, q3 = np.quantile(errs, [0.25, 0.5, 0.75])
        iqr = q3 - q1
        # 1.5xIQR whiskers, clipped to actual data extent
        lo_fence = q1 - 1.5 * iqr
        hi_fence = q3 + 1.5 * iqr
        within = errs[(errs >= lo_fence) & (errs <= hi_fence)]
        whisker_lo = within.min() if within.size else q1
        whisker_hi = within.max() if within.size else q3

        box_w = 0.55
        # Box rectangle
        ax.plot(
            [x_center - box_w / 2, x_center + box_w / 2],
            [q1, q1],
            color="0.25",
            linewidth=0.8,
            zorder=1,
        )
        ax.plot(
            [x_center - box_w / 2, x_center + box_w / 2],
            [q3, q3],
            color="0.25",
            linewidth=0.8,
            zorder=1,
        )
        ax.plot(
            [x_center - box_w / 2, x_center - box_w / 2],
            [q1, q3],
            color="0.25",
            linewidth=0.8,
            zorder=1,
        )
        ax.plot(
            [x_center + box_w / 2, x_center + box_w / 2],
            [q1, q3],
            color="0.25",
            linewidth=0.8,
            zorder=1,
        )
        # Median line
        ax.plot(
            [x_center - box_w / 2, x_center + box_w / 2],
            [med, med],
            color="black",
            linewidth=1.2,
            zorder=2,
        )
        # Whiskers
        ax.plot(
            [x_center, x_center],
            [whisker_lo, q1],
            color="0.25",
            linewidth=0.7,
            zorder=1,
        )
        ax.plot(
            [x_center, x_center],
            [q3, whisker_hi],
            color="0.25",
            linewidth=0.7,
            zorder=1,
        )
        # Whisker caps
        cap_w = box_w * 0.35
        ax.plot(
            [x_center - cap_w / 2, x_center + cap_w / 2],
            [whisker_lo, whisker_lo],
            color="0.25",
            linewidth=0.7,
            zorder=1,
        )
        ax.plot(
            [x_center - cap_w / 2, x_center + cap_w / 2],
            [whisker_hi, whisker_hi],
            color="0.25",
            linewidth=0.7,
            zorder=1,
        )

        #  Jittered points (foreground)
        x_jit = x_center + rng.uniform(-jitter_width, jitter_width, len(errs))
        ax.scatter(
            x_jit,
            errs,
            s=4.0,
            color=color,
            alpha=0.40,
            edgecolors="none",
            zorder=3,
            rasterized=True,
        )

    ax.set_xticks(range(1, len(archs) + 1))
    ax.set_xticklabels([ARCH_DISPLAY.get(a, a) for a in archs])
    ax.set_yscale("log")
    ax.set_ylabel(r"Per-IC relative $L^2$ error")
    ax.set_title(PDE_DISPLAY.get(pde, pde))
    ax.grid(True, which="both", axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    # Tighten x-axis a bit so the boxes aren't pushed against the spines
    ax.set_xlim(0.45, len(archs) + 0.55)
    fig.tight_layout()
    return save_fig(fig, output_dir / f"stripplot_{pde}.pdf")


def emit_loss_curves(
    loaded: Dict[str, Any], arch_order: List[str], output_dir: Path
) -> Path | None:
    import matplotlib.pyplot as plt

    pde = loaded["pde"]
    archs = [a for a in arch_order if a in loaded["by_arch"]]
    if not archs:
        return None

    fig, ax = plt.subplots(figsize=fig_size("col", aspect=0.6))
    for arch in archs:
        curves = []
        for cell_id, c in loaded["cells"].items():
            if c["meta"] and c["meta"]["arch"] == arch and c["loss_log"] is not None:
                curves.append(np.asarray(c["loss_log"]["loss_total"]))
        if not curves:
            continue
        min_len = min(len(c) for c in curves)
        if min_len == 0:
            continue
        stacked = np.stack([c[:min_len] for c in curves])
        x = np.arange(min_len) * 100
        # The loss spans orders of magnitude, so the seeds are combined on the log scale the
        # axis uses: geometric mean, with a band of +-1 standard deviation of log10(loss).
        # (An arithmetic mean +- std band drops to zero wherever one seed spikes.)
        logl = np.log10(stacked)
        mean = 10 ** logl.mean(axis=0)
        s = ARCH_STYLE.get(arch, {"color": None, "ls": "-"})
        if stacked.shape[0] > 1:
            sd = logl.std(axis=0, ddof=1)
            ax.fill_between(
                x,
                mean / 10**sd,
                mean * 10**sd,
                color=s["color"],
                alpha=0.18,
                linewidth=0,
            )
        ax.plot(
            x,
            mean,
            label=ARCH_DISPLAY.get(arch, arch),
            color=s["color"],
            linestyle=s["ls"],
            linewidth=1.2,
        )
    ax.set_yscale("log")
    ax.set_xlabel("Training iteration")
    ax.set_ylabel(
        r"Total loss $\mathcal{L}_{\mathrm{bcs}} + \mathcal{L}_{\mathrm{res}}$"
    )
    ax.set_title(PDE_DISPLAY.get(pde, pde))
    ax.legend(loc="upper right", ncol=2)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return save_fig(fig, output_dir / f"loss_curves_{pde}.pdf")


# Summary report (markdown + LaTeX)
def emit_summary_md(
    all_results: Dict[str, Dict[str, Any]],
    arch_order: List[str],
    pde_order: List[str],
    rows,
    all_stats: List[Dict[str, Any]],
    output_dir: Path,
) -> Path:
    """Write summary.md: main, per-PDE, pairwise and per-seed tables, compute, notes."""
    lines = []
    lines.append("# Experiment summary")
    lines.append("")
    lines.append("## Main results table (rel-L2, mean ± std over seeds)")
    lines.append("")
    md_path = output_dir / "main_table.md"
    if md_path.exists():
        lines.append(md_path.read_text().rstrip())
    lines.append("")

    lines.append("## Per-PDE detail")
    for pde in pde_order:
        loaded = all_results.get(pde)
        if not loaded:
            continue
        lines.append(f"\n### {PDE_DISPLAY.get(pde, pde)}")
        lines.append("")
        lines.append(
            "| Method | Seeds | rel-L2 mean | rel-L2 std (mean per seed) | "
            "Params | Train time |"
        )
        lines.append("|---|---|---|---|---|---|")
        for r in rows:
            entry = r[pde]
            if entry is None:
                continue
            arch_disp = ARCH_DISPLAY.get(r["arch"], r["arch"])
            lines.append(
                f"| {arch_disp} | {entry['n_seeds']} | "
                f"{100 * entry['mean_of_seed_means']:.2f}% | "
                f"{100 * entry['std_of_seed_means']:.2f}% | "
                f"{entry['param_count']:,} | "
                f"{entry['train_time_mean_s']:.1f}s |"
            )

    lines.append("\n## Pairwise comparisons against each architecture as baseline")
    lines.append("")
    for stats in all_stats:
        lines.extend(_pairwise_md_table(stats, pde_order))
        lines.append("")

    lines.append("## Per-seed results (one table per PDE)")
    lines.append("")
    for pde in pde_order:
        loaded = all_results.get(pde)
        if not loaded:
            continue
        lines.extend(_per_seed_md_table(loaded, arch_order))
        lines.append("")

    lines.append("## Final-evaluation training compute (totals across all PDEs)")
    lines.append("")
    tt_rows, tt_totals = _training_time_rows(all_results, arch_order)
    lines.extend(_training_time_md(tt_rows, tt_totals))
    lines.append("")

    lines.append("## Notes")
    lines.append("- Differences are reported in *percentage points* of rel-L2.")
    lines.append(
        "- With small numbers of seeds (typically 5), the Wilcoxon p-value "
        "has a hard floor around 0.0625, so **absence of significance is "
        "not absence of effect**. Bootstrap CIs are the more informative "
        "metric here."
    )
    lines.append(
        "- *Seeds favoring base.* is the direction-of-effect statistic: "
        "the count of paired seeds on which the baseline achieved lower "
        "rel-L2 than the comparison architecture."
    )

    path = output_dir / "summary.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def emit_summary_tex(
    all_results: Dict[str, Dict[str, Any]],
    arch_order: List[str],
    pde_order: List[str],
    rows,
    all_stats: List[Dict[str, Any]],
    output_dir: Path,
) -> Path:
    """Write summary.tex, a LaTeX fragment (no preamble) with the tables of summary.md."""
    lines = []
    lines.append(r"% Auto-generated by report.py --- do not edit by hand.")
    lines.append(r"\section*{Experiment summary}")
    lines.append("")

    lines.append(r"\subsection*{Main results table}")
    lines.append("")
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\small")
    main_tex = output_dir / "main_table.tex"
    if main_tex.exists():
        lines.append(main_tex.read_text().rstrip())
    lines.append(
        r"\caption{Mean per-sample relative $L^2$ error across all PDEs, "
        r"aggregated across training seeds and test input functions per seed. "
        r"The $\pm$ value reports the (sample) standard deviation of "
        r"per-seed mean errors. Best per PDE in bold.}"
    )
    lines.append(r"\label{tab:summary_main}")
    lines.append(r"\end{table}")
    lines.append("")

    lines.append(r"\subsection*{Per-PDE detail}")
    lines.append("")
    for pde in pde_order:
        loaded = all_results.get(pde)
        if not loaded:
            continue
        pde_disp = PDE_DISPLAY.get(pde, pde.replace("_", r"\_"))
        safe_pde = pde.replace("-", "_")
        lines.append(r"\begin{table}[h]")
        lines.append(r"\centering")
        lines.append(r"\small")
        lines.append(r"\begin{tabular}{lcrrrr}")
        lines.append(r"\toprule")
        lines.append(
            r"Method & Seeds & rel-$L^2$ mean & rel-$L^2$ std "
            r"& Params & Train time \\"
        )
        lines.append(r"\midrule")
        for r in rows:
            entry = r[pde]
            if entry is None:
                continue
            arch_disp = ARCH_DISPLAY.get(r["arch"], r["arch"])
            lines.append(
                f"{arch_disp} & {entry['n_seeds']} & "
                f"${100 * entry['mean_of_seed_means']:.2f}\\%$ & "
                f"${100 * entry['std_of_seed_means']:.2f}\\%$ & "
                f"{entry['param_count']:,} & "
                f"{entry['train_time_mean_s']:.0f}s \\\\"
            )
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(rf"\caption{{Per-cell detail for {pde_disp}.}}")
        lines.append(rf"\label{{tab:detail_{safe_pde}}}")
        lines.append(r"\end{table}")
        lines.append("")

    lines.append(
        r"\subsection*{Pairwise comparisons against each architecture as baseline}"
    )
    lines.append("")
    for stats in all_stats:
        lines.extend(_pairwise_tex_table(stats, pde_order))
        lines.append("")

    lines.append(r"\subsection*{Per-seed results (one table per PDE)}")
    lines.append("")
    for pde in pde_order:
        loaded = all_results.get(pde)
        if not loaded:
            continue
        lines.extend(_per_seed_tex_table(loaded, arch_order))
        lines.append("")

    lines.append(r"\subsection*{Final-evaluation training compute}")
    lines.append("")
    tt_rows, tt_totals = _training_time_rows(all_results, arch_order)
    lines.extend(_training_time_tex(tt_rows, tt_totals))
    lines.append("")

    path = output_dir / "summary.tex"
    path.write_text("\n".join(lines) + "\n")
    return path


# Top-level runner
DEFAULT_ARCH_ORDER = ["pi_deeponet", "pit", "pinto", "keronet", "mkeronet"]
DEFAULT_PDE_ORDER = [
    "burgers",
    "advection",
    "diffusion_reaction",
    "allen_cahn",
    "allen_cahn_2d",
]


def run_report(
    results_dirs: List[Path],
    output_dir: Path,
    *,
    baseline: str = "keronet",
    arch_order: List[str] | None = None,
    pde_order: List[str] | None = None,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results = load_all_results(results_dirs)

    detected_pdes = sorted(all_results.keys())
    if pde_order is None:
        pde_order = [p for p in DEFAULT_PDE_ORDER if p in all_results]
        pde_order += [p for p in detected_pdes if p not in pde_order]

    if arch_order is None:
        seen_archs = set()
        for loaded in all_results.values():
            seen_archs.update(loaded["by_arch"].keys())
        arch_order = [a for a in DEFAULT_ARCH_ORDER if a in seen_archs]
        arch_order += [a for a in sorted(seen_archs) if a not in arch_order]

    print(f"PDEs: {pde_order}")
    print(f"Archs: {arch_order}")
    print(f"Primary baseline: {baseline}")

    rows = main_table_rows(all_results, arch_order, pde_order)

    tex_path = emit_main_table_latex(rows, pde_order, output_dir)
    print(f"  wrote {tex_path}")
    md_path = emit_main_table_md(rows, pde_order, output_dir)
    print(f"  wrote {md_path}")

    # Stats: one full pairwise dict per architecture as baseline.
    # Primary baseline first, others in arch_order.
    all_stats = compute_all_pairwise_stats(all_results, arch_order, baseline)

    stats_path = output_dir / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(
            {"by_baseline": all_stats, "primary_baseline": baseline},
            f,
            indent=2,
            sort_keys=True,
            default=str,
        )
    print(f"  wrote {stats_path}")

    tt_paths = emit_training_time_table(all_results, arch_order, output_dir)
    if tt_paths:
        for p in tt_paths:
            print(f"  wrote {p}")

    # Figures (per PDE)
    for pde in pde_order:
        loaded = all_results[pde]
        boxp = emit_boxplot(loaded, arch_order, output_dir)
        if boxp:
            print(f"  wrote {boxp}")
        stripp = emit_stripplot(loaded, arch_order, output_dir)
        if stripp:
            print(f"  wrote {stripp}")
        lossp = emit_loss_curves(loaded, arch_order, output_dir)
        if lossp:
            print(f"  wrote {lossp}")

    summ_md_path = emit_summary_md(
        all_results,
        arch_order,
        pde_order,
        rows,
        all_stats,
        output_dir,
    )
    print(f"  wrote {summ_md_path}")

    summ_tex_path = emit_summary_tex(
        all_results,
        arch_order,
        pde_order,
        rows,
        all_stats,
        output_dir,
    )
    print(f"  wrote {summ_tex_path}")

    print(f"\nReport assembled in: {output_dir}")
    return {
        "rows": rows,
        "all_stats": all_stats,
        "all_results": all_results,
        "arch_order": arch_order,
        "pde_order": pde_order,
    }


# CLI
def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--results",
        type=str,
        nargs="+",
        required=True,
        help="Result directories from evaluate.py (one per PDE).",
    )
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument(
        "--baseline",
        type=str,
        default="keronet",
        help="Architecture used as the primary baseline; its pairwise "
        "table appears first. Tables for every other architecture "
        "as baseline are also emitted.",
    )
    p.add_argument("--arch_order", type=str, nargs="+", default=None)
    p.add_argument("--pde_order", type=str, nargs="+", default=None)
    args = p.parse_args()

    run_report(
        results_dirs=[Path(r) for r in args.results],
        output_dir=Path(args.output_dir),
        baseline=args.baseline,
        arch_order=args.arch_order,
        pde_order=args.pde_order,
    )


if __name__ == "__main__":
    main()
