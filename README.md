# KerONet

Code, hyperparameter-optimization (HPO) studies, and evaluation artifacts for

> Seth Dale, Carolyn Koh, Dinesh Mehta. **KerONet: Rethinking Cross-Attention for
> Physics-Informed Operator Learning.** NeurIPS 2026.
> [OpenReview PDF](https://openreview.net/pdf?id=kaHFJVwIAG)

KerONet replaces PI-DeepONet's bilinear branch-trunk readout with a single
softmax-weighted readout: the trunk emits a query, the branch emits `S`
key/value slots, and one softmax combines them before a small post-readout MLP.
This repository compares it with PI-DeepONet, PIT, and PINTO under matched
parameter budgets and a shared HPO and evaluation protocol, all trained purely
from physics (PDE residual plus initial/boundary conditions).

## Contents

```
architectures/   KerONet, mKerONet (readout ablation), PI-DeepONet, PIT, PINTO
pdes/            Burgers, advection, diffusion-reaction, Allen-Cahn (1D and 2D)
hpo/             Optuna HPO: tune_common.py (protocol), search_spaces.py, tune_<arch>.py
evaluate.py      multi-seed training and evaluation of a selected configuration
run_hpo_parallel.py   job-level HPO scheduler across GPUs
plotting/        report.py (tables, bootstrap CIs, figures), report_hpo.py, solution figures,
                 plot_s_sweep.py, plot_convergence_2d.py, plot_solutions_2d.py,
                 plot_ac_error_by_time.py
tools/           paper_tables.py, verify_search_spaces.py, replay_trial.py,
                 extend_groups.py, config_from_trial.py, ac2d_baselines.py,
                 hpo_compute.py, ac2d_error_by_time.py, ac1d_error_by_time.py
tests/           unit and smoke tests
experiments/     every result in the paper: HPO studies, selected configs, per-seed evaluations
```

## Setup

Python 3.13 and a CUDA-12 GPU (JAX 0.5.3). With [uv](https://docs.astral.sh/uv/):

```bash
uv sync                      # installs the exact versions in uv.lock
uv run pytest -q tests       # unit and smoke tests (minutes on a GPU)
```

All commands below run from the repository root.

## Reproducing the paper

**Data.** Training sets, the HPO validation set (100 functions, seed 1234567),
and the test set (100 functions, seed 7654321) are generated on first use and
cached under `data_cache/` (override with `DATA_CACHE_DIR`). Generation is
deterministic given the seeds. `tools/ac2d_baselines.py` reproduces the 2D
benchmark checks, including the reference-solver check reported in Appendix J.

**HPO.** One study per (architecture, PDE, parameter budget). The search space
is chosen from `--max_params` (or `--space`); the protocol defaults in
`hpo/tune_common.add_common_args` are the paper's (20,000 iterations, median
pruning at 5,000, AdamW with cosine decay, mini-batches of 14 functions).

```bash
uv run hpo/tune_keronet.py --pde burgers --max_params 200000 \
    --target_completed 15 --n_trials 100
```

The paper's convergence rule (Section 4) runs at least 15 non-pruned trials
per architecture and continues until the best-so-far validation error has been
unchanged for two consecutive non-pruned trials in every architecture of a
group, extending all of the group's studies together. `tools/extend_groups.py`
applies this rule to existing studies. `tools/hpo_compute.py` recomputes, from
the study files, the trial count at which each group met the rule and each
study's HPO GPU-hours up to that point (the paper's HPO tables).
`tools/replay_trial.py` retrains a completed trial; GPU nondeterminism applies.

**Evaluation.** Five seeds of the selected configuration, each re-drawing the
training data, mini-batch order, and initialization:

```bash
uv run evaluate.py --pde burgers --seeds 0 1 2 3 4 \
    --configs experiments/eval/main_200k/burgers/keronet/seed0/config_used.yaml \
    --output_dir results/burgers --n_iter 20000 --N_test 100
```

**Tables.** `tools/paper_tables.py` regenerates every reported mean, standard
deviation, parameter count, training time, and paired bootstrap CI from
`experiments/eval/`, using the same routine as the paper
(`plotting/report.py:bootstrap_diff_ci`, 10,000 resamples, seed 0):

```bash
uv run tools/paper_tables.py            # all cells
uv run tools/paper_tables.py --cell main_200k
```

**Figures.** `plotting/plot_s_sweep.py` (KerONet's slot-count response,
Appendix E.4) and `plotting/plot_convergence_2d.py` (the 2D Allen-Cahn panel of
the HPO convergence figure) read `experiments/` directly; `plotting/report_hpo.py`
makes the 1D convergence panels from studies named as `hpo/tune_*.py` writes them
(`optuna_<arch>_<pde>.db`; link the released ones as below), and `plotting/report.py`
the strip plots and training-loss curves from `evaluate.py` output directories.
The solution panels, the advection error spectrum, and the Allen-Cahn
error-by-time figure and numbers (Section 5.3, Appendix G) need the test-set
predictions that `evaluate.py` writes to
`<output_dir>/<arch>_seed<k>/predictions.npy`, which this release does not
include (165 MB per 2D seed); re-run `evaluate.py` with the shipped
configurations to regenerate them. The paper's figures use:

```bash
# strip plots and training-loss curves (Appendix G), one results directory per PDE
uv run plotting/report.py --results results/burgers results/advection \
    results/diffusion_reaction results/allen_cahn results/allen_cahn_2d \
    --output_dir figures --baseline keronet
# 1D HPO convergence panels (Appendix E), from the released 200k studies
mkdir -p studies && for p in burgers advection diffusion_reaction allen_cahn; do
    for a in pi_deeponet pit pinto keronet; do
        ln -sf "$PWD/experiments/hpo/main_200k/$p/$a/study.db" "studies/optuna_${a}_${p}.db"; done; done
uv run plotting/report_hpo.py --studies_dir studies --output_dir figures
# 1D solution panels (Figure 2 and Appendix G); the others: advection --seed 0
# --ic_index 79, allen_cahn --seed 3 --ic_index 88, diffusion_reaction --seed 4 --ic_index 93
uv run plotting/make_pde_figures.py --results_root results \
    --pde_dir_template '{pde}' --pdes burgers --seed 1 --n_ics 1 --ic_selection fixed \
    --ic_index 72 --metric_scope per_ic --out_dir figures
# 2D Allen-Cahn solution panels (Appendix G) and error by time (Section 5.3, Appendix G)
uv run plotting/plot_solutions_2d.py --seed 0 --rank 50 --out figures \
    --results keronet=DIR pi_deeponet=DIR pinto=DIR pit=DIR
uv run tools/ac2d_error_by_time.py \
    --results keronet=DIR pi_deeponet=DIR pinto=DIR pit=DIR
# 1D Allen-Cahn error by time (Section 5.3)
uv run tools/ac1d_error_by_time.py \
    --results keronet=DIR pi_deeponet=DIR pinto=DIR pit=DIR
# Allen-Cahn error-versus-time figure, 1D and 2D (Section 5.3)
uv run plotting/plot_ac_error_by_time.py --out figures/ac_error_by_time.pdf \
    --ac1d keronet=DIR pinto=DIR --ac2d keronet=DIR pinto=DIR
# advection error spectrum (Appendix G)
uv run plotting/inspect_advection.py --results_root results \
    --pde_dir advection --arch keronet --seed 0 --ic_index 28 --out_dir figures
```

## Experiments directory

`experiments/manifest.json` lists every result: its cell (`main_200k/burgers`,
`2d_200k/allen_cahn_2d`, `budget_500k/allen_cahn`, ...), architecture, the
paper tables it appears in (by LaTeX label), its status, and provenance notes.

| path | contents |
|---|---|
| `experiments/eval/<cell>/<arch>/seed<k>/` | `meta.json` (metrics, timing, environment), `per_ic_errors.npy` (rel-L2 per test function), `config_used.yaml`, `loss_log.npz` |
| `experiments/hpo/<cell>/<arch>/` | `study.db` (the Optuna study, all trials), `trials.csv`; the manifest's `rule_completed` is the trial count at which the convergence rule holds |


Model checkpoints (about 210 MB, Python pickles) are distributed separately
as a [release asset](https://github.com/sddale/keronet/releases) with the same
`<cell>/<arch>/seed<k>/` layout; only load pickles you trust. From the
repository root, this puts each `checkpoint.pkl` next to its `meta.json`:

```bash
tar --zstd --strip-components=1 -xf keronet-release-checkpoints.tar.zst
```

`tools/verify_search_spaces.py` checks every study in `experiments/hpo/`
against `hpo/search_spaces.py` (parameters, choices, and suggestion order) and
reports its convergence status.

## Hardware

Results were produced on NVIDIA RTX 4090 GPUs (24 GB), one job per GPU.
Wall-clock training times are hardware-specific.

## Citation

See `CITATION.cff`.

## License

MIT (see `LICENSE`).
