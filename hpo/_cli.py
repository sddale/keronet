"""Shared command-line entry point for the hpo/tune_*.py scripts.

run_cli adds --space (default chosen from --max_params) and, if has_sigma, --sigma to the
arguments of tune_common.add_common_args, fixes n_sensors and weight_decay, and calls
tune_common.run_study.
"""
import argparse

from pdes import get_pde
from hpo.search_spaces import DEFAULT_SPACE_FOR_CAP, SEARCH_SPACES, make_sampler
from hpo.tune_common import SearchConfig, add_common_args, run_study


def run_cli(arch_name, arch_class, doc, has_sigma=False):
    p = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    p.add_argument(
        "--space", choices=sorted(SEARCH_SPACES[arch_name]), default=None,
        help="Search space (hpo/search_spaces.py). Default: chosen from --max_params "
             "(100000 -> 100k, 200000 -> 200k, 300000 -> 300k, 500000 -> 500k).")
    if has_sigma:
        p.add_argument("--sigma", type=float, default=None,
                       help="Trunk Fourier-feature scale (default: architecture default, 0.5).")
    args = p.parse_args()

    space = args.space
    if space is None:
        space = DEFAULT_SPACE_FOR_CAP.get(args.max_params)
        if space is None or space not in SEARCH_SPACES[arch_name]:
            p.error(f"no default search space for {arch_name} at --max_params {args.max_params}; "
                    f"pass --space (available: {sorted(SEARCH_SPACES[arch_name])})")

    pde_spec = get_pde(args.pde)
    fixed = {"n_sensors": pde_spec.default_training_kwargs.get("m", 80),
             "weight_decay": args.weight_decay}
    if has_sigma and args.sigma is not None:
        fixed["sigma"] = args.sigma
    print(f"[{arch_name}] search space '{space}', fixed hyperparameters {fixed}")

    search = SearchConfig(arch_name=arch_name, arch_class=arch_class,
                          sample_hparams=make_sampler(arch_name, space), fixed_hparams=fixed,
                          pde_spec=pde_spec, pde_kwargs={})
    return run_study(
        search, n_trials=args.n_trials, target_completed=args.target_completed, n_iter=args.n_iter,
        intermediate_iter=args.intermediate_iter, N_test=args.N_test, eval_seed=args.eval_seed,
        train_seed=args.train_seed, max_params=args.max_params, predict_chunk=args.predict_chunk,
        bcs_batch=args.bcs_batch, res_batch=args.res_batch, init_lr=args.init_lr,
        end_lr_frac=args.end_lr_frac, warmup_frac=args.warmup_frac, weight_decay=args.weight_decay,
        study_name=args.study_name, storage=args.storage, out_dir=args.out_dir,
        config_dir=args.config_dir, config_version=args.config_version,
        sampler_seed=args.sampler_seed)
