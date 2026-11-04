"""Run HPO studies for (architecture, PDE) cells in parallel, one job per GPU.

Each cell runs hpo/tune_<arch>.py (plus any unrecognized flags), logging to <log_dir>/<arch>_<pde>.log;
cells whose study already has --target_completed completed trials are skipped, so re-running resumes.

Usage (from the repository root): uv run run_hpo_parallel.py --target_completed 15 --gpus 0 1
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


# Job specification


ARCHS_DEFAULT = ["keronet", "pinto", "pit", "pi_deeponet", "mkeronet"]
PDES_DEFAULT = ["burgers", "advection", "allen_cahn", "diffusion_reaction"]

# Map architecture key -> tune script
TUNE_SCRIPTS = {
    "keronet": "hpo/tune_keronet.py",
    "mkeronet": "hpo/tune_mkeronet.py",
    "pinto": "hpo/tune_pinto.py",
    "pit": "hpo/tune_pit.py",
    "pi_deeponet": "hpo/tune_pi_deeponet.py",
}


# Approximate wall-clock per non-pruned trial (seconds), per architecture.
# Calibrated from completed Burgers HPO. Used only to schedule longest-job-
# first so the slow architectures don't end up trailing on a single GPU at
# the end of the run. Numbers don't need to be precise --- they just need to
# rank-order the architectures correctly.
TRIAL_TIME_ESTIMATE = {
    "pit": 1200.0,  # ~20 min/trial; n_blocks dominates wall-time
    "pinto": 850.0,  # ~14 min/trial at n_cross_attn=3
    "keronet": 330.0,  # ~5.5 min/trial
    "pi_deeponet": 330.0,  # ~5.5 min/trial; comparable to KerONet
    "mkeronet": 330.0,  # ~5.5 min/trial
}


@dataclass
class Job:
    arch: str
    pde: str

    @property
    def key(self) -> str:
        return f"{self.arch}_{self.pde}"

    @property
    def db_path(self) -> Path:
        return Path(f"optuna_{self.key}.db")

    @property
    def trial_time_estimate(self) -> float:
        return TRIAL_TIME_ESTIMATE[self.arch]


def make_job_list(archs: List[str], pdes: List[str]) -> List[Job]:
    """Return one Job per (arch, pde), stably sorted by decreasing estimated trial time."""
    jobs = [Job(arch, pde) for arch in archs for pde in pdes]
    # Stable sort by descending trial-time estimate. Stable so the
    # caller's intra-architecture PDE order is preserved.
    jobs.sort(key=lambda j: -j.trial_time_estimate)
    return jobs


# Optuna study inspection (for resumability)


def n_completed_trials(job: Job) -> Optional[int]:
    """Return the number of completed trials in the job's study DB, or None if it is missing or unreadable."""
    if not job.db_path.exists():
        return None
    try:
        import optuna

        study = optuna.load_study(
            study_name=job.key,
            storage=f"sqlite:///{job.db_path}",
        )
        completed = sum(
            1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        )
        return completed
    except Exception as e:
        print(f"  [warn] could not load existing study for {job.key}: {e}")
        return None


# Worker process


def launch_job(
    job: Job,
    gpu_id: int,
    target_completed: int,
    n_trials_cap: int,
    log_dir: Path,
    extra_args: List[str],
) -> subprocess.Popen:
    """Start the job's tune script on one GPU, logging to <log_dir>/<arch>_<pde>.log; return the Popen."""
    script = TUNE_SCRIPTS[job.arch]
    cmd = [
        sys.executable,
        "-u",
        script,
        "--pde",
        job.pde,
        "--target_completed",
        str(target_completed),
        "--n_trials",
        str(n_trials_cap),
        *extra_args,
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env.pop("XLA_PYTHON_CLIENT_ALLOCATOR", None)
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
    env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.90"
    env["TF_CPP_MIN_LOG_LEVEL"] = "3"
    # Keep PYTHONPATH so siblings (architectures/, pdes/, training_schedules)
    # are importable when tune_*.py runs from cwd.
    env["PYTHONPATH"] = str(Path.cwd()) + os.pathsep + env.get("PYTHONPATH", "")

    log_path = log_dir / f"{job.key}.log"
    log_fh = open(log_path, "ab", buffering=0)

    log_fh.write(
        f"\n\n=== launched {time.strftime('%Y-%m-%d %H:%M:%S')} on GPU {gpu_id} ===\n"
        f"=== command: {' '.join(cmd)} ===\n\n".encode()
    )
    log_fh.flush()

    proc = subprocess.Popen(
        cmd,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        env=env,
    )
    proc._log_fh = log_fh  # type: ignore[attr-defined]
    proc._log_path = log_path  # type: ignore[attr-defined]
    proc._job = job  # type: ignore[attr-defined]
    proc._gpu_id = gpu_id  # type: ignore[attr-defined]
    proc._start_time = time.time()  # type: ignore[attr-defined]
    return proc


def reap_proc(proc: subprocess.Popen) -> tuple[Job, int, int, float]:
    """Close the job's log file and return (job, gpu_id, returncode, elapsed_seconds)."""
    proc._log_fh.close()  # type: ignore[attr-defined]
    elapsed = time.time() - proc._start_time  # type: ignore[attr-defined]
    return proc._job, proc._gpu_id, proc.returncode, elapsed  # type: ignore[attr-defined]


# Top-level scheduler


def run_all(
    jobs: List[Job],
    gpus: List[int],
    target_completed: int,
    n_trials_cap: int,
    log_dir: Path,
    extra_args: List[str],
    dry_run: bool = False,
) -> int:
    log_dir.mkdir(parents=True, exist_ok=True)

    #  Filter out already-complete studies
    pending: List[Job] = []
    skipped: List[tuple[Job, int]] = []
    for job in jobs:
        n_done = n_completed_trials(job)
        if n_done is not None and n_done >= target_completed:
            skipped.append((job, n_done))
        else:
            pending.append(job)

    print(f"\n{'=' * 72}")
    print(f"HPO runner --- {len(jobs)} total cells")
    print(f"{'=' * 72}")
    print(f"Target completed trials per cell: {target_completed}")
    print(f"GPUs available:                   {gpus}")
    print(f"Log directory:                    {log_dir}")
    print(f"Cells already at target:          {len(skipped)}")
    for job, n_done in skipped:
        print(f"  ✓ {job.key}  (already has {n_done} completed trials)")
    print(f"Cells to run/resume:              {len(pending)}")
    print(f"  (ordered longest-first to balance load across GPUs)")
    for job in pending:
        n_done = n_completed_trials(job)
        marker = "+" if n_done is None else "↻"
        prog = "" if n_done is None else f"  ({n_done}/{target_completed} done)"
        est_min = job.trial_time_estimate / 60.0
        print(f"  {marker} {job.key:32s}  ~{est_min:.0f} min/trial{prog}")

    if dry_run:
        print(f"\n[dry-run] exiting without launching")
        return 0

    if not pending:
        print(f"\nAll cells already at target. Nothing to do.")
        return 0

    print(f"\n{'=' * 72}")
    print(f"Starting parallel HPO with {len(gpus)} workers")
    print(f"{'=' * 72}\n")

    #  Worker pool
    available_gpus = list(gpus)  # GPU IDs ready to take a job
    running: list[subprocess.Popen] = []  # active processes
    queue = list(pending)  # jobs not yet started
    completed_jobs: list[tuple[Job, int, float]] = []  # (job, rc, elapsed)
    overall_start = time.time()

    while queue or running:
        # Launch as many jobs as we have free GPUs and queued work
        while queue and available_gpus:
            gpu_id = available_gpus.pop(0)
            job = queue.pop(0)
            proc = launch_job(
                job,
                gpu_id,
                target_completed,
                n_trials_cap,
                log_dir,
                extra_args,
            )
            running.append(proc)
            print(
                f"[{time_since(overall_start)}] launched  "
                f"{job.key:32s} on GPU {gpu_id}  "
                f"(log: {proc._log_path})"  # type: ignore[attr-defined]
            )

        # Wait for at least one job to finish
        if running:
            time.sleep(5)  # poll interval; tune_*.py runs for tens of minutes
            still_running = []
            for proc in running:
                if proc.poll() is None:
                    still_running.append(proc)
                    continue
                job, gpu_id, rc, elapsed = reap_proc(proc)
                available_gpus.append(gpu_id)
                completed_jobs.append((job, rc, elapsed))
                status = "ok" if rc == 0 else f"FAILED (rc={rc})"
                final_n = n_completed_trials(job)
                print(
                    f"[{time_since(overall_start)}] finished  "
                    f"{job.key:32s} on GPU {gpu_id}  "
                    f"in {format_elapsed(elapsed)}  [{status}]  "
                    f"completed_trials={final_n}"
                )
            running = still_running

    #  Summary
    print(f"\n{'=' * 72}")
    print(f"HPO runner finished in {format_elapsed(time.time() - overall_start)}")
    print(f"{'=' * 72}")
    n_failed = sum(1 for _, rc, _ in completed_jobs if rc != 0)
    print(f"Jobs run:    {len(completed_jobs)}")
    print(f"Failed:      {n_failed}")
    if n_failed:
        print(f"\nFailed cells:")
        for job, rc, _ in completed_jobs:
            if rc != 0:
                print(f"  ✗ {job.key}  (rc={rc}, see log)")
        print(f"\nRe-running this script will resume failed cells.")
    return 0 if n_failed == 0 else 1


# Pretty-printing helpers


def time_since(start: float) -> str:
    """Return the time elapsed since start as HH:MM:SS."""
    return format_elapsed(time.time() - start)


def format_elapsed(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# CLI


def main() -> int:
    p = argparse.ArgumentParser(
        description="Run HPO for all (arch, PDE) cells in parallel across GPUs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage", 1)[1] if __doc__ else None,
    )
    p.add_argument(
        "--target_completed",
        type=int,
        default=15,
        help="Non-pruned trials to accumulate per cell (default: 15).",
    )
    p.add_argument(
        "--gpus",
        type=int,
        nargs="+",
        default=[0, 1],
        help="GPU IDs to use (default: 0 1).",
    )
    p.add_argument(
        "--n_trials_cap",
        type=int,
        default=100,
        help="Per-session safety cap on total trials attempted "
        "before giving up. Should be >> target_completed to "
        "allow for pruned trials. (default: 100)",
    )
    p.add_argument(
        "--log_dir",
        type=str,
        default="hpo_logs",
        help="Directory for per-job stdout/stderr logs.",
    )
    p.add_argument(
        "--archs",
        type=str,
        nargs="+",
        default=ARCHS_DEFAULT,
        choices=ARCHS_DEFAULT,
        help=f"Architectures (default: all of {ARCHS_DEFAULT}).",
    )
    p.add_argument(
        "--pdes",
        type=str,
        nargs="+",
        default=PDES_DEFAULT,
        help=f"PDEs (default: {PDES_DEFAULT}).",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Print the job list and exit without running.",
    )
    # Any flags this runner doesn't recognize are passed through to
    # tune_*.py. Optional leading `--` separator is accepted but ignored.
    args, extra = p.parse_known_args()

    # Strip a leading '--' separator if the user used the documented form
    # `... -- --config_version 2`. argparse leaves it in extra otherwise.
    if extra and extra[0] == "--":
        extra = extra[1:]

    # Sanity-check that all tune scripts exist where we expect
    missing = [s for s in TUNE_SCRIPTS.values() if not Path(s).exists()]
    if missing:
        print(f"ERROR: tune scripts not found in cwd: {missing}", file=sys.stderr)
        print("       run from the repository root.", file=sys.stderr)
        return 2

    jobs = make_job_list(args.archs, args.pdes)
    return run_all(
        jobs=jobs,
        gpus=args.gpus,
        target_completed=args.target_completed,
        n_trials_cap=args.n_trials_cap,
        log_dir=Path(args.log_dir),
        extra_args=extra,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    sys.exit(main())
