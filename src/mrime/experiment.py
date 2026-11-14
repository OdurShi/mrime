"""Experiment runner: suite x functions x algorithms x independent runs, with raw outputs.

An experiment writes three files into its output directory:

``runs.csv``
    one row per independent run (final best value, error, true evaluation count, seed, ...).
``curves.csv.gz``
    one row per run with the best-so-far value sampled on an even grid of the evaluation
    budget (the convergence curve of that run).
``meta.json``
    the resolved configuration, software versions and wall-clock time.

Seeds depend on (base seed, suite, dimension, function, run) only, so every algorithm starts
run ``r`` of a function from the same initial population, as in the author's MATLAB scripts.
"""

from __future__ import annotations

import csv
import gzip
import json
import logging
import math
import os
import platform
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, fields
from functools import lru_cache
from pathlib import Path

import numpy as np
import scipy
import yaml

import mrime
from mrime.benchmarks import SUITES, Problem, get_problem, suite_functions
from mrime.config import parse_algorithm
from mrime.core import minimize

logger = logging.getLogger(__name__)

RUN_FIELDS: tuple[str, ...] = (
    "suite", "function", "dim", "algorithm", "run", "seed", "best", "error", "n_evals",
    "max_evals", "pop_size", "n_generations", "gcls_candidates", "spdm_candidates",
    "spdm_accepted", "time_s",
)
CURVE_KEYS: tuple[str, ...] = ("suite", "function", "dim", "algorithm", "run")


@dataclass(frozen=True)
class ExperimentConfig:
    """Everything that defines an experiment.

    Attributes:
        name: experiment name (used for the default output directory).
        suite: ``"classical"``, ``"cec2017"`` or ``"cec2022"``.
        dim: problem dimension.
        algorithms: algorithm specs ``"<variant>[@<profile>]"``, see :func:`mrime.minimize`.
        runs: number of independent runs per (function, algorithm); the paper uses 51.
        functions: function ids to run; ``None`` means the whole suite.
        max_evals: evaluation budget; ``None`` means ``evals_per_dim * dim``.
        evals_per_dim: budget per dimension; the paper uses 3000 (Table 2).
        pop_size: population size; ``None`` means ``ceil(pop_coefficient * dim**(2/3))``.
        pop_coefficient: coefficient of the population-size rule of the author's run scripts
            (15 in ``Run_CEC2017_total_MRIME.m``, 30 in the other three scripts). The paper
            does not state the population size.
        seed: base seed.
        workers: number of worker processes.
        backend: CEC backend, ``"native"`` or ``"opfunu"`` (ignored by the classical suite).
        matlab_bounds: use ``[-600, 600]`` for CEC2022 F11 as ``Get_Functions_cec2022.m`` does.
        curve_points: number of grid points of the stored convergence curves.
        paper_reference: which table/figure of the paper the experiment corresponds to.
    """

    name: str
    suite: str
    dim: int
    algorithms: tuple[str, ...]
    runs: int
    functions: tuple[int, ...] | None = None
    max_evals: int | None = None
    evals_per_dim: int = 3000
    pop_size: int | None = None
    pop_coefficient: float = 15.0
    seed: int = 2025
    workers: int = 1
    backend: str = "native"
    matlab_bounds: bool = False
    curve_points: int = 51
    paper_reference: str = ""

    def __post_init__(self) -> None:
        if self.suite not in SUITES:
            raise ValueError(f"unknown suite {self.suite!r}; choose from {SUITES}")
        if self.runs < 1 or self.dim < 2 or self.workers < 1 or self.curve_points < 2:
            raise ValueError("runs >= 1, dim >= 2, workers >= 1 and curve_points >= 2 required")
        if not self.algorithms:
            raise ValueError("at least one algorithm is required")
        for spec in self.algorithms:
            parse_algorithm(spec)
        valid = set(suite_functions(self.suite))
        for function in self.functions or ():
            if function not in valid:
                raise ValueError(f"{self.suite} has no function {function}")

    @property
    def budget(self) -> int:
        """Resolved evaluation budget ``FEsmax``."""
        return self.max_evals if self.max_evals is not None else self.evals_per_dim * self.dim

    @property
    def population(self) -> int:
        """Resolved population size ``NP``."""
        if self.pop_size is not None:
            return self.pop_size
        return math.ceil(self.pop_coefficient * self.dim ** (2.0 / 3.0))

    @property
    def function_ids(self) -> tuple[int, ...]:
        """Resolved function ids."""
        return self.functions if self.functions is not None else tuple(suite_functions(self.suite))


def load_config(path: str | Path, **overrides) -> ExperimentConfig:
    """Read an experiment from a YAML file; keyword arguments override file values."""
    with open(path, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    known = {f.name for f in fields(ExperimentConfig)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"unknown keys in {path}: {sorted(unknown)}")
    raw.update({key: value for key, value in overrides.items() if value is not None})
    for key in ("algorithms", "functions"):
        if raw.get(key) is not None:
            raw[key] = tuple(raw[key])
    return ExperimentConfig(**raw)


def run_seed(base_seed: int, suite: str, dim: int, function: int, run: int) -> int:
    """Seed of run ``run`` of ``function``; identical for all algorithms."""
    sequence = np.random.SeedSequence([base_seed, SUITES.index(suite), dim, function, run])
    return int(sequence.generate_state(1)[0])


@lru_cache(maxsize=8)
def _problem(suite: str, function: int, dim: int, backend: str, matlab_bounds: bool) -> Problem:
    return get_problem(suite, function, dim, backend=backend, matlab_bounds=matlab_bounds)


def _run_block(args: tuple[ExperimentConfig, int, str]) -> list[dict]:
    """All runs of one (function, algorithm) pair; executed inside a worker process."""
    config, function, algorithm = args
    problem = _problem(config.suite, function, config.dim, config.backend, config.matlab_bounds)
    grid = np.linspace(0.0, config.budget, config.curve_points)
    rows = []
    for run in range(config.runs):
        seed = run_seed(config.seed, config.suite, config.dim, function, run)
        started = time.perf_counter()
        result = minimize(
            problem, problem.lower, problem.upper, config.dim, config.population,
            config.budget, seed, algorithm=algorithm,
        )
        elapsed = time.perf_counter() - started
        error = result.fun - problem.f_opt if problem.f_opt is not None else float("nan")
        rows.append({
            "suite": config.suite,
            "function": function,
            "dim": config.dim,
            "algorithm": algorithm,
            "run": run,
            "seed": seed,
            "best": result.fun,
            "error": error,
            "n_evals": result.n_evals,
            "max_evals": config.budget,
            "pop_size": config.population,
            "n_generations": result.n_generations,
            "gcls_candidates": result.extras["gcls_candidates"],
            "spdm_candidates": result.extras["spdm_candidates"],
            "spdm_accepted": result.extras["spdm_accepted"],
            "time_s": round(elapsed, 4),
            "curve": result.best_at(grid),
        })
    return rows


def run_experiment(config: ExperimentConfig, output_dir: str | Path) -> Path:
    """Run ``config`` and write ``runs.csv``, ``curves.csv.gz`` and ``meta.json``.

    Returns:
        The output directory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    blocks = [
        (config, function, algorithm)
        for function in config.function_ids
        for algorithm in config.algorithms
    ]
    started = time.time()
    if config.workers == 1:
        results = [_run_block(block) for block in blocks]
    else:
        # One BLAS thread per worker: the workload is many small independent runs.
        for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            os.environ.setdefault(variable, "1")
        with ProcessPoolExecutor(max_workers=config.workers) as pool:
            results = list(pool.map(_run_block, blocks))
    wall = time.time() - started
    rows = [row for block in results for row in block]

    with open(output_dir / "runs.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RUN_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "best": repr(row["best"]), "error": repr(row["error"])})

    header = [*CURVE_KEYS, *(f"p{i}" for i in range(config.curve_points))]
    with gzip.open(output_dir / "curves.csv.gz", "wt", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in rows:
            keys = [row[key] for key in CURVE_KEYS]
            writer.writerow([*keys, *(f"{value:.10g}" for value in row["curve"])])

    meta = {
        "config": {**asdict(config), "resolved_max_evals": config.budget,
                   "resolved_pop_size": config.population,
                   "resolved_functions": list(config.function_ids)},
        "n_runs_total": len(rows),
        "wall_time_s": round(wall, 2),
        "cpu_time_s": round(sum(row["time_s"] for row in rows), 2),
        "versions": {
            "mrime": mrime.__version__,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "platform": f"{platform.system()} {platform.machine()}",
        "curve_grid": "best-so-far at evenly spaced evaluation-counter values in [0, max_evals]",
    }
    with open(output_dir / "meta.json", "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
        handle.write("\n")
    logger.info("wrote %d runs to %s in %.1f s", len(rows), output_dir, wall)
    return output_dir
