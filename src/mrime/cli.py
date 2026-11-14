"""Command-line interface: ``mrime demo | run | analyze | plot | list | verify-cec``."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from mrime.analysis import analyze, load_runs, summary_markdown
from mrime.benchmarks import CLASSICAL_NAMES, SUITES, get_problem
from mrime.config import MATLAB_FILES, PROFILES, VARIANTS
from mrime.core import minimize
from mrime.experiment import ExperimentConfig, load_config, run_experiment

logger = logging.getLogger("mrime")

DATA_DIR_HELP = (
    "folder with cec17_func.cpp, cec22_func.cpp, input_data/ and input_data22/ "
    "(default: matlab/ of the checkout, or $MRIME_CEC_SOURCE_DIR); see data/README.md"
)


def _cmd_demo(args: argparse.Namespace) -> int:
    """RIME against MRIME-CD on built-in functions; needs no data and runs in seconds."""
    dim, pop, budget = args.dim, args.pop_size, args.max_evals
    names = ("sphere", "rosenbrock", "ackley", "ellipsoid")
    print(f"dim={dim}  pop_size={pop}  max_evals={budget}  runs={args.runs}  (median final value)")
    print(f"{'function':<12}{'RIME':>14}{'MRIME-CD':>14}{'evals':>10}")
    started = time.perf_counter()
    for name in names:
        problem = get_problem("classical", name, dim)
        medians = []
        for algorithm in ("rime", "mrime-cd"):
            results = [
                minimize(problem, problem.lower, problem.upper, dim, pop, budget,
                         seed=args.seed + run, algorithm=algorithm)
                for run in range(args.runs)
            ]
            medians.append(float(np.median([r.fun for r in results])))
            evals = results[0].n_evals
        print(f"{name:<12}{medians[0]:>14.4e}{medians[1]:>14.4e}{evals:>10d}")
    print(f"done in {time.perf_counter() - started:.1f} s")
    return 0


def _use_data_dir(data_dir: str | None) -> None:
    """Point the native CEC backend (in this process and its workers) at ``data_dir``."""
    if data_dir:
        os.environ["MRIME_CEC_SOURCE_DIR"] = str(Path(data_dir).expanduser().resolve())


def _cmd_run(args: argparse.Namespace) -> int:
    _use_data_dir(args.data_dir)
    overrides = {
        "runs": args.runs, "workers": args.workers, "seed": args.seed, "dim": args.dim,
        "max_evals": args.max_evals, "pop_size": args.pop_size, "backend": args.backend,
        "algorithms": args.algorithms, "functions": args.functions,
    }
    if args.config:
        config = load_config(args.config, **overrides)
    else:
        if not (args.suite and args.dim and args.algorithms):
            print("without --config, --suite, --dim and --algorithms are required", file=sys.stderr)
            return 2
        given = {key: value for key, value in overrides.items() if value is not None}
        given["algorithms"] = tuple(given["algorithms"])
        if "functions" in given:
            given["functions"] = tuple(given["functions"])
        given.setdefault("runs", 5)
        config = ExperimentConfig(name=args.suite, suite=args.suite, **given)
    output = Path(args.output or Path("results") / f"{config.name}_D{config.dim}")
    logger.info(
        "%s: %s D=%d, %d functions x %d algorithms x %d runs, FEsmax=%d, NP=%d, workers=%d",
        config.name, config.suite, config.dim, len(config.function_ids), len(config.algorithms),
        config.runs, config.budget, config.population, config.workers,
    )
    run_experiment(config, output)
    summary = analyze(output, baseline=args.baseline)
    print(summary_markdown(load_runs(output)))
    for algorithm, wel in summary["win_equal_loss"].items():
        print(f"Wilcoxon rank-sum, {algorithm} vs {summary['baseline']} (w/e/l): {wel}")
    print(f"results written to {output}")
    return 0


def _cmd_analyze(args: argparse.Namespace) -> int:
    summary = analyze(args.results, baseline=args.baseline, alpha=args.alpha)
    print(summary_markdown(load_runs(args.results)))
    friedman = summary["friedman"]
    print(f"Friedman p-value: {friedman.p_value:.3e}")
    for algorithm, wel in summary["win_equal_loss"].items():
        print(f"Wilcoxon rank-sum, {algorithm} vs {summary['baseline']} (w/e/l): {wel}")
    return 0


def _cmd_plot(args: argparse.Namespace) -> int:
    from mrime.plotting import make_figures

    for path in make_figures(args.results, args.algorithms, args.functions):
        print(path)
    return 0


def _cmd_list(_: argparse.Namespace) -> int:
    print("variants (paper Table 5) and the MATLAB file implementing each:")
    for variant, (gcls, abs_, spdm) in VARIANTS.items():
        flags = " ".join(
            f"{name}={'Y' if on else 'N'}"
            for name, on in (("GCLS", gcls), ("ABS", abs_), ("SPDM", spdm))
        )
        print(f"  {variant:<9} {flags}   matlab/{MATLAB_FILES[variant]}")
    print(f"profiles (append as '@profile'): {', '.join(PROFILES)}")
    print(f"suites: {', '.join(SUITES)}")
    print(f"classical functions: {', '.join(CLASSICAL_NAMES)}")
    return 0


def _cmd_verify_cec(args: argparse.Namespace) -> int:
    from mrime.benchmarks.verify import verify_backends

    _use_data_dir(args.data_dir)
    print(verify_backends(points=args.points, seed=args.seed))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mrime", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true", help="log progress")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="RIME vs MRIME-CD on built-in functions (seconds)")
    demo.add_argument("--dim", type=int, default=10)
    demo.add_argument("--pop-size", type=int, default=70)
    demo.add_argument("--max-evals", type=int, default=30_000)
    demo.add_argument("--runs", type=int, default=5)
    demo.add_argument("--seed", type=int, default=0)
    demo.set_defaults(handler=_cmd_demo)

    run = sub.add_parser("run", help="run an experiment and write raw results")
    run.add_argument("--config", help="YAML file, see configs/")
    run.add_argument("--suite", choices=SUITES)
    run.add_argument("--dim", type=int)
    run.add_argument("--algorithms", nargs="+", help="e.g. rime mrime-cd rime-ga@matlab")
    run.add_argument("--functions", nargs="+", type=int)
    run.add_argument("--runs", type=int)
    run.add_argument("--max-evals", type=int)
    run.add_argument("--pop-size", type=int)
    run.add_argument("--seed", type=int)
    run.add_argument("--workers", type=int)
    run.add_argument("--backend", choices=("native", "opfunu"))
    run.add_argument("--data-dir", help=DATA_DIR_HELP)
    run.add_argument("--baseline", help="reference algorithm of the rank-sum test")
    run.add_argument("--output", help="output directory (default: results/<name>_D<dim>)")
    run.set_defaults(handler=_cmd_run)

    analyze_cmd = sub.add_parser("analyze", help="tables and statistical tests from runs.csv")
    analyze_cmd.add_argument("results", help="directory containing runs.csv")
    analyze_cmd.add_argument("--baseline")
    analyze_cmd.add_argument("--alpha", type=float, default=0.05)
    analyze_cmd.set_defaults(handler=_cmd_analyze)

    plot = sub.add_parser("plot", help="convergence curves, box plots and Friedman ranks")
    plot.add_argument("results", help="directory containing runs.csv and curves.csv.gz")
    plot.add_argument("--algorithms", nargs="+")
    plot.add_argument("--functions", nargs="+", type=int)
    plot.set_defaults(handler=_cmd_plot)

    list_cmd = sub.add_parser("list", help="available variants, profiles and suites")
    list_cmd.set_defaults(handler=_cmd_list)

    verify = sub.add_parser(
        "verify-cec", help="check the CEC backends against the reference C code"
    )
    verify.add_argument("--points", type=int, default=5)
    verify.add_argument("--data-dir", help=DATA_DIR_HELP)
    verify.add_argument("--seed", type=int, default=0)
    verify.set_defaults(handler=_cmd_verify_cec)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point of the ``mrime`` console script."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(name)s %(message)s",
    )
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
