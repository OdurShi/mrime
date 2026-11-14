#!/usr/bin/env python
"""Write ``report.md`` for one or more experiment directories.

The report states the protocol actually used (from ``meta.json``), the mean true number of
objective evaluations per run, mean (std) of every algorithm per function, the Wilcoxon
rank-sum win/equal/loss counts against a baseline and the Friedman mean ranks.

Example::

    python scripts/summarize_results.py results/reduced/cec2022_d10 --baseline rime
"""

import argparse
import json
from pathlib import Path

from mrime.analysis import (
    friedman_test,
    load_runs,
    markdown_table,
    summary_rows,
    win_equal_loss,
)


def report(directory: Path, baseline: str | None) -> str:
    table = load_runs(directory)
    meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    config = meta["config"]
    baseline = baseline or table.algorithms[0]
    rows = {(r["function"], r["algorithm"]): r for r in summary_rows(table)}
    lines = [
        f"# {config['name']}: {table.suite.upper()}, D = {table.dim}",
        "",
        f"- paper reference: {config['paper_reference'] or 'n/a'}",
        f"- runs per function and algorithm: {config['runs']}",
        f"- FEsmax = {config['resolved_max_evals']}, NP = {config['resolved_pop_size']}, "
        f"base seed = {config['seed']}, backend = {config['backend']}, "
        f"MATLAB bounds for CEC2022 F11 = {config['matlab_bounds']}",
        f"- wall-clock time {meta['wall_time_s']} s with {config['workers']} worker processes "
        f"({meta['cpu_time_s']} s summed over runs), {meta['platform']}, "
        f"Python {meta['versions']['python']}, NumPy {meta['versions']['numpy']}",
        "",
        "## True number of objective evaluations per run (mean over all functions and runs)",
        "",
    ]
    evals = []
    for algorithm in table.algorithms:
        per_function = [rows[(f, algorithm)]["mean_evals"] for f in table.functions]
        evals.append([algorithm, f"{sum(per_function) / len(per_function):.0f}"])
    lines.append(markdown_table(["algorithm", "evaluations"], evals))
    lines += ["## Mean (standard deviation) of the final objective value", ""]
    body = []
    for function in table.functions:
        best = min(rows[(function, a)]["mean"] for a in table.algorithms)
        cells = [f"F{function}"]
        for algorithm in table.algorithms:
            row = rows[(function, algorithm)]
            text = f"{row['mean']:.4e} ({row['std']:.2e})"
            cells.append(f"**{text}**" if row["mean"] == best else text)
        body.append(cells)
    lines.append(markdown_table(["function", *table.algorithms], body))

    others = [a for a in table.algorithms if a != baseline]
    if others:
        lines += [
            f"## Wilcoxon rank-sum test against `{baseline}` (5 % level)",
            "",
            "`+` / `=` / `-`: significantly better / not significantly different / "
            f"significantly worse than `{baseline}`.",
            "",
        ]
        comparisons = [win_equal_loss(table, a, baseline) for a in others]
        signs = [
            [f"F{f}", *(c.signs[f] for c in comparisons)] for f in table.functions
        ]
        signs.append(["**w/e/l**", *(f"**{c}**" for c in comparisons)])
        lines.append(markdown_table(["function", *others], signs))

    friedman = friedman_test(table)
    lines += ["## Friedman test on the mean values", ""]
    lines.append(markdown_table(
        ["algorithm", "mean rank"],
        sorted(([a, f"{r:.2f}"] for a, r in friedman.as_dict().items()), key=lambda x: float(x[1])),
    ))
    lines.append(f"p-value: {friedman.p_value:.3e}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", nargs="+", type=Path, help="directories containing runs.csv")
    parser.add_argument("--baseline", help="reference algorithm (default: first in the file)")
    args = parser.parse_args()
    for directory in args.results:
        (directory / "report.md").write_text(report(directory, args.baseline), encoding="utf-8")
        print(directory / "report.md")


if __name__ == "__main__":
    main()
