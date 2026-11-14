"""Statistics in the style of the paper: summary tables and the three rank-based tests.

The functions mirror what the author's MATLAB scripts compute:

* per function and algorithm: best, mean, standard deviation (sample, ``ddof=1``) and the rank
  of the mean (Tables 10, 11, 17-20);
* Wilcoxon rank-sum test at the 5 % level, summarised as win / equal / loss counts, with the
  direction decided by the mean when the test is significant (Tables 7 and 8);
* Friedman test on the function-by-algorithm matrix of mean values, with average ranks
  (Tables 6, 9 and 14);
* Kruskal-Wallis test per function across algorithms (Tables 12-13, Figs. 4 and 10).
"""

from __future__ import annotations

import csv
import gzip
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class RunTable:
    """Final values of an experiment: ``values[function][algorithm]`` is a 1-D array of runs."""

    values: dict[int, dict[str, np.ndarray]]
    errors: dict[int, dict[str, np.ndarray]]
    n_evals: dict[int, dict[str, np.ndarray]]
    algorithms: tuple[str, ...]
    functions: tuple[int, ...]
    suite: str
    dim: int


def load_runs(path: str | Path) -> RunTable:
    """Read ``runs.csv`` (or the directory that contains it)."""
    path = Path(path)
    if path.is_dir():
        path = path / "runs.csv"
    values: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    errors: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    evals: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    algorithms: list[str] = []
    suite, dim = "", 0
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            function, algorithm = int(row["function"]), row["algorithm"]
            if algorithm not in algorithms:
                algorithms.append(algorithm)
            values[function][algorithm].append(float(row["best"]))
            errors[function][algorithm].append(float(row["error"]))
            evals[function][algorithm].append(float(row["n_evals"]))
            suite, dim = row["suite"], int(row["dim"])
    if not values:
        raise ValueError(f"no runs in {path}")

    def freeze(table: dict) -> dict[int, dict[str, np.ndarray]]:
        return {f: {a: np.asarray(v) for a, v in by_alg.items()} for f, by_alg in table.items()}

    return RunTable(
        values=freeze(values),
        errors=freeze(errors),
        n_evals=freeze(evals),
        algorithms=tuple(algorithms),
        functions=tuple(sorted(values)),
        suite=suite,
        dim=dim,
    )


def load_curves(path: str | Path) -> dict[int, dict[str, np.ndarray]]:
    """Read ``curves.csv.gz``: ``curves[function][algorithm]`` has shape ``(runs, points)``."""
    path = Path(path)
    if path.is_dir():
        path = path / "curves.csv.gz"
    curves: dict[int, dict[str, list[list[float]]]] = defaultdict(lambda: defaultdict(list))
    with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        first = header.index("p0")
        function_col, algorithm_col = header.index("function"), header.index("algorithm")
        for row in reader:
            curves[int(row[function_col])][row[algorithm_col]].append(
                [float(v) for v in row[first:]]
            )
    return {f: {a: np.asarray(v) for a, v in by_alg.items()} for f, by_alg in curves.items()}


def _select(table: RunTable, algorithms: Sequence[str] | None) -> tuple[str, ...]:
    if algorithms is None:
        return table.algorithms
    missing = [a for a in algorithms if a not in table.algorithms]
    if missing:
        raise ValueError(f"algorithms not in the results: {missing}")
    return tuple(algorithms)


def mean_matrix(table: RunTable, algorithms: Sequence[str] | None = None) -> np.ndarray:
    """Matrix of mean final values, shape ``(n_functions, n_algorithms)``."""
    algorithms = _select(table, algorithms)
    return np.array(
        [[table.values[f][a].mean() for a in algorithms] for f in table.functions]
    )


def summary_rows(table: RunTable, algorithms: Sequence[str] | None = None) -> list[dict]:
    """Best / mean / std / median / worst and rank of the mean per function and algorithm.

    Ranks use the competition convention of ``matlab/rank_mean_23.m``: equal means share the
    smallest rank.
    """
    algorithms = _select(table, algorithms)
    rows = []
    for function in table.functions:
        means = np.array([table.values[function][a].mean() for a in algorithms])
        ranks = stats.rankdata(means, method="min")
        for algorithm, rank in zip(algorithms, ranks, strict=True):
            data = table.values[function][algorithm]
            rows.append({
                "function": function,
                "algorithm": algorithm,
                "runs": int(data.size),
                "best": float(data.min()),
                "mean": float(data.mean()),
                "std": float(data.std(ddof=1)) if data.size > 1 else 0.0,
                "median": float(np.median(data)),
                "worst": float(data.max()),
                "mean_error": float(table.errors[function][algorithm].mean()),
                "mean_evals": float(table.n_evals[function][algorithm].mean()),
                "rank": int(rank),
            })
    return rows


def ranksum_p(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sided Wilcoxon rank-sum (Mann-Whitney U) p-value; 1.0 when all values coincide."""
    if np.all(a == a[0]) and np.all(b == a[0]):
        return 1.0
    return float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)


@dataclass(frozen=True)
class WinEqualLoss:
    """Counts of functions on which ``algorithm`` is better / not different / worse."""

    algorithm: str
    baseline: str
    wins: int
    equals: int
    losses: int
    p_values: dict[int, float]
    signs: dict[int, str]

    def __str__(self) -> str:
        return f"{self.wins}/{self.equals}/{self.losses}"


def win_equal_loss(
    table: RunTable, algorithm: str, baseline: str, alpha: float = 0.05
) -> WinEqualLoss:
    """Wilcoxon rank-sum comparison of ``algorithm`` against ``baseline`` over all functions.

    A function counts as a win (loss) when the test is significant at level ``alpha`` and the
    mean of ``algorithm`` is smaller (larger) than that of ``baseline``; otherwise as equal.
    """
    _select(table, (algorithm, baseline))
    wins = equals = losses = 0
    p_values, signs = {}, {}
    for function in table.functions:
        a, b = table.values[function][algorithm], table.values[function][baseline]
        p = ranksum_p(a, b)
        p_values[function] = p
        if p < alpha and a.mean() < b.mean():
            wins, signs[function] = wins + 1, "+"
        elif p < alpha and a.mean() > b.mean():
            losses, signs[function] = losses + 1, "-"
        else:
            equals, signs[function] = equals + 1, "="
    return WinEqualLoss(algorithm, baseline, wins, equals, losses, p_values, signs)


@dataclass(frozen=True)
class RankTest:
    """Average ranks per algorithm and the p-value of the omnibus test."""

    algorithms: tuple[str, ...]
    mean_ranks: tuple[float, ...]
    statistic: float
    p_value: float

    def as_dict(self) -> dict[str, float]:
        return dict(zip(self.algorithms, self.mean_ranks, strict=True))


def friedman_test(table: RunTable, algorithms: Sequence[str] | None = None) -> RankTest:
    """Friedman test on the mean values (functions are blocks, algorithms are treatments).

    Average ranks use mid-ranks for ties (rank 1 is the smallest mean). The p-value needs at
    least three algorithms; with two it is reported as NaN.
    """
    algorithms = _select(table, algorithms)
    means = mean_matrix(table, algorithms)
    ranks = stats.rankdata(means, axis=1, method="average")
    if len(algorithms) >= 3 and len(table.functions) >= 2:
        statistic, p_value = stats.friedmanchisquare(*means.T)
    else:
        statistic, p_value = float("nan"), float("nan")
    return RankTest(
        algorithms, tuple(float(r) for r in ranks.mean(axis=0)), float(statistic), float(p_value)
    )


def kruskal_test(table: RunTable, algorithms: Sequence[str] | None = None) -> dict[int, RankTest]:
    """Kruskal-Wallis test per function across algorithms, with the mean rank of each group."""
    algorithms = _select(table, algorithms)
    out = {}
    for function in table.functions:
        groups = [table.values[function][a] for a in algorithms]
        pooled = stats.rankdata(np.concatenate(groups))
        bounds = np.cumsum([0, *(g.size for g in groups)])
        mean_ranks = tuple(
            float(pooled[bounds[i]:bounds[i + 1]].mean()) for i in range(len(groups))
        )
        if all(np.all(g == groups[0][0]) for g in groups):
            statistic, p_value = 0.0, 1.0
        else:
            statistic, p_value = stats.kruskal(*groups)
        out[function] = RankTest(algorithms, mean_ranks, float(statistic), float(p_value))
    return out


# ----------------------------------------------------------------------------------------
# Writers
# ----------------------------------------------------------------------------------------


def write_csv(path: str | Path, rows: Iterable[dict]) -> None:
    """Write dictionaries with identical keys to a CSV file."""
    rows = list(rows)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(header: Sequence[str], rows: Iterable[Sequence[object]]) -> str:
    """Render a GitHub-flavoured Markdown table."""
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    return "\n".join(lines) + "\n"


def summary_markdown(table: RunTable, algorithms: Sequence[str] | None = None) -> str:
    """Mean (std) per function and algorithm, plus the Friedman mean rank, as Markdown."""
    algorithms = _select(table, algorithms)
    by_key = {(r["function"], r["algorithm"]): r for r in summary_rows(table, algorithms)}
    rows = []
    for function in table.functions:
        cells = [f"F{function}"]
        best = min(by_key[(function, a)]["mean"] for a in algorithms)
        for algorithm in algorithms:
            row = by_key[(function, algorithm)]
            text = f"{row['mean']:.4e} ({row['std']:.2e})"
            cells.append(f"**{text}**" if row["mean"] == best else text)
        rows.append(cells)
    friedman = friedman_test(table, algorithms)
    rows.append(["Friedman mean rank", *(f"{r:.2f}" for r in friedman.mean_ranks)])
    return markdown_table(["Function", *algorithms], rows)


def analyze(results_dir: str | Path, baseline: str | None = None, alpha: float = 0.05) -> dict:
    """Compute all statistics of an experiment and write them next to ``runs.csv``.

    Files written: ``summary.csv`` (best/mean/std/rank), ``summary.md``, ``wilcoxon.csv``
    (per-function p-values and signs against ``baseline``), ``friedman.csv`` and
    ``kruskal.csv``.

    Args:
        results_dir: directory containing ``runs.csv``.
        baseline: algorithm every other algorithm is compared against with the rank-sum test
            (default: the first algorithm of the experiment).
        alpha: significance level.

    Returns:
        A dictionary with the Friedman result and the win/equal/loss strings.
    """
    results_dir = Path(results_dir)
    table = load_runs(results_dir)
    baseline = baseline or table.algorithms[0]
    write_csv(results_dir / "summary.csv", summary_rows(table))
    (results_dir / "summary.md").write_text(summary_markdown(table), encoding="utf-8")

    others = [a for a in table.algorithms if a != baseline]
    comparisons = [win_equal_loss(table, a, baseline, alpha) for a in others]
    if comparisons:
        write_csv(
            results_dir / "wilcoxon.csv",
            (
                {"function": f, "algorithm": c.algorithm, "baseline": baseline,
                 "p_value": c.p_values[f], "sign": c.signs[f]}
                for c in comparisons for f in table.functions
            ),
        )
    friedman = friedman_test(table)
    write_csv(
        results_dir / "friedman.csv",
        (
            {"algorithm": a, "mean_rank": r, "statistic": friedman.statistic,
             "p_value": friedman.p_value}
            for a, r in friedman.as_dict().items()
        ),
    )
    kruskal = kruskal_test(table)
    write_csv(
        results_dir / "kruskal.csv",
        (
            {"function": f, "algorithm": a, "mean_rank": r, "statistic": test.statistic,
             "p_value": test.p_value}
            for f, test in kruskal.items() for a, r in test.as_dict().items()
        ),
    )
    return {
        "suite": table.suite,
        "dim": table.dim,
        "baseline": baseline,
        "friedman": friedman,
        "win_equal_loss": {c.algorithm: str(c) for c in comparisons},
    }
