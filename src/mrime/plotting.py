"""Figures in the style of the paper: convergence curves, box plots, Friedman ranks.

Colour is assigned to the algorithm variant (a fixed mapping, so a variant keeps its colour
in every figure), the line style to the compatibility profile. At most eight variants exist,
one per slot of the categorical palette. Requires matplotlib (``pip install 'mrime[plot]'``).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from mrime.analysis import RunTable, friedman_test, load_curves, load_runs
from mrime.config import parse_algorithm

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"

#: Fixed colour of each variant (categorical palette, assigned to the entity, never to a rank).
VARIANT_COLORS: dict[str, str] = {
    "mrime-cd": "#2a78d6",
    "rime": "#eb6834",
    "rime-g": "#1baf7a",
    "rime-a": "#eda100",
    "rime-s": "#e87ba4",
    "rime-ga": "#008300",
    "rime-gs": "#6250d6",
    "rime-as": "#e34948",
}
PROFILE_LINESTYLES: dict[str, object] = {
    "paper": "-",
    "matlab": (0, (5, 2)),
    "matlab-as-shipped": (0, (1, 2)),
}
VARIANT_LABELS: dict[str, str] = {
    "rime": "RIME", "rime-g": "RIME-G", "rime-a": "RIME-A", "rime-s": "RIME-S",
    "rime-ga": "RIME-GA", "rime-gs": "RIME-GS", "rime-as": "RIME-AS", "mrime-cd": "MRIME-CD",
}


def algorithm_label(spec: str) -> str:
    """Display name of an algorithm spec, e.g. ``"mrime-cd@matlab"`` -> ``"MRIME-CD (matlab)"``."""
    variant, profile = parse_algorithm(spec)
    label = VARIANT_LABELS[variant]
    return label if profile == "paper" else f"{label} ({profile})"


def _style(spec: str) -> dict:
    variant, profile = parse_algorithm(spec)
    return {"color": VARIANT_COLORS[variant], "linestyle": PROFILE_LINESTYLES[profile]}


def _pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise ImportError("plotting needs matplotlib: pip install 'mrime[plot]'") from error
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT_SECONDARY,
        "axes.titlecolor": TEXT_PRIMARY,
        "text.color": TEXT_PRIMARY,
        "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 9,
        "axes.titlesize": 9,
        "legend.frameon": False,
    })
    return plt


def _grid(n_panels: int, max_columns: int = 4) -> tuple[int, int]:
    columns = min(max_columns, n_panels)
    return math.ceil(n_panels / columns), columns


def _figure_title(table: RunTable, what: str) -> str:
    return f"{what} - {table.suite.upper()}, D = {table.dim}"


def plot_convergence(
    results_dir: str | Path,
    output: str | Path,
    algorithms: Sequence[str] | None = None,
    functions: Sequence[int] | None = None,
) -> Path:
    """Mean best-so-far error ``f - f*`` against the evaluation counter, one panel per function.

    The y-axis is logarithmic; errors below ``1e-10`` are drawn at ``1e-10``.
    """
    plt = _pyplot()
    table = load_runs(results_dir)
    curves = load_curves(results_dir)
    algorithms = list(algorithms or table.algorithms)
    functions = list(functions or table.functions)
    rows, columns = _grid(len(functions))
    figure, axes = plt.subplots(
        rows, columns, figsize=(3.1 * columns, 2.5 * rows + 0.9), squeeze=False
    )
    handles = {}
    for axis, function in zip(axes.flat, functions, strict=False):
        offset = table.values[function][algorithms[0]] - table.errors[function][algorithms[0]]
        f_opt = float(offset[0]) if np.all(np.isfinite(offset)) else 0.0
        for spec in algorithms:
            mean_curve = curves[function][spec].mean(axis=0) - f_opt
            x = np.linspace(0.0, 1.0, mean_curve.size)
            (line,) = axis.plot(
                x, np.maximum(mean_curve, 1e-10), linewidth=2.0, solid_capstyle="round",
                label=algorithm_label(spec), **_style(spec),
            )
            handles[spec] = line
        axis.set_yscale("log")
        axis.set_title(f"F{function}")
    for axis in axes.flat[len(functions):]:
        axis.set_visible(False)
    for axis in axes[-1]:
        axis.set_xlabel("evaluation counter / FEsmax")
    for axis in axes[:, 0]:
        axis.set_ylabel("mean error  f - f*")
    # Reserve a fixed band (in inches) above the panels for the title and the shared legend.
    height = figure.get_figheight()
    legend_rows = math.ceil(len(handles) / 4)
    figure.suptitle(_figure_title(table, "Convergence (mean of runs)"), y=1 - 0.10 / height)
    figure.legend(
        handles.values(), [algorithm_label(s) for s in handles], loc="upper center",
        bbox_to_anchor=(0.5, 1 - 0.33 / height), ncol=min(len(handles), 4), handlelength=3.0,
    )
    figure.tight_layout(rect=(0, 0, 1, 1 - (0.36 + 0.22 * legend_rows) / height))
    output = Path(output)
    figure.savefig(output, dpi=150)
    plt.close(figure)
    return output


def plot_boxplots(
    results_dir: str | Path,
    output: str | Path,
    algorithms: Sequence[str] | None = None,
    functions: Sequence[int] | None = None,
) -> Path:
    """Distribution of the final values over the independent runs, one panel per function."""
    plt = _pyplot()
    table = load_runs(results_dir)
    algorithms = list(algorithms or table.algorithms)
    functions = list(functions or table.functions)
    rows, columns = _grid(len(functions))
    width = max(2.3, 0.42 * len(algorithms) + 1.2)
    figure, axes = plt.subplots(
        rows, columns, figsize=(width * columns, 2.7 * rows + 0.5), squeeze=False
    )
    labels = [algorithm_label(s) for s in algorithms]
    for axis, function in zip(axes.flat, functions, strict=False):
        data = [table.values[function][spec] for spec in algorithms]
        boxes = axis.boxplot(
            data, widths=0.38, patch_artist=True, showfliers=True,
            medianprops={"color": TEXT_PRIMARY, "linewidth": 1.2},
            whiskerprops={"color": TEXT_SECONDARY, "linewidth": 0.8},
            capprops={"color": TEXT_SECONDARY, "linewidth": 0.8},
            flierprops={"marker": "o", "markersize": 2.5, "markerfacecolor": TEXT_SECONDARY,
                        "markeredgecolor": "none", "alpha": 0.6},
        )
        for patch, spec in zip(boxes["boxes"], algorithms, strict=True):
            patch.set_facecolor(VARIANT_COLORS[parse_algorithm(spec)[0]])
            patch.set_edgecolor(SURFACE)
            patch.set_linewidth(1.0)
            if parse_algorithm(spec)[1] != "paper":
                patch.set_alpha(0.55)
        axis.set_title(f"F{function}")
        axis.set_xticks(range(1, len(algorithms) + 1))
        axis.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
        axis.grid(axis="x", visible=False)
    for axis in axes.flat[len(functions):]:
        axis.set_visible(False)
    for axis in axes[:, 0]:
        axis.set_ylabel("final value f")
    figure.suptitle(_figure_title(table, "Final values over independent runs"))
    figure.tight_layout()
    output = Path(output)
    figure.savefig(output, dpi=150)
    plt.close(figure)
    return output


def plot_friedman(
    results_dir: str | Path, output: str | Path, algorithms: Sequence[str] | None = None
) -> Path:
    """Friedman mean rank of each algorithm (smaller is better) as horizontal bars."""
    plt = _pyplot()
    table = load_runs(results_dir)
    result = friedman_test(table, algorithms)
    order = np.argsort(result.mean_ranks)
    labels = [algorithm_label(result.algorithms[i]) for i in order]
    ranks = [result.mean_ranks[i] for i in order]
    figure, axis = plt.subplots(figsize=(5.2, 0.36 * len(labels) + 1.3))
    positions = np.arange(len(labels))[::-1]
    axis.barh(positions, ranks, height=0.5, color=VARIANT_COLORS["mrime-cd"])
    for position, rank in zip(positions, ranks, strict=True):
        axis.text(rank + 0.06, position, f"{rank:.2f}", va="center", color=TEXT_PRIMARY)
    axis.set_yticks(positions)
    axis.set_yticklabels(labels)
    axis.set_xlim(0, max(ranks) * 1.15)
    axis.set_xlabel("Friedman mean rank (smaller is better)")
    axis.grid(axis="y", visible=False)
    p_text = "n/a" if math.isnan(result.p_value) else f"{result.p_value:.2e}"
    axis.set_title(f"{_figure_title(table, 'Friedman mean rank')} (p = {p_text})")
    figure.tight_layout()
    output = Path(output)
    figure.savefig(output, dpi=150)
    plt.close(figure)
    return output


def make_figures(
    results_dir: str | Path,
    algorithms: Sequence[str] | None = None,
    functions: Sequence[int] | None = None,
) -> list[Path]:
    """Write ``convergence.png``, ``boxplots.png`` and ``friedman.png`` into ``results_dir``."""
    results_dir = Path(results_dir)
    return [
        plot_convergence(results_dir, results_dir / "convergence.png", algorithms, functions),
        plot_boxplots(results_dir, results_dir / "boxplots.png", algorithms, functions),
        plot_friedman(results_dir, results_dir / "friedman.png", algorithms),
    ]
