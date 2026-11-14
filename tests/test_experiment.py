"""Experiment runner, statistics, figures and the command-line interface (end to end)."""

import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from mrime import analysis
from mrime.cli import main
from mrime.experiment import ExperimentConfig, load_config, run_experiment, run_seed

REPO = Path(__file__).resolve().parents[1]
TINY = {
    "name": "tiny", "suite": "classical", "dim": 5, "algorithms": ("rime", "mrime-cd"),
    "runs": 3, "functions": (1, 8), "max_evals": 600, "pop_size": 12, "seed": 7,
    "curve_points": 11,
}


def _read_runs(directory: Path) -> list[dict]:
    with open(directory / "runs.csv", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture(scope="module")
def tiny_results(tmp_path_factory) -> Path:
    return run_experiment(ExperimentConfig(**TINY), tmp_path_factory.mktemp("tiny"))


def test_experiment_writes_raw_runs_curves_and_meta(tiny_results):
    rows = _read_runs(tiny_results)
    assert len(rows) == 2 * 2 * 3
    assert {row["algorithm"] for row in rows} == {"rime", "mrime-cd"}
    assert all(int(row["n_evals"]) == 600 == int(row["max_evals"]) for row in rows)
    assert all(float(row["error"]) == float(row["best"]) for row in rows)  # f* = 0
    # every algorithm starts run r of a function from the same seed
    for function in ("1", "8"):
        for run in ("0", "1", "2"):
            seeds = {r["seed"] for r in rows if r["function"] == function and r["run"] == run}
            assert len(seeds) == 1
    assert len({row["seed"] for row in rows}) == 6
    curves = analysis.load_curves(tiny_results)
    assert curves[1]["rime"].shape == (3, 11)
    assert np.all(np.diff(curves[8]["mrime-cd"], axis=1) <= 0)
    finals = analysis.load_runs(tiny_results).values
    assert np.allclose(curves[1]["rime"][:, -1], finals[1]["rime"], rtol=1e-9)
    meta = json.loads((tiny_results / "meta.json").read_text())
    assert meta["config"]["resolved_max_evals"] == 600 and meta["n_runs_total"] == 12
    assert "/Users/" not in json.dumps(meta) and "/home/" not in json.dumps(meta)


def test_parallel_workers_reproduce_the_serial_results(tiny_results, tmp_path):
    parallel = run_experiment(ExperimentConfig(**{**TINY, "workers": 2}), tmp_path)
    serial_rows, parallel_rows = _read_runs(tiny_results), _read_runs(parallel)
    assert [r["best"] for r in serial_rows] == [r["best"] for r in parallel_rows]
    assert [r["seed"] for r in serial_rows] == [r["seed"] for r in parallel_rows]


def test_run_seed_is_stable_and_algorithm_independent():
    assert run_seed(1, "cec2022", 10, 3, 0) == run_seed(1, "cec2022", 10, 3, 0)
    values = {run_seed(1, "cec2022", 10, f, r) for f in range(1, 13) for r in range(51)}
    assert len(values) == 12 * 51
    assert run_seed(1, "cec2022", 10, 3, 0) != run_seed(2, "cec2022", 10, 3, 0)


def test_config_resolution_and_validation():
    config = ExperimentConfig(name="x", suite="cec2022", dim=10, algorithms=("rime",), runs=1)
    assert config.budget == 30_000  # Table 2: 3000 x Dim
    assert config.population == 70  # ceil(15 * 10^(2/3))
    assert config.function_ids == tuple(range(1, 13))
    wide = ExperimentConfig(name="x", suite="cec2017", dim=100, algorithms=("rime",), runs=1,
                            pop_coefficient=30)
    assert wide.population == math.ceil(30 * 100 ** (2 / 3)) == 647
    with pytest.raises(ValueError):
        ExperimentConfig(name="x", suite="cec2030", dim=10, algorithms=("rime",), runs=1)
    with pytest.raises(ValueError):
        ExperimentConfig(name="x", suite="cec2022", dim=10, algorithms=("pso",), runs=1)
    with pytest.raises(ValueError):
        ExperimentConfig(name="x", suite="cec2022", dim=10, algorithms=("rime",), runs=1,
                         functions=(13,))
    with pytest.raises(ValueError):
        ExperimentConfig(name="x", suite="cec2022", dim=10, algorithms=(), runs=1)


@pytest.mark.parametrize("path", sorted((REPO / "configs").rglob("*.yaml")), ids=lambda p: p.stem)
def test_every_shipped_config_loads_with_paper_protocol_values(path):
    config = load_config(path)
    assert config.algorithms and config.runs >= 1
    if config.suite != "classical":
        assert config.evals_per_dim == 3000 and config.budget == 3000 * config.dim  # Table 2
        assert config.runs == 51 or path.parent.name == "reduced"  # Table 2: 51 runs
        assert config.pop_coefficient in (15, 30) and config.backend == "native"


def test_load_config_overrides_and_unknown_keys(tmp_path):
    config = load_config(REPO / "configs" / "quick_classical.yaml", runs=2, dim=None)
    assert config.runs == 2 and config.dim == 10
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: x\nsuite: classical\ndim: 5\nalgorithms: [rime]\nruns: 1\nbudget: 3\n")
    with pytest.raises(ValueError, match="unknown keys"):
        load_config(bad)


# ------------------------------------------------------------------ statistics


def _table(values: dict[int, dict[str, list[float]]]) -> analysis.RunTable:
    arrays = {
        f: {a: np.asarray(v, dtype=float) for a, v in row.items()} for f, row in values.items()
    }
    algorithms = tuple(next(iter(values.values())))
    functions = tuple(sorted(values))
    return analysis.RunTable(arrays, arrays, arrays, algorithms, functions, "synthetic", 2)


def test_summary_rows_and_ranks():
    table = _table({1: {"a": [1.0, 3.0], "b": [2.0, 2.0], "c": [5.0, 7.0]}})
    rows = {r["algorithm"]: r for r in analysis.summary_rows(table)}
    assert rows["a"]["mean"] == 2.0 and rows["a"]["best"] == 1.0 and rows["a"]["worst"] == 3.0
    assert rows["a"]["std"] == pytest.approx(math.sqrt(2.0))  # sample standard deviation
    assert (rows["a"]["rank"], rows["b"]["rank"], rows["c"]["rank"]) == (1, 1, 3)  # tie -> min


def test_win_equal_loss_uses_the_rank_sum_test_and_the_mean():
    low, high = list(np.linspace(0.0, 1.0, 20)), list(np.linspace(10.0, 11.0, 20))
    table = _table({
        1: {"new": low, "base": high},    # significantly better
        2: {"new": high, "base": low},    # significantly worse
        3: {"new": low, "base": low},     # identical samples
        4: {"new": [1.0] * 20, "base": [1.0] * 20},  # all values equal -> p = 1
    })
    result = analysis.win_equal_loss(table, "new", "base")
    assert (result.wins, result.equals, result.losses) == (1, 2, 1)
    assert str(result) == "1/2/1"
    assert result.signs == {1: "+", 2: "-", 3: "=", 4: "="}
    assert result.p_values[1] < 1e-6 and result.p_values[4] == 1.0


def test_friedman_mean_ranks():
    table = _table({
        1: {"a": [1.0], "b": [2.0], "c": [3.0]},
        2: {"a": [1.0], "b": [3.0], "c": [2.0]},
        3: {"a": [2.0], "b": [1.0], "c": [3.0]},
        4: {"a": [1.0], "b": [2.0], "c": [3.0]},
    })
    result = analysis.friedman_test(table)
    assert result.as_dict() == {"a": 1.25, "b": 2.0, "c": 2.75}
    assert 0.0 < result.p_value < 1.0
    two = analysis.friedman_test(table, ["a", "b"])
    assert two.mean_ranks == (1.25, 1.75) and math.isnan(two.p_value)
    with pytest.raises(ValueError):
        analysis.friedman_test(table, ["a", "zzz"])


def test_kruskal_mean_ranks():
    table = _table({1: {"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]}})
    result = analysis.kruskal_test(table)[1]
    assert result.mean_ranks == (2.0, 5.0)
    assert result.p_value == pytest.approx(0.0495, abs=1e-3)


def test_analyze_writes_tables(tiny_results):
    summary = analysis.analyze(tiny_results, baseline="rime")
    for name in ("summary.csv", "summary.md", "wilcoxon.csv", "friedman.csv", "kruskal.csv"):
        assert (tiny_results / name).stat().st_size > 0
    assert set(summary["win_equal_loss"]) == {"mrime-cd"}
    text = (tiny_results / "summary.md").read_text()
    assert text.startswith("| Function | rime | mrime-cd |") and "Friedman mean rank" in text


def test_figures_are_written(tiny_results):
    pytest.importorskip("matplotlib")
    from mrime.plotting import algorithm_label, make_figures

    paths = make_figures(tiny_results)
    assert [p.name for p in paths] == ["convergence.png", "boxplots.png", "friedman.png"]
    assert all(p.stat().st_size > 5_000 for p in paths)
    assert algorithm_label("mrime-cd@matlab") == "MRIME-CD (matlab)"
    assert algorithm_label("rime-ga") == "RIME-GA"


# ------------------------------------------------------------------ command line


def test_cli_demo_and_list(capsys):
    demo = ["demo", "--dim", "5", "--pop-size", "12", "--max-evals", "600", "--runs", "1"]
    assert main(demo) == 0
    out = capsys.readouterr().out
    assert "MRIME-CD" in out and "ellipsoid" in out
    assert main(["list"]) == 0
    out = capsys.readouterr().out
    assert "MRIME_RL_Cov.m" in out and "matlab-as-shipped" in out


def test_cli_run_analyze_plot(tmp_path, capsys):
    out_dir = tmp_path / "cli"
    code = main([
        "run", "--suite", "classical", "--dim", "4", "--algorithms", "rime", "rime-ga@matlab",
        "--functions", "1", "5", "--runs", "2", "--max-evals", "300", "--pop-size", "10",
        "--output", str(out_dir),
    ])
    assert code == 0
    assert "rime-ga@matlab" in capsys.readouterr().out
    assert len(_read_runs(out_dir)) == 8
    assert main(["analyze", str(out_dir), "--baseline", "rime"]) == 0
    pytest.importorskip("matplotlib")
    assert main(["plot", str(out_dir)]) == 0
    assert (out_dir / "convergence.png").is_file()


def test_cli_run_from_config_with_overrides(tmp_path):
    code = main([
        "run", "--config", str(REPO / "configs" / "quick_classical.yaml"), "--runs", "1",
        "--max-evals", "400", "--workers", "1", "--functions", "1", "--output", str(tmp_path),
    ])
    assert code == 0 and len(_read_runs(tmp_path)) == 2


def test_cli_run_requires_a_config_or_the_basic_arguments(capsys):
    assert main(["run", "--suite", "classical"]) == 2
    assert "required" in capsys.readouterr().err


def test_module_entry_point_smoke():
    completed = subprocess.run(
        [sys.executable, "-m", "mrime", "demo", "--dim", "4", "--pop-size", "10",
         "--max-evals", "300", "--runs", "1"],
        capture_output=True, text=True, check=False, timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    assert "MRIME-CD" in completed.stdout


def test_cli_run_accepts_a_data_directory(tmp_path, monkeypatch):
    from mrime.benchmarks.native import NativeBackendError, default_source_dir, find_compiler

    try:
        find_compiler()
        data_dir = default_source_dir()
    except NativeBackendError:
        pytest.skip("needs the reference C sources and a C++ compiler")
    monkeypatch.delenv("MRIME_CEC_SOURCE_DIR", raising=False)
    code = main([
        "run", "--suite", "cec2022", "--dim", "10", "--algorithms", "rime", "--functions", "1",
        "--runs", "1", "--max-evals", "300", "--pop-size", "10", "--data-dir", str(data_dir),
        "--output", str(tmp_path),
    ])
    assert code == 0 and len(_read_runs(tmp_path)) == 1
    monkeypatch.delenv("MRIME_CEC_SOURCE_DIR", raising=False)  # set by --data-dir
