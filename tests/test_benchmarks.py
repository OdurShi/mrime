"""Benchmark problems: built-in functions and the CEC suites."""

import importlib.util
import shutil

import numpy as np
import pytest

from mrime.benchmarks import (
    CLASSICAL_NAMES,
    cec_f_opt,
    classical_problem,
    get_problem,
    suite_functions,
)
from mrime.benchmarks.cec import CEC2022_F_OPT, cec2017_official_number
from mrime.benchmarks.native import NativeBackendError, default_source_dir


def _native_available() -> bool:
    try:
        default_source_dir()
    except NativeBackendError:
        return False
    return any(shutil.which(name) for name in ("c++", "g++", "clang++"))


needs_native = pytest.mark.skipif(
    not _native_available(), reason="needs the reference C sources in matlab/ and a C++ compiler"
)
needs_opfunu = pytest.mark.skipif(
    importlib.util.find_spec("opfunu") is None or importlib.util.find_spec("pkg_resources") is None,
    reason="needs the optional opfunu dependency",
)


@pytest.mark.parametrize("name", CLASSICAL_NAMES)
@pytest.mark.parametrize("dim", [2, 7])
def test_classical_optimum_and_batch_consistency(name, dim):
    problem = classical_problem(name, dim)
    assert problem.dim == dim and problem.lower.shape == (dim,) and problem.upper.shape == (dim,)
    assert np.all(problem.x_opt >= problem.lower) and np.all(problem.x_opt <= problem.upper)
    assert problem(problem.x_opt) == pytest.approx(0.0, abs=1e-12)
    rng = np.random.default_rng(0)
    points = problem.lower + rng.random((20, dim)) * (problem.upper - problem.lower)
    batch = problem.evaluate_batch(points)
    assert batch.shape == (20,)
    assert np.all(batch > 0.0)  # the optimum is the unique minimum of every function
    assert np.allclose(batch, [problem(p) for p in points])


def test_classical_reference_values():
    assert classical_problem("sphere", 3)(np.array([1.0, 2.0, 3.0])) == pytest.approx(14.0)
    assert classical_problem("rosenbrock", 2)(np.array([0.0, 0.0])) == pytest.approx(1.0)
    assert classical_problem("rastrigin", 2)(np.array([0.5, 0.0])) == pytest.approx(20.25)
    assert classical_problem("zakharov", 2)(np.array([1.0, 1.0])) == pytest.approx(
        2.0 + 1.5**2 + 1.5**4
    )
    assert classical_problem("schwefel_2_22", 2)(np.array([-2.0, 3.0])) == pytest.approx(11.0)


def test_suite_registry():
    assert suite_functions("classical") == list(range(1, 9))
    assert suite_functions("cec2017") == list(range(1, 30))
    assert suite_functions("cec2022") == list(range(1, 13))
    assert get_problem("classical", 1, 4).name == "classical-sphere-D4"
    assert get_problem("classical", "ellipsoid", 4).info["function"] == 8
    with pytest.raises(ValueError):
        suite_functions("cec2005")
    with pytest.raises(ValueError):
        get_problem("classical", 9, 4)
    with pytest.raises(ValueError):
        get_problem("classical", "nope", 4)
    with pytest.raises(ValueError):
        classical_problem("sphere", 1)


def test_cec_numbering_and_optimum_values_follow_the_paper_tables():
    # Table 3: F1 -> 100, F2 -> 300, ..., F29 -> 3000 (the official F2 was withdrawn)
    assert cec2017_official_number(1) == 1
    assert [cec2017_official_number(k) for k in (2, 9, 10, 29)] == [3, 10, 11, 30]
    assert cec_f_opt("cec2017", 1) == 100.0
    for k in range(2, 30):
        assert cec_f_opt("cec2017", k) == 100.0 * (k + 1)
    # Table 4
    assert CEC2022_F_OPT == (300, 400, 600, 800, 900, 1800, 2000, 2200, 2300, 2400, 2600, 2700)
    with pytest.raises(ValueError):
        cec_f_opt("cec2017", 30)
    with pytest.raises(ValueError):
        cec_f_opt("cec2022", 0)
    with pytest.raises(ValueError):
        get_problem("cec2022", 1, 10, backend="other")


@needs_native
@pytest.mark.parametrize(("suite", "dim"), [("cec2017", 10), ("cec2017", 30), ("cec2022", 10),
                                            ("cec2022", 20)])
def test_native_backend_reproduces_the_documented_optimum_values(suite, dim):
    for function in suite_functions(suite):
        problem = get_problem(suite, function, dim)
        assert problem.info["backend"] == "native"
        assert problem(problem.x_opt) == pytest.approx(problem.f_opt, abs=1e-8)
        assert np.all(problem.lower == -100.0) and np.all(problem.upper == 100.0)


@needs_native
def test_native_backend_batch_equals_single_and_is_above_the_optimum():
    rng = np.random.default_rng(1)
    points = rng.uniform(-100, 100, size=(16, 10))
    for suite, function in (("cec2017", 1), ("cec2017", 12), ("cec2017", 25), ("cec2022", 7)):
        problem = get_problem(suite, function, 10)
        batch = problem.evaluate_batch(points)
        assert np.array_equal(batch, [problem(p) for p in points])
        assert np.all(batch > problem.f_opt)


@needs_native
def test_native_backend_interleaves_functions_correctly():
    # the reference code keeps one (function, dimension) loaded at a time
    a, b = get_problem("cec2022", 1, 10), get_problem("cec2022", 5, 20)
    xa, xb = np.full(10, 3.0), np.full(20, -7.0)
    first = (a(xa), b(xb))
    assert (a(xa), b(xb)) == first and (b(xb), a(xa)) == first[::-1]


@needs_native
def test_native_backend_rejects_unsupported_dimensions_and_matlab_bounds():
    with pytest.raises(ValueError, match="not defined for D=7"):
        get_problem("cec2022", 1, 7)
    with pytest.raises(ValueError, match="not defined for D=20"):
        get_problem("cec2017", 12, 20)  # hybrid functions have no D=20 data
    assert get_problem("cec2022", 11, 10).upper[0] == 100.0  # paper Table 4
    assert get_problem("cec2022", 11, 10, matlab_bounds=True).upper[0] == 600.0
    assert get_problem("cec2022", 10, 10, matlab_bounds=True).upper[0] == 100.0


@needs_opfunu
@pytest.mark.parametrize(("suite", "dim"), [("cec2017", 10), ("cec2022", 10)])
def test_opfunu_backend_is_rebased_to_the_official_optimum_values(suite, dim):
    for function in suite_functions(suite):
        problem = get_problem(suite, function, dim, backend="opfunu")
        assert problem.info["backend"] == "opfunu"
        assert problem.f_opt == cec_f_opt(suite, function)
        assert problem(problem.x_opt) == pytest.approx(problem.f_opt, abs=1e-6)


@needs_native
@needs_opfunu
def test_opfunu_agrees_with_the_reference_code_only_on_a_few_functions():
    """Documents the finding recorded in docs/implementation_notes.md."""
    rng = np.random.default_rng(2)
    points = rng.uniform(-100, 100, size=(4, 10))

    def identical(suite: str, function: int) -> bool:
        native = get_problem(suite, function, 10).evaluate_batch(points)
        other = get_problem(suite, function, 10, backend="opfunu").evaluate_batch(points)
        return bool(np.allclose(native, other, rtol=1e-9))

    assert identical("cec2017", 1) and identical("cec2022", 2)
    assert not identical("cec2022", 1)  # opfunu's Zakharov differs from the reference code
    assert not identical("cec2017", 2)


@needs_native
def test_data_directory_can_be_set_by_environment_variable(tmp_path, monkeypatch):
    """A copy of the CEC2022 sources and data in another folder is found through the variable."""
    import shutil as sh

    from mrime.benchmarks.native import load_suite

    source = default_source_dir()
    assert source.name == "matlab"  # default: the folder shipped with the repository
    copy = tmp_path / "cec"
    copy.mkdir()
    sh.copy(source / "cec22_func.cpp", copy)
    sh.copy(source / "cec17_func.cpp", copy)
    sh.copytree(source / "input_data22", copy / "input_data22")
    monkeypatch.setenv("MRIME_CEC_SOURCE_DIR", str(copy))
    monkeypatch.setenv("MRIME_NATIVE_BUILD_DIR", str(tmp_path / "build"))
    assert default_source_dir() == copy.resolve()
    library = load_suite("cec2022")
    assert library.data_dir == copy.resolve() / "input_data22"
    problem = get_problem("cec2022", 3, 10)
    assert problem(problem.x_opt) == pytest.approx(600.0, abs=1e-9)
    monkeypatch.delenv("MRIME_CEC_SOURCE_DIR")
    assert load_suite("cec2022").data_dir == source / "input_data22"
