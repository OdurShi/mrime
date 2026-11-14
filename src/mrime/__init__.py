"""MRIME-CD: a modified RIME algorithm with covariance learning and diversity enhancement.

Reference implementation of the paper published in Cluster Computing 28:658 (2025),
https://doi.org/10.1007/s10586-025-05348-3.

Quick start::

    import numpy as np
    from mrime import mrime_cd

    result = mrime_cd(lambda x: float(np.sum(x**2)), -100, 100, dim=10,
                      pop_size=70, max_evals=30_000, seed=0)
    print(result.fun, result.n_evals)
"""

from mrime.config import (
    MATLAB,
    MATLAB_AS_SHIPPED,
    PAPER,
    PROFILES,
    VARIANTS,
    MRIMEConfig,
    make_config,
    parse_algorithm,
)
from mrime.core import minimize, mrime_cd, rime, run
from mrime.problem import BudgetExceededError, Evaluator, OptimizeResult

__version__ = "0.1.0"

__all__ = [
    "MATLAB",
    "MATLAB_AS_SHIPPED",
    "PAPER",
    "PROFILES",
    "VARIANTS",
    "BudgetExceededError",
    "Evaluator",
    "MRIMEConfig",
    "OptimizeResult",
    "__version__",
    "make_config",
    "minimize",
    "mrime_cd",
    "parse_algorithm",
    "rime",
    "run",
]
