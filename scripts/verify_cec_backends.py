#!/usr/bin/env python
"""Check the CEC2017 / CEC2022 backends against the reference C code in ``matlab/``.

Prints (and optionally writes) a Markdown report with
1. the largest ``|f(x*) - f*|`` of the native backend per suite and dimension, and
2. a function-by-function comparison of the optional ``opfunu`` backend with the native one.

Example::

    python scripts/verify_cec_backends.py --output results/reduced/cec_backend_check.md
"""

import argparse
from pathlib import Path

from mrime.benchmarks.verify import verify_backends


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--points", type=int, default=5, help="random points per function")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, help="also write the report to this file")
    args = parser.parse_args()
    report = verify_backends(points=args.points, seed=args.seed)
    print(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
