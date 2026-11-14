#!/usr/bin/env python
"""Run one experiment from a YAML config (same as ``mrime run``).

Example::

    python scripts/run_experiment.py --config configs/paper/cec2022_comparison.yaml --dim 20
"""

import sys

from mrime.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["run", *sys.argv[1:]]))
