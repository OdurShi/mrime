#!/usr/bin/env python
"""Convergence curves, box plots and Friedman ranks of an experiment (same as ``mrime plot``).

Example::

    python scripts/make_figures.py results/reduced/cec2022_d10 --algorithms rime mrime-cd
"""

import sys

from mrime.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["plot", *sys.argv[1:]]))
