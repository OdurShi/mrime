#!/usr/bin/env python
"""Mean/std tables and the rank-sum, Friedman and Kruskal-Wallis tests (``mrime analyze``).

Example::

    python scripts/make_tables.py results/reduced/cec2022_d10 --baseline rime
"""

import sys

from mrime.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["analyze", *sys.argv[1:]]))
