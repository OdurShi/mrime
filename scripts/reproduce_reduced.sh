#!/usr/bin/env bash
# Reduced-scale reproduction: the four experiments reported in the README section
# "Verified in this repository". Needs a C++ compiler for the native CEC backend; run from the
# repository root:
#   bash scripts/reproduce_reduced.sh [python-interpreter]
# About 8 minutes of wall-clock time with 4 worker processes on the laptop used for the README
# (measured while other jobs were running on the same machine).
set -euo pipefail
PY="${1:-python}"
OUT=results/reduced

"$PY" scripts/verify_cec_backends.py --output "$OUT/cec_backend_check.md" > /dev/null

# A, B: CEC2022 at D = 10 and 20, all eight variants of Table 5, 51 runs.
# C: CEC2017 at D = 10, RIME against MRIME-CD, 15 runs.
for name in cec2022_d10 cec2022_d20 cec2017_d10; do
  "$PY" -m mrime run --config "configs/reduced/$name.yaml" --baseline rime --output "$OUT/$name" > /dev/null
  "$PY" scripts/summarize_results.py "$OUT/$name" --baseline rime
  "$PY" -m mrime plot "$OUT/$name" > /dev/null
done

# D: MATLAB profile and the evaluation counter of matlab/MRIME_RL_Cov.m as shipped, 10 runs.
name=cec2022_d10_matlab
"$PY" -m mrime run --config "configs/reduced/$name.yaml" --baseline rime@matlab --output "$OUT/$name" > /dev/null
"$PY" scripts/summarize_results.py "$OUT/$name" --baseline rime@matlab
"$PY" -m mrime plot "$OUT/$name" > /dev/null
