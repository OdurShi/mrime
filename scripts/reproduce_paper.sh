#!/usr/bin/env bash
# Full protocol of the paper for the algorithms available in Python (RIME, MRIME-CD and the
# six ablation variants): 51 runs, FEsmax = 3000 x D, CEC2017 at D = 10/30/50/100 and CEC2022
# at D = 10/20. One CEC2017 D = 100 run takes 2-10 s on a 2025 laptop CPU; the whole script
# is estimated at roughly 40 CPU-hours.
#   bash scripts/reproduce_paper.sh [python-interpreter] [workers]
set -euo pipefail
PY="${1:-python}"
WORKERS="${2:-4}"
OUT=results/paper

for dim in 10 30 50 100; do
  for kind in comparison ablation; do
    "$PY" -m mrime run --config "configs/paper/cec2017_$kind.yaml" --dim "$dim" \
      --workers "$WORKERS" --baseline rime --output "$OUT/cec2017_${kind}_D$dim"
  done
done
for dim in 10 20; do
  for kind in comparison ablation; do
    "$PY" -m mrime run --config "configs/paper/cec2022_$kind.yaml" --dim "$dim" \
      --workers "$WORKERS" --baseline rime --output "$OUT/cec2022_${kind}_D$dim"
  done
done
