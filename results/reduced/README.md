# Outputs of the reduced-scale configurations

Generated with the Python package (`mrime run`, native CEC backend, base seed 2025) from the
configurations in `configs/reduced/`; `scripts/reproduce_reduced.sh` chains the commands.

| Folder | Config | Suite, D | Algorithms | Runs | `FEsmax` / `NP` |
|---|---|---|---|---|---|
| `cec2022_d10/` | `configs/reduced/cec2022_d10.yaml` | CEC2022, 10 | RIME, the six variants of Table 5, MRIME-CD | 51 | 30000 / 140 |
| `cec2022_d20/` | `configs/reduced/cec2022_d20.yaml` | CEC2022, 20 | RIME, the six variants of Table 5, MRIME-CD | 51 | 60000 / 222 |
| `cec2017_d10/` | `configs/reduced/cec2017_d10.yaml` | CEC2017, 10 | RIME, MRIME-CD | 15 | 30000 / 70 |
| `cec2022_d10_matlab/` | `configs/reduced/cec2022_d10_matlab.yaml` | CEC2022, 10 | RIME and MRIME-CD with the `matlab` profile; MRIME-CD with the `matlab-as-shipped` profile | 10 | 30000 / 140 |

Files in every folder:

| File | Content |
|---|---|
| `runs.csv` | one row per run: seed, final value, error, number of objective evaluations, timing |
| `curves.csv.gz` | best-so-far value of every run on a 51-point grid of the evaluation counter |
| `meta.json` | resolved configuration, software versions, wall-clock time |
| `report.md` | protocol, mean (standard deviation) per function, Wilcoxon rank-sum and Friedman tests |
| `summary.csv`, `summary.md` | best / mean / std / median / worst and rank per function and algorithm |
| `wilcoxon.csv`, `friedman.csv`, `kruskal.csv` | the statistical tests |
| `convergence.png`, `boxplots.png`, `friedman.png` | figures made by `mrime plot` |

`cec_backend_check.md` is the output of `python scripts/verify_cec_backends.py`. The profiles and
the record of these runs are described in `docs/implementation_notes.md`.
