# Data

The experiments of the paper use no external dataset. Their inputs are two public benchmark
suites of test functions, which are defined by reference C code plus data files (shift vectors,
rotation matrices and shuffle indices). Both are shipped with the repository, in the location
the MATLAB experiment scripts expect, so a clone runs without any download:

| Suite | Used for | Reference code | Definition data | Files | Size |
|---|---|---|---|---|---|
| CEC2017 single-objective bound-constrained suite | Sect. 4.2 and 4.3 of the paper (Tables 6-9, 17-20; Figs. 3-8, 13-16), D = 10/30/50/100 | `matlab/cec17_func.cpp` | `matlab/input_data/` | 329 | 41.9 MB |
| CEC2022 single-objective bound-constrained suite | Sect. 4.2 and 4.4 of the paper (Tables 6-7, 10-14; Figs. 9-12, 17-18), D = 10/20 | `matlab/cec22_func.cpp` | `matlab/input_data22/` | 54 | 0.77 MB |

The built-in classical functions of the Python package (`src/mrime/benchmarks/classical.py`)
need no data; the rotation and shift of its `ellipsoid` problem are generated in code from a
fixed seed (`ellipsoid_parameters`).

## How the code finds the data

* MATLAB: the MEX functions built from the two `.cpp` files read `input_data/` and
  `input_data22/` relative to the current folder, i.e. run the scripts from `matlab/`.
* Python: `mrime.benchmarks.native` compiles the same two sources and reads the same folders.
  The folder is resolved in this order: the `--data-dir` option of `mrime run` and
  `mrime verify-cec`; the environment variable `MRIME_CEC_SOURCE_DIR`; the `matlab/` folder of
  the source checkout (default). Nothing is downloaded.

## CEC2017

* **What it is.** 30 shifted / rotated / hybrid / composition test functions with search range
  `[-100, 100]^D`. The official function F2 was withdrawn by the organisers; the paper uses the
  remaining 29 and numbers them F1-F29 (paper `Fk` is official function `k + 1` for `k >= 2`).
  Optimum values are `100 x (official number)`.
* **Source.** Competition material published by the organisers at
  https://github.com/P-N-Suganthan/CEC2017-BoundContrained . Definition: N. H. Awad, M. Z. Ali,
  J. J. Liang, B. Y. Qu, P. N. Suganthan, "Problem Definitions and Evaluation Criteria for the
  CEC 2017 Special Session and Competition on Single Objective Real-Parameter Numerical
  Optimization", Technical Report, Nanyang Technological University, Singapore, 2016.
* **Version.** `cec17_func.cpp` carries the header "CEC17 Test Function Suite for Single
  Objective Optimization - Bound constraints, Noor Awad, Sep. 10th 2016, Last update on
  26_5_2017". Files are those of the authors' experiment archive, unchanged.
* **Licence.** Neither the files nor the organisers' repository state a licence (checked on
  2026-11-13). The material is the publicly distributed competition code and data; it is
  redistributed here unchanged with the attribution above and is not covered by this
  repository's MIT licence (see `THIRD_PARTY_NOTICES.md`).
* **Layout of `matlab/input_data/`.**

  | Files | Count | Content |
  |---|---|---|
  | `shift_data_<k>.txt` | 30 | shift vector(s) of official function `k` (one row of 100 values; ten rows for the composition functions) |
  | `M_<k>_D<d>.txt` | 159 | rotation matrix (matrices) of function `k` in dimension `d`: 30 files each for d = 10, 30, 50, 100; 20 for d = 2; 19 for d = 20 |
  | `shuffle_data_<k>_D<d>.txt` | 139 | variable permutation of function `k` (used by the hybrid functions 11-20 and the composition functions 29-30): 30 files each for d = 10, 30, 50, 100; 19 for d = 20 |
  | `Rand_Seeds.txt` | 1 | 1000 seeds; the MATLAB run scripts draw the seed of each run from this list |

  The dimensions used in the paper (10, 30, 50, 100) are complete for all 29 functions.

## CEC2022

* **What it is.** 12 test functions (1 unimodal, 4 basic, 3 hybrid, 4 composition) with search
  range `[-100, 100]^D` and optimum values 300, 400, 600, 800, 900, 1800, 2000, 2200, 2300,
  2400, 2600, 2700 (Table 4 of the paper).
* **Source.** Competition material published by the organisers at
  https://github.com/P-N-Suganthan/2022-SO-BO . Definition: A. Kumar, K. V. Price,
  A. W. Mohamed, A. A. Hadi, P. N. Suganthan, "Problem Definitions and Evaluation Criteria for
  the CEC 2022 Special Session and Competition on Single Objective Bound Constrained Numerical
  Optimization", Technical Report, Nanyang Technological University, Singapore, 2021.
* **Version.** `cec22_func.cpp` carries the header "CEC22 Test Function Suite for Single
  Objective Bound Constrained Numerical Optimization" without a date. Files are those of the
  authors' experiment archive, unchanged.
* **Licence.** As for CEC2017: no licence statement in the files or in the organisers'
  repository (checked on 2026-11-13); redistributed unchanged with attribution.
* **Layout of `matlab/input_data22/`.**

  | Files | Count | Content |
  |---|---|---|
  | `shift_data_<k>.txt` | 12 | shift vector(s) of function `k` |
  | `M_<k>_D<d>.txt` | 35 | rotation matrix (matrices): 12 files each for d = 10 and 20; 11 for d = 2 |
  | `shuffle_data_<k>_D<d>.txt` | 6 | variable permutation of the hybrid functions 6-8 for d = 10 and 20 |
  | `Rand_Seeds.txt` | 1 | 1000 seeds |

## Integrity

`data/SHA256SUMS` lists the SHA-256 of every shipped file (385 lines: 329 + 54 data files and
the two C sources) in `sha256sum` format with paths relative to the repository root:

```bash
shasum -a 256 -c data/SHA256SUMS            # or: sha256sum -c data/SHA256SUMS
python scripts/data_checksums.py --check    # same check; --write regenerates the manifest
```

| Item | SHA-256 |
|---|---|
| `data/SHA256SUMS` (the manifest itself) | `5523d1ca97b726a666af68e848420c5e61bb1858ecaa16cce00415294ded2378` |
| `matlab/cec17_func.cpp` | `fcd96a87a473dae6784e1ed85410546e9848c56d9fe6cddf0b7183cc72588b1b` |
| `matlab/cec22_func.cpp` | `4264547ffd3548259f5de2a9a645c373086f830565fa675fd68849acceb829cb` |

The hashes are those of the files as committed; a checkout that converts line endings changes
them (the function values are unaffected). The test suite checks the manifest against the
files (`tests/test_assets.py`).

## Other data in the repository

`results/reduced/` holds outputs of the Python package (per-run results, convergence curves,
tables, figures) for the configurations in `configs/reduced/`, generated with base seed 2025;
see the `report.md` and `meta.json` in each folder.
