# Third-party material in this repository

The MIT licence in `LICENSE` covers the authors' own code only. Everything listed under
"Third-party" below was written by other people, is redistributed here unchanged as part of the
paper's experiment archive, and stays under the terms of its original authors. The attribution
column quotes or summarises what the file header itself states; "none" means the file carries no
attribution header. Unless noted, the files state **no licence**; if you want to reuse one of
them, obtain it from its original source and follow that source's terms.

## First-party (covered by `LICENSE`)

| Path | Content |
|---|---|
| `src/`, `tests/`, `scripts/`, `examples/`, `configs/` | Python reference implementation and tooling |
| `matlab/MRIME_RL_Cov.m` | MRIME-CD as called by the comparison scripts |
| `matlab/MRIME_RL_1.m`, `_2.m`, `_3.m`, `_12.m`, `_13.m`, `_23.m` | ablation variants RIME-G, -A, -S, -GA, -GS, -AS |
| `matlab/MRIME_RL.m`, `MRIME_RL_22.m`, `MRIME_RL_33.m`, `RIME_RL.m` | earlier development variants, not called by any script |
| `matlab/Run_CEC2017_*_MRIME.m`, `Run_CEC2022_*_MRIME.m` | experiment drivers |
| `matlab/Statistics_*.m`, `A_Fiugre*.m`, `Draw_4_3.m`, `rank_mean_23.m`, `Get_Functions_cec20*.m` | statistics, figure and wrapper scripts |

## Third-party: algorithms compared in the paper (Table 2)

| File | Algorithm | Attribution stated in the file header |
|---|---|---|
| `matlab/RIME.m` | RIME (base algorithm) | none. RIME was proposed by H. Su et al., "RIME: A physics-based optimization", *Neurocomputing* (2023) |
| `matlab/EO.m` | Equilibrium Optimizer | none (empty header box) |
| `matlab/SAO.m` | Snow Ablation Optimizer | "Author: Lingyun Deng, Sanyang Liu"; main paper in *Expert Systems with Applications*, DOI 10.1016/j.eswa.2023.120069 |
| `matlab/ACGRIME.m` | ACGRIME | none |
| `matlab/IRIME.m` | IRIME | none |
| `matlab/TERIME.m` | TERIME | none |
| `matlab/EOSMA.m` | EOSMA | "Author: Shihong Yin, Qifang Luo, and Yongquan Zhou", *Arabian Journal for Science and Engineering* (2022) |
| `matlab/RDGMVO.m`, `matlab/RDGMVO-main/` | RDGMVO | none in the file; the README of the upstream archive is kept in `docs/third-party/rdgmvo-original-readme.md` |
| `matlab/MTVSCA.m` | MTV-SCA | "MTV-SCA: Multi-trial Vector-based Sine Cosine Algorithm", M. H. Nadimi-Shahraki, S. Taghian, D. Javaheri, A. S. Sadiq, N. Khodadadi, S. Mirjalili |

## Third-party: other algorithm files in the archive (not used by the paper's scripts)

| File | Algorithm | Attribution stated in the file header |
|---|---|---|
| `matlab/ALSHADE.m` | AL-SHADE | Y. Li, T. Han, H. Zhou, S. Tang, H. Zhao, *Information Sciences*, DOI 10.1016/j.ins.2022.05.058 |
| `matlab/LSHADE.m` | L-SHADE | none |
| `matlab/JADE.m` | JADE | cites Zhang and Sanderson (2009); the header is that of a redistributor ("KAU的云实验台") and contains a link to paid code |
| `matlab/MPA.m` | Marine Predators Algorithm | "programming: Afshin Faramarzi & Seyedali Mirjalili", DOI 10.1016/j.eswa.2020.113377 |
| `matlab/MRFO.m` | Manta Ray Foraging Optimization | "MRFO code v1.0", W. Zhao, Z. Zhang, L. Wang (2019) |
| `matlab/MVO.m`, `MVORouletteWheelSelection.m`, `RouletteWheelSelection.m` | Multi-Verse Optimizer | "Author and programmer: Seyedali Mirjalili"; the roulette code is credited to a playmedusa.com blog post |
| `matlab/MGO.m` | Moss Growth Optimization | B. Zheng, Y. Chen, C. Wang, A. A. Heidari, L. Liu, H. Chen (2024), aliasgharheidari.com |
| `matlab/ECO.m` | Educational Competition Optimizer | J. Lian, T. Zhu, L. Ma, X. Wu, A. A. Heidari, Y. Chen, H. Chen, G. Hui (2024), github.com/junbolian/ECO |
| `matlab/EDO.m` | Exponential Distribution Optimizer | "Developed in MATLAB R2022a by Abdel-Basset et al.", *Artificial Intelligence Review* (2023) |
| `matlab/DBO.m` | Dung Beetle Optimizer | "Programmed by Jian-kai Xue", Xue and Shen (2022), DOI 10.1007/s11227-022-04959-6 |
| `matlab/CFOA.m` | Catch Fish Optimization Algorithm | header box with the algorithm title, author fields left blank |
| `matlab/IGWO.m` | Improved Grey Wolf Optimizer | title only |
| `matlab/SAOMPSO.m` | SAO-inspired PSO | title only |
| `matlab/AFDB_ARO.m`, `FDBARO.m`, `BRDGMVO.m`, `CSOAOA.m`, `IYDSE.m` | further variants | none |

## Third-party: benchmark code and data

| Path | Content | Attribution stated in the file |
|---|---|---|
| `matlab/cec17_func.cpp`, `matlab/input_data/` | CEC2017 single-objective bound-constrained suite | "CEC17 Test Function Suite for Single Objective Optimization - Bound constraints", Noor Awad, 2016-2017 |
| `matlab/cec22_func.cpp`, `matlab/input_data22/` | CEC2022 single-objective bound-constrained suite | "CEC22 Test Function Suite for Single Objective Bound Constrained Numerical Optimization" |

Sources, versions and checksums of these files: `data/README.md`.

The Python package compiles these two sources unmodified (see `src/mrime/benchmarks/native.py`).

## Third-party: helper functions

| File | Attribution stated in the file |
|---|---|
| `matlab/cauchyinv.m`, `cauchyrnd.m` | "Copyright (C) Peder Axensten" |
| `matlab/gnR1R2.m` | "Written by Jingqiao Zhang", 2008 (JADE) |
| `matlab/levy.m`, `levyGen.m` | "Coded by Hemanth Manjunatha on Nov 13 2015", after Yang and Deb |
| `matlab/logsumexp.m` | "Written by Mo Chen" |
| `matlab/SHeatmap.m` | Zhaoxu Liu / slandarer (2023), "special heatmap", MATLAB Central File Exchange 125520 |
| `matlab/boundConstraint.m`, `boundConstraint2.m` | L-SHADE bound handling |
| `matlab/RFDB.m`, `dFDB.m`, `fitnessDistanceBalance*.m` | fitness-distance-balance selection family; no author stated |
| remaining small helpers (`initialization*.m`, `Bernoulli.m`, `Cubic*.m`, `Goodnode.m`, `RotMatrix.m`, `get_orthonormal.m`, `mixGaussEm.m`, `loggausspdf.m`, `levyFlight.m`, `levyrand.m`, `randCauchy.m`, ...) | none |

## Figures of the article

`assets/paper_figures/fig01.png` ... `fig18.png` are reproduced from the published article
(*Cluster Computing* 28:658, 2025, https://doi.org/10.1007/s10586-025-05348-3); copyright
remains with the publisher/authors as stated there. They are not covered by `LICENSE`.

## Python dependencies

NumPy, SciPy, PyYAML, matplotlib (optional), PyMuPDF (optional, only used by
`scripts/extract_paper_figures.py`) and `opfunu` (optional; its package metadata states GPLv3;
it is only imported by the `opfunu` benchmark backend) are installed from PyPI under their own
licences and are not redistributed here.
