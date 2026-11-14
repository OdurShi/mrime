# Implementation notes

Paper: S. Shi, L. Zhang, Y. Yin, X. Yang, H. Lee, "A modified RIME algorithm with covariance
learning and diversity enhancement for numerical optimization", *Cluster Computing* 28:658 (2025),
DOI [10.1007/s10586-025-05348-3](https://doi.org/10.1007/s10586-025-05348-3).
Equation, table and algorithm numbers below refer to that article.

Contents

1. [Paper to code map](#1-paper-to-code-map)
2. [One generation, step by step](#2-one-generation-step-by-step)
3. [Audit of the MATLAB code against the paper](#3-audit-of-the-matlab-code-against-the-paper)
4. [Unspecified details and choices](#4-unspecified-details-and-choices)
5. [Inconsistencies and probable typos in the paper](#5-inconsistencies-and-probable-typos-in-the-paper)
6. [Benchmark backends](#6-benchmark-backends)
7. [Statistics](#7-statistics)
8. [Known limitations and what is not verified](#8-known-limitations-and-what-is-not-verified)
9. [Record of the runs made with the Python package](#9-record-of-the-runs-made-with-the-python-package)

## 1. Paper to code map

| Paper | What | Code |
|---|---|---|
| Eq. (1) | uniform random initial population | `src/mrime/core.py:run` |
| Eq. (3) | fitness vector | `src/mrime/problem.py:Evaluator.evaluate` |
| Eq. (4) | soft-rime search | `src/mrime/operators.py:soft_rime_positions` |
| Eq. (5) | angle `theta` | `operators.py:soft_rime_angle` |
| Eq. (6) | step function `beta` | `operators.py:soft_rime_step` |
| Eqs. (4)-(6) | factor `alpha cos(theta) beta` | `operators.py:rime_factor` |
| Eq. (7) | `E = sqrt(FEs/FEsmax)` | `operators.py:exploration_probability` |
| Eq. (8) | hard-rime puncture, `nrom(Fit)` | `operators.py:normalize_fitness`, `hard_rime_mask` |
| Sect. 2.4 | positive greedy selection | `core.py:run` |
| Sect. 3.1 | roulette domain selection, first-in-first-out dominant population | `operators.py:rfdb_select`, `DominantArchive`, `select_dominant_group` |
| Eq. (9) | weighted mean `X_mean` | `operators.py:weighted_mean` |
| Eq. (10) | weights `w_i` | `operators.py:rank_weights` |
| Eq. (11) | covariance `C` | `operators.py:covariance` |
| Eqs. (9)-(11) | Gaussian model of the dominant group | `operators.py:fit_gaussian_model`, `GaussianModel` |
| Eq. (12) | GCLS candidate | `operators.py:gcls_candidates` |
| Eq. (13) | switch between soft-rime and GCLS | `core.py:run` |
| Eq. (14) | average bootstrapping strategy | `operators.py:abs_puncture` |
| Eqs. (15)-(17) | `V_lim`, `V_pop`, `nVOL` | `operators.py:normalized_volume` |
| Eq. (18) | SPDM candidate | `operators.py:spdm_candidates` |
| Algorithm 1, Fig. 2 | MRIME-CD loop | `core.py:run`, `core.py:mrime_cd` |
| Sect. 2 | base RIME | `core.py:rime` (= `run` with the three strategies off) |
| Table 2 | `w = 5`, `FEsmax = 3000 x Dim`, 51 runs | `config.py:MRIMEConfig.w`, `experiment.py:ExperimentConfig`, `configs/paper/*.yaml` |
| Tables 3-4 | CEC2017 (29 functions), CEC2022 (12 functions) | `benchmarks/cec.py`, `benchmarks/native.py` |
| Table 5 | ablation variants RIME-G/A/S/GA/GS/AS | `config.py:VARIANTS`, `configs/paper/*_ablation.yaml` |
| Tables 6, 9, 14 | Friedman test | `analysis.py:friedman_test` |
| Tables 7, 8 | Wilcoxon rank-sum test, w/e/l | `analysis.py:win_equal_loss` |
| Tables 12-13, Figs. 4, 10 | Kruskal-Wallis test | `analysis.py:kruskal_test` |
| Tables 10, 11, 17-20 | best / mean / std / rank | `analysis.py:summary_rows` |
| Figs. 6, 9, 13-18 | convergence curves | `plotting.py:plot_convergence` |
| Sect. 5, Tables 15-16 | ten constrained engineering problems | not implemented (see section 8) |

## 2. One generation, step by step

`core.py:run` performs, per generation (`FEs` is the number of objective evaluations consumed
so far, including the initial population):

1. `progress = FEs / FEsmax`, `E = sqrt(progress)` (Eq. 7), one `alpha ~ U(-1, 1)`,
   `factor = alpha * cos(theta) * beta` (Eqs. 5-6), `nrom(Fit) = Fit / ||Fit||_2`.
2. If any of GCLS / ABS / SPDM is enabled: append the best `round(0.3 NP)` agents to the
   first-in-first-out archive (capacity `25 D`), draw `X_r` from the archive by roulette
   fitness-distance balance, take the `floor(|archive| / 2)` archive members nearest to `X_r` as
   the dominant group `S`, and compute `X_mean` (Eqs. 9-10) and `C` (Eq. 11).
3. Candidate of agent `i` (Eq. 13): draw `r1`. If `r1 < E`, every dimension is replaced by the
   soft-rime value of Eq. (4) with probability `E`; otherwise the whole vector is the GCLS
   sample of Eq. (12). Without GCLS, step 3 is the base RIME rule (Eq. 4 per dimension).
4. Hard-rime puncture: every dimension `j` with `rand < nrom(Fit_i)` is set to `X_best,j`
   (Eq. 8). With ABS (Eq. 14) the value is `X_best,j` if `r1 < E` and
   `(X_best,j + X_mean,j) / 2` otherwise.
5. Boundary absorption (clip to `[LB, UB]`), evaluation, positive greedy selection; `Count_i`
   is incremented for every agent whose candidate was not better.
6. SPDM: if `nVOL < threshold` (Eqs. 15-17), every agent with `Count_i > 2 D` except the
   current best is re-sampled with Eq. (18), evaluated and accepted if better (`Count_i` is
   reset to 0 on acceptance and incremented otherwise).
7. The best-so-far value is recorded together with `FEs`.

The loop stops when `FEs = FEsmax`. The last generation evaluates only as many candidates as
the budget still allows; SPDM is truncated in the same way.

## 3. Audit of the MATLAB code against the paper

The MATLAB code could not be executed for this audit (no MATLAB or Octave available); the
findings below come from reading the sources. Nothing under `matlab/` was modified.

### 3.1 Which file is which

| Paper name (Table 5) | GCLS | ABS | SPDM | MATLAB file | Called by |
|---|---|---|---|---|---|
| RIME | N | N | N | `RIME.m` | all four `Run_*` scripts |
| RIME-G | Y | N | N | `MRIME_RL_1.m` | `Run_CEC2017_Strategy_MRIME.m`, `Run_CEC2022_Strategy_MRIME.m` |
| RIME-A | N | Y | N | `MRIME_RL_2.m` | strategy scripts |
| RIME-S | N | N | Y | `MRIME_RL_3.m` | strategy scripts |
| RIME-GA | Y | Y | N | `MRIME_RL_12.m` | strategy scripts |
| RIME-GS | Y | N | Y | `MRIME_RL_13.m` | strategy scripts |
| RIME-AS | N | Y | Y | `MRIME_RL_23.m` | strategy scripts |
| MRIME-CD | Y | Y | Y | `MRIME_RL_Cov.m` | all four `Run_*` scripts |

Not called by any script: `MRIME_RL.m` (all three strategies, dominant group taken from the
current population instead of the archive; its call is commented out in the strategy scripts),
`MRIME_RL_22.m` and `MRIME_RL_33.m` (earlier ABS-only and SPDM-only versions without the
archive), `RIME_RL.m` (a different restart idea).

### 3.2 Discrepancies between paper and MATLAB code

"Default" is what the Python package does out of the box (profile `paper`); "MATLAB behaviour"
is selected with the profile `matlab` (e.g. `algorithm="mrime-cd@matlab"` or
`config=mrime.MATLAB`) or with the individual `MRIMEConfig` field given in the last column.
Line numbers refer to `matlab/MRIME_RL_Cov.m` unless another file is named.

| # | Topic | Paper | MATLAB code (file:line) | Python default | Flag for MATLAB behaviour |
|---|---|---|---|---|---|
| D1 | evaluation counter of MRIME-CD | "we use the maximum number of evaluations as the algorithm stopping criterion, so that MRIME-CD and RIME can be compared under fair conditions" (Sect. 3.4) | `MRIME_RL_Cov.m:164-166`: `if rand<0.1, it=it+1; end` - the counter that stops the run advances with probability 0.1 per objective evaluation. All other files (`RIME.m:60`, `MRIME_RL_1.m:137`, ...) advance it by 1 per evaluation | every evaluation counts | `accounting="matlab-as-shipped"` (profile `matlab-as-shipped`), diagnostic only |
| D2 | SPDM evaluations | counted in the complexity analysis (`T x (NP + N_re) x D`) | `MRIME_RL_Cov.m:193` evaluates, `:207-211` the counter update is commented out, so SPDM evaluations are free; same in `MRIME_RL.m`. They are counted in `MRIME_RL_3.m:183`, `MRIME_RL_13.m`, `MRIME_RL_23.m` | counted | same as D1 |
| D3 | initial population | not specified | all files: the `N` initial evaluations (`:20-27`) are not counted, the loop then runs until the counter exceeds `FesMax`, so a run uses `FesMax + N` evaluations | counted (total is exactly `FEsmax`) | same as D1 |
| D4 | `theta` (Eq. 5) | `theta = pi FEs / (10 FEsmax)` | `:43` (and `RIME.m:26`): `cos(pi*it/(FesMax/10))`, i.e. `theta = 10 pi FEs / FEsmax` (five full cosine periods instead of a slow decay from 1 to 0.95) | paper | `theta_mode="matlab"` |
| D5 | weights (Eq. 10) | `ln(|S| + 1)` | `:88`: `log(SEL+1/2)-log(1:SEL)` | `ln(|S| + 1) - ln(i)` | `weight_offset=0.5` |
| D6 | ranking behind the weights | "better individuals have a greater impact" | `:67-71, 87-90`: members are ordered by distance to `X_r`, the nearest gets the largest weight | by fitness | `weight_order="distance"` |
| D7 | covariance (Eq. 11) | `1/|S|` | `:93`: `1/(SEL-1)` | `1/|S|` | `cov_ddof=1` |
| D8 | reference point of ABS (Eq. 14) | `X_best,j` | `:122, 124, 138, 140` (and `MRIME_RL_2.m:98, 100`, `MRIME_RL_23.m`): `best(1,j)`, where `best` (`:66`) is the roulette-selected `X_r` from the archive, not `Best_rime` | `X_best` | `abs_reference="roulette"` |
| D9 | gate of ABS (Eq. 14) | the same `r1` as in Eq. (13) | `:107, 121, 137`: an independent per-agent `r4` | shared `r1` | `abs_gate="independent"` |
| D10 | stagnation threshold | `nVOL < 0.01` (Sect. 3.3 text), `nVOL < 0.1` (Algorithm 1, Fig. 2) | `:183`: `nVOL < 0.001` | `0.01` | `nvol_threshold=1e-3` |
| D11 | search range of CEC2022 F11 | `[-100, 100]^D` (Table 4) | `Get_Functions_cec2022.m:70-71`: `[-600, 600]^D` | `[-100, 100]` | `matlab_bounds=True` in `get_problem` / the experiment config |
| D12 | Gaussian sample in SPDM (Eq. 18) | a sample of `Gaussian(X_mean, C)` | `:188`: re-uses `zz(i,:)`, the standard-normal vector already used for agent `i`'s GCLS candidate in the same generation | fresh sample | none (same distribution) |
| D13 | SPDM update order | per agent inside the loop (Algorithm 1) | `:175-216`: after the whole population has been evaluated; agents are re-sampled one after another, so `X_r1`, `X_r2` can already be re-sampled agents | after the population, all stagnant agents at once from the same population snapshot | none |
| D14 | fallback of the dominant group | not described | `:62-84`: if `25*dim <= N` the dominant group is taken from the current population instead of the archive. Never true in the paper's settings | archive always | none |

D1-D3 change the amount of search an algorithm is allowed to do and therefore matter far more
than D4-D14; section 3.3 quantifies them.

### 3.3 The evaluation counter of `MRIME_RL_Cov.m`

`Run_CEC2017_total_MRIME.m:48,63` and `Run_CEC2022_total_MRIME.m:48,63` call
`MRIME_RL_Cov(pop_size, max_iter, ...)` and `RIME(pop_size, max_iter, ...)` with the same
`max_iter = 3000*dim`. Inside `RIME.m` and the six ablation files the counter `it` is
incremented once per objective evaluation (the competitor files were not audited line by line;
`EO.m`, `SAO.m` and `IRIME.m`, which were inspected for this point, also count every
evaluation). Inside `MRIME_RL_Cov.m` it is
incremented only when `rand < 0.1` (line 164), and not at all for SPDM evaluations (lines
207-211 are commented out). With the file as shipped, MRIME-CD therefore stops after about
`10 x FEsmax` main-loop evaluations plus all SPDM evaluations, while RIME and the six
ablation variants stop after `FEsmax` (+ `N`). The schedule variables `E`,
`theta` and `beta` are driven by the same slow counter, and `Convergence_curve(it)` is indexed
by it, so a convergence plot against `it` compresses roughly ten evaluations of MRIME-CD into
one unit of the x-axis.

What the Python package does:

* default (`accounting="strict"`): one unit of budget per objective call, for every variant.
  `tests/test_core.py::test_strict_evaluation_accounting_and_bounds` checks this with an
  independent call counter for all eight variants and both profiles;
* `accounting="matlab-as-shipped"` reproduces the counter of `MRIME_RL_Cov.m` so that its
  effect can be measured. `OptimizeResult.n_evals` still reports the true number of calls.
  In the reduced run of this repository (CEC2022, D = 10, `FEsmax = 30000`, `NP = 140`,
  10 runs per function) this mode used on average 528,027 objective evaluations per run, i.e.
  17.6 times `FEsmax` (`results/reduced/cec2022_d10_matlab/report.md`). In that run the
  as-shipped arm, and not the equal-budget arms, reproduces the magnitude of the MRIME-CD
  column of the paper's Table 10 on the functions where precision matters (F1: mean 300.00,
  std 2.2e-08 here against 1.4e-08 in the paper; F3: std 1.5e-05 against 1.6e-05; F9: std
  8.4e-11 against 1.0e-10; with every evaluation counted the same operators give F1 std
  4.4e-02, F3 std 7.7e-03, F9 std 7.2e-05).

Whether the tables of the paper were produced with the file as shipped cannot be determined
from the repository (no result files are included, and the `save` lines of the run scripts are
commented out). Suggested fix for the MATLAB file, to be applied and re-run by the authors:

```matlab
% matlab/MRIME_RL_Cov.m, replace lines 164-166
it = it + 1;
% and re-enable lines 207-211 so that SPDM evaluations advance and check the counter
it = it + 1;
Convergence_curve(it) = Best_rime_rate;
if it > FesMax
    break;
end
```

and, in all files, start the counter at `N` after the initial evaluations (or subtract `N` from
`FesMax`) so that the total equals `FEsmax`.

### 3.4 Other findings in the MATLAB archive

| File:line | Finding | Suggested fix |
|---|---|---|
| `Run_CEC2017_total_MRIME.m:28` vs `Run_CEC2022_total_MRIME.m:28`, `Run_CEC2017_Strategy_MRIME.m:28`, `Run_CEC2022_Strategy_MRIME.m:28` | population size is `ceil(15*dim^(2/3))` in the first script and `ceil(30*dim^(2/3))` in the other three; the paper does not state the population size | state the value(s) used in the README of the MATLAB code / an erratum |
| all `Run_*` scripts `:7` | `load('input_data\Rand_Seeds.txt')` uses a Windows path separator | `fullfile('input_data','Rand_Seeds.txt')` |
| all `Run_*` scripts `:31-32` | the `save` command is commented out, so a run leaves no result file for `Statistics_*.m` and `A_Fiugre*.m` | uncomment / add `save` at the end of the script |
| all `Run_*` scripts, e.g. `Run_CEC2017_total_MRIME.m:78` | after `rk = kruskalwallis(...)` the NaN guard tests `isnan(rs)` instead of `isnan(rk)` | `if isnan(rk)` |
| `MRIME_RL_Cov.m:167` (all optimiser files) | `Convergence_curve(it)` is written after `it` is incremented: element 1 is never written (stays 0) and element `FesMax+1` can be written | write before incrementing |
| `MRIME_RL_Cov.m:31-40` | the `Par_restart` struct is rebuilt every generation; its fields `counter`, `m`, `m_max`, `m_min` are unused | compute `V_lim` once before the loop |
| `MRIME_RL_13.m:175`, `MRIME_RL_23.m:166`, `MRIME_RL.m:159` | `xmean++randnum(1)*...` (a stray unary plus, harmless) | remove one `+` |
| `Statistics_Strategy1_selection.m:5-6` | loads `MSAO_EDA_Strategy1_Selection_..._10000xDim_...mat`, a file name from another study with another budget | point it at the file saved by the strategy scripts |
| `Statistics_Duibi.m:56-78` | the w/e/l counts are derived from the Kruskal-Wallis p-values; the block using `ranksum` (lines 31-53) is commented out, while the paper describes Table 8 as a Wilcoxon rank-sum test (for two samples the two tests are nearly equivalent) | use `rank_sum_RESULT` |
| `A_Fiugre*.m`, `Statistics_Strategy1_selection.m:61, 184` | need `slanCL` (a colour-map function that is not in the archive) and `.mat` result files | add the dependency or replace the colour map |
| optimiser files | `normr` needs the Deep Learning Toolbox; `pdist2`, `ranksum`, `kruskalwallis`, `friedman` need the Statistics and Machine Learning Toolbox; `parfor` the Parallel Computing Toolbox | document; `normr(v)` can be replaced by `v/norm(v)` |
| `SAO.m:152-156` | a leftover `if rand<0.1 ... end` block that only repeats the termination check (no effect on the count) | remove |
| - | no code for the ten engineering problems of Sect. 5 (Tables 15-16) is in the archive | add it or state that it is not released |

## 4. Unspecified details and choices

| Topic | Paper | Choice made here | Basis |
|---|---|---|---|
| population size `NP` | not stated | no default; the experiment configs use `ceil(15 D^(2/3))` for CEC2017 and `ceil(30 D^(2/3))` for CEC2022 and the ablation | the author's run scripts (section 3.4) |
| size of the elite added to the archive, archive capacity, size of the dominant group | "selects the part of the individuals with closer distance", "when the number of dominant populations is too large" - no numbers | `round(0.3 NP)` best agents per generation, capacity `25 D`, nearest `floor(0.5 |archive|)` to `X_r` | `MRIME_RL_Cov.m:49-71`; exposed as `elite_fraction`, `archive_capacity_per_dim`, `neighbour_fraction` |
| "adaptive distance balance mechanism based on roulette" | no formula, no reference | roulette fitness-distance balance: score = (1 - normalised fitness) + normalised L1 distance to the best archive member, selection probability proportional to the score, uniform choice when all fitness values are equal | `matlab/RFDB.m` |
| order of selection steps for the dominant group | the text can be read as "select neighbours of `X_r` in the population, then archive them" | elite by fitness is archived first; `X_r` and its neighbours are then selected inside the archive | `MRIME_RL_Cov.m:49-71` |
| soft-rime inside Eq. (13) | Eq. (13) is written per agent, Eq. (4) per dimension | per-agent `r1` selects soft-rime or GCLS; within a soft-rime agent each dimension is updated with probability `E` (a second, per-dimension draw), as Algorithm 1 ("If r1 < E: compute the position using Eq. (4)") implies | `MRIME_RL_Cov.m:108-116` |
| which `r1` gates ABS when GCLS is off (RIME-A, RIME-AS) | not discussed | the per-dimension `r1` of Eq. (4) | literal reading of Eq. (14); it is what the early file `MRIME_RL_22.m` does |
| `alpha` in Eq. (4) | "a randomly generated value between -1 and 1" | one draw per generation, shared by all agents | `RIME.m:26` |
| `nrom(Fit_i)` in Eq. (8) | "the normalized value" | `Fit / ||Fit||_2` | `normr(Rime_rates)` in `RIME.m:29` |
| boundary handling | not stated | clipping to the bound | `RIME.m:47-49` |
| Gaussian sampling when `C` is singular | not stated | eigen-decomposition, negative eigenvalues clipped to 0, `1e-7` added to every eigenvalue before the square root, condition guard at `1e20` | `MRIME_RL_Cov.m:94-102` (`eig_floor`) |
| which count `Count_i` holds | text: incremented when the offspring is better; Algorithm 1: incremented when it is not | incremented on failure (the only reading under which "Count > 2 D" indicates stagnation) | Algorithm 1; `MRIME_RL_Cov.m:161-163` |
| reset of `Count_i` | not stated | never reset by a success in the main loop; reset to 0 when an SPDM candidate is accepted | `MRIME_RL_Cov.m:198` |
| acceptance of SPDM candidates | "update the stagnant individuals" | greedy (kept only if better), the current best agent is not re-sampled | `MRIME_RL_Cov.m:185, 195` |
| `X_r1`, `X_r2` in Eq. (18) | "two different individuals randomly selected from the population" | distinct from each other, may coincide with `X_i` | `randperm(N)` in `MRIME_RL_Cov.m:186` |
| stagnation threshold | 0.01 or 0.1 (see section 5) | 0.01, the value in the prose that defines the mechanism | configurable (`nvol_threshold`) |
| counter used in `E`, `theta`, `beta` | `FEs` | all evaluations so far including the initial population (MATLAB starts at 1 after the initial population) | strict accounting |
| last generation when fewer than `NP` evaluations remain | not stated | the first `remaining` candidates are evaluated | same effect as the `break` in `RIME.m:62` |
| seeds | not stated | NumPy `Generator` (PCG64); the seed of run `r` of a function is shared by all algorithms, so they start from the same initial population | the MATLAB scripts re-seed `rng(run_seed,'twister')` per run and algorithm |
| direction of a significant rank-sum result | not stated | the algorithm with the smaller mean wins | `Statistics_*.m` use the rank of the mean |

## 5. Inconsistencies and probable typos in the paper

These are reported for the authors; none of them was "fixed" silently.

1. **Eq. (10)**: as printed, `w_i = ln(|S|+1) / sum_i (ln(|S|+1) - ln(i))` does not depend on
   `i` and does not sum to one. The numerator should read `ln(|S|+1) - ln(i)`.
2. **Eq. (14)**: the conditions are joined by a union sign; a logical "and" is meant.
3. **Stagnation threshold**: Sect. 3.3 states `nVOL < 0.01`; Algorithm 1 and Fig. 2 state
   `nVOL < 0.1`; the released code uses `0.001`.
4. **Count**: Sect. 3.3 says Count is incremented when the offspring is better than its parent;
   Algorithm 1 increments it in the `Else` branch (offspring not better).
5. **Equation references in Algorithm 1 and Fig. 2** are shifted: "C, X_mean using Eq. (9) and
   Eq. (10)" (Eqs. 9-11), "position using Eq. (11) GCLS" (Eq. 12), "Eq. (13) ABS" (Eq. 14),
   "nVOL using Eq. (16)" (Eq. 17), "position using Eq. (16) SPDM" (Eq. 18). The update
   `FEs = FEs + NP` omits the SPDM evaluations that the complexity analysis counts.
6. **`nVOL` is strongly dimension dependent**: if the population spans the fraction `rho` of
   the search range in every dimension, `nVOL = (rho/2)^(D/4)`. Since `rho <= 1`, `nVOL` is
   below 0.1 for every population when `D >= 14`, below 0.01 when `D >= 27` and below 0.001
   when `D >= 40`. For the D = 30/50/100 experiments the volume condition is therefore always
   true and SPDM is governed by `Count_i > 2 D` alone.
7. **Population size** is not given anywhere (Table 2 lists `FEsmax`, the dimensions and the
   number of runs only).
8. **Eq. (5) versus the RIME reference code**: see D4; the same difference exists between the
   original RIME article and its public code, so results of any code-based RIME differ from the
   equation as printed.
9. **Reference for RIME**: the text cites "[15]" for the RIME algorithm of Su et al. (2023), but
   entry [15] of the reference list is the Alpha evolution paper; the RIME article (H. Su et
   al., "RIME: A physics-based optimization", *Neurocomputing*, 2023) is not in the list.
10. **Table 6** carries the caption of Table 5 ("Description of MRIME-CD variants") although it
    contains the Friedman ranks; the text quotes mean ranks "1.51 and 1.67" while the table
    lists 1.51 and 2.00 (1.67 is the CEC2022 D = 20 row).
11. **Fig. 3** is captioned "Friedman ranking of MRIME-CD and other variants" but its legend
    shows the competitor algorithms (MRIME-CD, RIME, EO, SAO, ...), i.e. it appears to be the
    figure of the competitor comparison.
12. **Table 3** lists "Hybrid function 6" five times (F15-F19) instead of hybrid functions
    6-10; the function called "Expanded Scaffer's F6" is implemented as Schaffer's F7 in the
    official CEC2017 code (a known property of that code, not of the paper).
13. Minor: "MRIMR-CD", "MRIMME-CD", "CE2017", "nrom", "D = 20D" in Table 7.

## 6. Benchmark backends

**Native (default).** `src/mrime/benchmarks/native.py` compiles `matlab/cec17_func.cpp` and
`matlab/cec22_func.cpp` *unmodified* into shared libraries (a generated translation unit
includes the source after a stub `mex.h`) and calls `cec17_test_func` / `cec22_test_func`
through `ctypes`, reading the data in `matlab/input_data` and `matlab/input_data22`. These are
the functions the MATLAB experiments evaluate through MEX. Checks performed
(`python scripts/verify_cec_backends.py`, stored in `results/reduced/cec_backend_check.md`):
`|f(x*) - f*| <= 1.1e-10` for all 29 CEC2017 functions at D = 10/30/50/100 and all 12 CEC2022
functions at D = 10/20, with `f* = 100 x (official function number)` for CEC2017 and the
values of Table 4 for CEC2022. Two properties of the reference code are worth knowing:

* the official CEC2017 numbering has 30 functions with F2 withdrawn; the paper's `F1..F29`
  are the official `F1, F3..F30` (`benchmarks/cec.py:cec2017_official_number`);
* the optimum of the CEC2017 Levy function (paper F8, official F9) is not at the shift vector
  `o` but at `o + M^-1 1`, because the code uses `w = 1 + (z - 1)/4`.

Compiler flags: `-O2 -ffp-contract=off`. Built and run here with Apple clang 21 on arm64;
the build on other platforms (g++ on Linux is what the CI workflow would use) was not
executed. The MATLAB MEX build of the same source may differ in the last bits.

**opfunu (optional).** `opfunu` 1.0.4 offers CEC2017 `F1..F29` and CEC2022 `F1..F12`, so no
function of the paper is missing, but it is *not* a numerically faithful implementation of the
reference code:

* only CEC2017 F1 and CEC2022 F2 agree with the reference code (relative difference below
  1e-9 at random points); the other 28 + 28 + 11 + 11 = 78 of 82 tested (suite, dimension,
  function) cases differ, often by orders of magnitude at the same point;
* its CEC2017 classes are numbered 1..29 with bias `100 k`; for `k >= 2` the official optimum
  value is `100 (k + 1)`, and for F2-F19 opfunu pairs the base function with the shift and
  rotation data of the *previous* official function, so the optimum is at a different
  location (evaluating the reference code at opfunu's optimum gives values far from `f*`);
* some base functions use different formulas (e.g. opfunu's Zakharov uses `sum 0.5 x_i`
  where the reference code uses `sum 0.5 i x_i`);
* opfunu 1.0.4 imports `pkg_resources`, which requires `setuptools < 81`.

The wrapper in `benchmarks/cec.py` shifts opfunu's bias so that `f*` equals the official
value. Results obtained with `--backend opfunu` are results on opfunu's functions and must not
be compared number-for-number with the paper. The full comparison table is in
`results/reduced/cec_backend_check.md`.

**Classical functions.** `benchmarks/classical.py`: sphere, Schwefel 2.22, Rosenbrock,
Rastrigin, Ackley, Griewank, Zakharov and a shifted rotated ill-conditioned quadratic
(`ellipsoid`). They are not part of the paper; they exist for the tests and the quick start.

## 7. Statistics

* Standard deviation: sample standard deviation (`ddof = 1`), as MATLAB's `std`.
* Rank of the mean within a function: competition ranking, ties share the smallest rank
  (`matlab/rank_mean_23.m`).
* Wilcoxon rank-sum: `scipy.stats.mannwhitneyu(..., alternative="two-sided")`. For samples of
  51 runs both SciPy and MATLAB's `ranksum` use the normal approximation with tie and
  continuity correction; for very small samples both switch to an exact method, with slightly
  different switching rules. Identical samples give `p = 1`, as in the MATLAB scripts. Win /
  loss needs `p < 0.05` and is decided by the mean.
* Friedman: `scipy.stats.friedmanchisquare` on the matrix of means (functions x algorithms);
  mean ranks use mid-ranks for ties. With only two algorithms the p-value is reported as NaN.
* Kruskal-Wallis: `scipy.stats.kruskal` per function.

## 8. Known limitations and what is not verified

* **MATLAB code**: not executed. All statements about it come from reading the sources.
* **Paper numbers**: the Python package was not used to produce the numbers in the paper, and
  it differs from the MATLAB code in the random number generator, in the vectorised order of
  random draws, in the points D1-D14 above and (default profile) in following the printed
  equations. Agreement can only be statistical. Section 9 lists what was run.
* **Competitors**: EO, SAO, ACGRIME, IRIME, TERIME, EOSMA, RDGMVO and MTVSCA exist only as
  third-party MATLAB files; Tables 8-14 and 17-20 can therefore be reproduced in Python only
  for the MRIME-CD and RIME columns.
* **Engineering problems** (Sect. 5, Tables 15-16): the paper names ten problems and their
  dimensions but gives neither the formulations nor the penalty settings, and the archive
  contains no code for them. They are not implemented.
* **Scale of the Python runs**: CEC2022 was run with the full 51 runs at D = 10 and 20 for all
  eight variants; CEC2017 only at D = 10 with 15 runs for RIME and MRIME-CD; the MATLAB-profile
  diagnostic with 10 runs. The CEC2017 ablation, D = 30/50/100 and the full driver
  (`scripts/reproduce_paper.sh`) were not run (a single CEC2017 D = 100 run takes 2-10 s on the
  machine used here; the full protocol is estimated at roughly 40 CPU-hours). `scripts/reproduce_reduced.sh` in its
  final form was not executed end to end; its four `mrime run` commands were executed
  individually.
* **Vectorisation**: candidates of a generation are generated from one population snapshot and
  evaluated in a block. This is equivalent to the MATLAB loop for the main update (candidates
  are generated before any evaluation there as well) but not for SPDM (D13).
* **CI**: `.github/workflows/ci.yml` was written but never executed (no push was made).
* **Figures and data**: `assets/paper_figures/` holds crops of the article PDF (Figs. 1-18,
  made with `scripts/extract_paper_figures.py`); they are the paper's figures, not outputs of
  this code. The CEC data in `matlab/input_data*` are the files of the authors' archive; 328 of
  the 329 CEC2017 files and all 54 CEC2022 files are numerically identical to the copies
  bundled with `opfunu` 1.0.4 (the remaining file, `Rand_Seeds.txt`, is not part of that
  package's CEC2017 data). They were not compared with a fresh download from the organisers.

## 9. Record of the runs made with the Python package

### 9.1 Commands and environment

One machine (Apple M5 laptop, macOS, Python 3.11.18, NumPy 2.4.6), 4 worker processes, native
CEC backend, base seed 2025, strict evaluation accounting unless stated otherwise:

```bash
python -m mrime run --config configs/reduced/cec2022_d10.yaml --baseline rime --output results/reduced/cec2022_d10                       # A
python -m mrime run --config configs/reduced/cec2022_d20.yaml --baseline rime --output results/reduced/cec2022_d20                       # B
python -m mrime run --config configs/reduced/cec2017_d10.yaml --baseline rime --output results/reduced/cec2017_d10                       # C
python -m mrime run --config configs/reduced/cec2022_d10_matlab.yaml --baseline rime@matlab --output results/reduced/cec2022_d10_matlab  # D
python scripts/summarize_results.py results/reduced/<name> --baseline <baseline>   # report.md
python -m mrime plot results/reduced/<name>                                        # figures
python scripts/verify_cec_backends.py --output results/reduced/cec_backend_check.md
```

Parts C and D were launched with `--algorithms rime mrime-cd --runs 15` and `--runs 10` on the
command line; the two YAML files now contain these values. `scripts/reproduce_reduced.sh`
chains the same commands.

| Part | Suite, D | Algorithms | Runs | `FEsmax` / `NP` | Wall-clock |
|---|---|---|---|---|---|
| A | CEC2022, 10 | eight variants of Table 5, profile `paper` | 51 | 30000 / 140 | 118 s |
| B | CEC2022, 20 | eight variants, profile `paper` | 51 | 60000 / 222 | 192 s |
| C | CEC2017, 10 | RIME, MRIME-CD, profile `paper` | 15 | 30000 / 70 | 78 s |
| D | CEC2022, 10, F11 in `[-600, 600]` | RIME and MRIME-CD with profile `matlab`; MRIME-CD with `matlab-as-shipped` | 10 | 30000 / 140 | 110 s |

A first attempt at part C with all eight variants and 51 runs was stopped before it finished;
nothing from it is recorded. Differences from the paper's protocol: Python instead of MATLAB
(different random number generator, vectorised draws); the RIME family only; D = 10 and 20;
15 and 10 runs in parts C and D; the default profile follows the printed equations; every
objective evaluation is counted. The benchmark functions are the C code the MATLAB experiments
call.

### 9.2 Parts A and B: CEC2022, 51 runs

Mean (standard deviation) of the final value; the sign column is the Wilcoxon rank-sum result
of MRIME-CD against RIME at the 5 % level. Full tables: `results/reduced/cec2022_d10/report.md`
and `results/reduced/cec2022_d20/report.md`.

| Function | D = 10: RIME | D = 10: MRIME-CD | | D = 20: RIME | D = 20: MRIME-CD | |
|---|---|---|---|---|---|---|
| F1 | 3.0962e+02 (7.39e+00) | 3.0000e+02 (1.06e-06) | + | 6.2648e+02 (1.77e+02) | 3.0004e+02 (4.45e-02) | + |
| F2 | 4.1152e+02 (1.74e+01) | 4.0638e+02 (3.06e+00) | + | 4.5178e+02 (1.16e+01) | 4.5017e+02 (5.15e+00) | + |
| F3 | 6.0060e+02 (2.69e-01) | 6.0000e+02 (4.15e-04) | + | 6.0229e+02 (1.14e+00) | 6.0003e+02 (4.16e-02) | + |
| F4 | 8.1713e+02 (5.67e+00) | 8.0860e+02 (3.83e+00) | + | 8.4463e+02 (1.32e+01) | 8.2141e+02 (8.42e+00) | + |
| F5 | 9.0065e+02 (5.89e-01) | 9.0000e+02 (1.04e-07) | + | 9.2430e+02 (3.08e+01) | 9.0003e+02 (9.13e-02) | + |
| F6 | 5.7867e+03 (2.64e+03) | 1.8042e+03 (7.47e+00) | + | 5.4107e+04 (4.68e+04) | 1.8957e+03 (4.03e+01) | + |
| F7 | 2.0154e+03 (9.00e+00) | 2.0159e+03 (9.02e+00) | = | 2.0441e+03 (1.24e+01) | 2.0307e+03 (6.68e+00) | + |
| F8 | 2.2192e+03 (7.22e+00) | 2.2152e+03 (9.37e+00) | + | 2.2311e+03 (5.97e+00) | 2.2264e+03 (1.81e+00) | + |
| F9 | 2.5293e+03 (3.46e-02) | 2.5293e+03 (3.03e-07) | + | 2.4815e+03 (4.37e-01) | 2.4808e+03 (2.10e-05) | + |
| F10 | 2.5004e+03 (8.46e-02) | 2.5327e+03 (5.07e+01) | - | 2.5461e+03 (1.02e+02) | 2.5650e+03 (1.58e+02) | - |
| F11 | 2.6365e+03 (8.61e+01) | 2.6564e+03 (1.04e+02) | - | 3.0205e+03 (1.27e+02) | 2.9181e+03 (5.56e+01) | + |
| F12 | 2.8634e+03 (1.49e+00) | 2.8640e+03 (1.09e+00) | = | 2.9458e+03 (7.95e+00) | 2.9455e+03 (1.02e+01) | = |
| w/e/l | | | 8/2/2 | | | 10/1/1 |

| Friedman mean rank | RIME | RIME-G | RIME-A | RIME-S | RIME-GA | RIME-GS | RIME-AS | MRIME-CD | p-value |
|---|---|---|---|---|---|---|---|---|---|
| CEC2022, D = 10 | 6.17 | 4.25 | 5.42 | 5.33 | 4.00 | 3.58 | 4.17 | 3.08 | 3.4e-02 |
| CEC2022, D = 20 | 7.33 | 2.58 | 6.58 | 5.58 | 2.92 | 3.00 | 4.67 | 3.33 | 6.3e-08 |

| w/e/l vs RIME | RIME-G | RIME-A | RIME-S | RIME-GA | RIME-GS | RIME-AS | MRIME-CD |
|---|---|---|---|---|---|---|---|
| CEC2022, D = 10 | 7/3/2 | 3/9/0 | 7/4/1 | 8/3/1 | 7/3/2 | 9/2/1 | 8/2/2 |
| CEC2022, D = 20 | 10/1/1 | 3/9/0 | 8/3/1 | 10/1/1 | 10/1/1 | 10/2/0 | 10/1/1 |

Observations next to the corresponding values of the paper (Tables 6, 7, 10, 11):

* MRIME-CD against RIME: 8/2/2 at D = 10 (paper Table 7: 8/2/2) and 10/1/1 at D = 20 (paper:
  12/0/0). RIME has the largest Friedman rank in both dimensions, as in the paper.
* MRIME-CD is significantly worse than RIME on F10 (both dimensions) and F11 (D = 10), and not
  significantly different on F7 (D = 10) and F12 (both).
* At D = 20 the Friedman ranks of RIME-G, RIME-GA and RIME-GS (2.58, 2.92, 3.00) are smaller
  than that of MRIME-CD (3.33); the paper's Table 6 ranks MRIME-CD first (1.67). RIME-A is not
  significantly different from RIME on 9 of 12 functions in both dimensions.
* Standard deviations of MRIME-CD on F1, F3 and F9 at D = 10 are 76, 26 and 3000 times the
  values of the paper's Table 10; on F1 at D = 20 the mean is 3.0004e+02 (4.45e-02) against
  3.0000e+02 (1.40e-07) in Table 11, on F6 at D = 20 it is 1.8957e+03 against 1.8149e+03.
* F11 at D = 10 is searched in `[-100, 100]` here and in `[-600, 600]` by the MATLAB code (D11);
  with the MATLAB range (part D) the values are 2.83e+03 to 2.89e+03, the size of the paper's.

### 9.3 Part C: CEC2017, D = 10, 15 runs

MRIME-CD against RIME, Wilcoxon rank-sum w/e/l: 16/10/3 (paper Table 8, 51 runs: 26/2/1).
MRIME-CD is significantly worse on F21, F23 and F25 and has the smaller mean on 21 of 29
functions. On F1 the means are 2.6921e+03 (3.35e+03) for MRIME-CD and 1.9084e+05 (1.56e+05) for
RIME, against 1.0001e+02 and 1.2427e+04 in the paper's Table 17. All 29 functions:
`results/reduced/cec2017_d10/report.md`.

### 9.4 Part D: MATLAB profile and the counter of `MRIME_RL_Cov.m`, CEC2022, D = 10, 10 runs

The three arms use the `matlab` profile and the MATLAB search range for F11. The first two
count every evaluation; the third uses `accounting="matlab-as-shipped"` (section 3.3).

| | RIME (`matlab`) | MRIME-CD (`matlab`) | MRIME-CD (`matlab-as-shipped`) |
|---|---|---|---|
| objective evaluations per run (mean) | 30,000 | 30,000 | 528,027 |
| F1 | 3.0084e+02 (4.42e-01) | 3.0003e+02 (4.42e-02) | 3.0000e+02 (2.17e-08) |
| F2 | 4.1643e+02 (2.61e+01) | 4.0439e+02 (4.17e+00) | 4.0209e+02 (3.05e+00) |
| F3 | 6.0024e+02 (1.09e-01) | 6.0001e+02 (7.66e-03) | 6.0000e+02 (1.45e-05) |
| F4 | 8.1993e+02 (6.38e+00) | 8.0886e+02 (2.67e+00) | 8.0557e+02 (1.70e+00) |
| F5 | 9.0041e+02 (3.14e-01) | 9.0001e+02 (2.82e-02) | 9.0005e+02 (1.43e-01) |
| F6 | 3.8105e+03 (1.76e+03) | 1.8124e+03 (1.16e+01) | 1.8008e+03 (9.02e-01) |
| F7 | 2.0199e+03 (6.31e+00) | 2.0187e+03 (6.24e+00) | 2.0103e+03 (1.04e+01) |
| F8 | 2.2156e+03 (9.32e+00) | 2.2173e+03 (8.39e+00) | 2.2106e+03 (1.05e+01) |
| F9 | 2.5293e+03 (4.44e-03) | 2.5293e+03 (7.22e-05) | 2.5293e+03 (8.36e-11) |
| F10 | 2.5120e+03 (3.67e+01) | 2.5114e+03 (3.50e+01) | 2.5109e+03 (3.39e+01) |
| F11 | 2.8341e+03 (1.52e+02) | 2.8862e+03 (4.77e+01) | 2.8412e+03 (1.27e+02) |
| F12 | 2.8650e+03 (3.22e+00) | 2.8642e+03 (6.80e-01) | 2.8638e+03 (1.83e+00) |
| w/e/l vs RIME (`matlab`) | | 9/3/0 | 10/2/0 |

Observations (10 runs, indicative):

* The RIME column is close to the RIME column of the paper's Table 10 (F1 3.0084e+02 and
  3.0095e+02; F6 3.81e+03 and 3.97e+03; F11 2.83e+03 and 2.86e+03).
* The MRIME-CD column of Table 10 is matched in magnitude by the as-shipped arm (standard
  deviations: F1 2.2e-08 here and 1.4e-08 in the paper, F3 1.5e-05 and 1.6e-05, F6 9.0e-01 and
  6.8e-01, F9 8.4e-11 and 1.0e-10), which used 17.6 times `FEsmax` evaluations on average. With
  every evaluation counted the same operators give 4.4e-02 (F1), 7.7e-03 (F3), 1.2e+01 (F6) and
  7.2e-05 (F9).
* With every evaluation counted, MRIME-CD with the `matlab` profile is significantly better
  than RIME on 9 of 12 functions and not significantly worse on any.

### 9.5 Checks

* `python -m pytest`: 169 tests pass; `ruff check .` reports no findings.
* `python scripts/verify_cec_backends.py` (`results/reduced/cec_backend_check.md`): native
  backend `|f(x*) - f*| <= 1.1e-10` for all CEC2017 functions at D = 10/30/50/100 and all
  CEC2022 functions at D = 10/20; `opfunu` agrees with the reference code on 4 of 82 cases
  (section 6).
* `python scripts/data_checksums.py --check` and `shasum -a 256 -c data/SHA256SUMS`: 385 files
  match the manifest.
* `python scripts/extract_paper_figures.py <PDF>`: 18 figures, every crop inspected.
