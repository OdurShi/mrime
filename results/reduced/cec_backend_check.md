Native backend: largest |f(x*) - f*| per suite and dimension

| suite | D | max abs gap |
|---|---|---|
| cec2017 | 10 | 0 |
| cec2017 | 30 | 0 |
| cec2017 | 50 | 1.82e-11 |
| cec2017 | 100 | 1.09e-10 |
| cec2022 | 10 | 0 |
| cec2022 | 20 | 0 |

opfunu vs native on 5 random points per function: 4 of 82 (suite, D, function) cases agree to 1e-9 relative

| suite | D | function | name | max rel. difference | identical | reference f - f* at opfunu's optimum |
|---|---|---|---|---|---|---|
| cec2017 | 10 | F1 | Shifted and Rotated Bent Cigar | 2.2e-16 | yes | 0 |
| cec2017 | 10 | F2 | Shifted and Rotated Zakharov | 1.0e+00 | no | 7.16e+09 |
| cec2017 | 10 | F3 | Shifted and Rotated Rosenbrock | 8.9e-01 | no | 1.43e+04 |
| cec2017 | 10 | F4 | Shifted and Rotated Rastrigin | 1.3e+02 | no | 329 |
| cec2017 | 10 | F5 | Shifted and Rotated Expanded Schaffer F6 | 3.1e-01 | no | 117 |
| cec2017 | 10 | F6 | Shifted and Rotated Lunacek Bi-Rastrigin | 9.9e+02 | no | 696 |
| cec2017 | 10 | F7 | Shifted and Rotated Non-Continuous Rastrigin | 4.0e-01 | no | 305 |
| cec2017 | 10 | F8 | Shifted and Rotated Levy | 9.8e-01 | no | 1.03e+04 |
| cec2017 | 10 | F9 | Shifted and Rotated Schwefel | 2.1e-01 | no | 3.6e+03 |
| cec2017 | 10 | F10 | Hybrid Function 1 | 3.3e+03 | no | 1.96e+08 |
| cec2017 | 10 | F11 | Hybrid Function 2 | 4.6e+00 | no | 1.7e+10 |
| cec2017 | 10 | F12 | Hybrid Function 3 | 5.1e+00 | no | 4.95e+09 |
| cec2017 | 10 | F13 | Hybrid Function 4 | 5.5e+01 | no | 1.17e+09 |
| cec2017 | 10 | F14 | Hybrid Function 5 | 4.9e+01 | no | 2.31e+09 |
| cec2017 | 10 | F15 | Hybrid Function 6 | 4.0e+07 | no | 1.26e+03 |
| cec2017 | 10 | F16 | Hybrid Function 7 | 1.6e+14 | no | 9.86e+04 |
| cec2017 | 10 | F17 | Hybrid Function 8 | 1.3e+00 | no | 1.99e+10 |
| cec2017 | 10 | F18 | Hybrid Function 9 | 4.5e+10 | no | 2.9e+10 |
| cec2017 | 10 | F19 | Hybrid Function 10 | 2.9e+01 | no | 956 |
| cec2017 | 10 | F20 | Composition Function 1 | 3.2e+01 | no | 0 |
| cec2017 | 10 | F21 | Composition Function 2 | 3.3e-01 | no | 0 |
| cec2017 | 10 | F22 | Composition Function 3 | 1.9e+01 | no | 0 |
| cec2017 | 10 | F23 | Composition Function 4 | 1.7e+01 | no | 0 |
| cec2017 | 10 | F24 | Composition Function 5 | 6.4e-01 | no | 0 |
| cec2017 | 10 | F25 | Composition Function 6 | 7.4e+00 | no | 0 |
| cec2017 | 10 | F26 | Composition Function 7 | 8.9e-01 | no | 0 |
| cec2017 | 10 | F27 | Composition Function 8 | 1.7e+00 | no | 0 |
| cec2017 | 10 | F28 | Composition Function 9 | 3.1e+14 | no | 0 |
| cec2017 | 10 | F29 | Composition Function 10 | 7.9e+08 | no | 0 |
| cec2017 | 30 | F1 | Shifted and Rotated Bent Cigar | 3.2e-16 | yes | 0 |
| cec2017 | 30 | F2 | Shifted and Rotated Zakharov | 1.0e+00 | no | 2.85e+15 |
| cec2017 | 30 | F3 | Shifted and Rotated Rosenbrock | 8.0e-01 | no | 7.25e+04 |
| cec2017 | 30 | F4 | Shifted and Rotated Rastrigin | 2.8e+02 | no | 935 |
| cec2017 | 30 | F5 | Shifted and Rotated Expanded Schaffer F6 | 2.9e-01 | no | 167 |
| cec2017 | 30 | F6 | Shifted and Rotated Lunacek Bi-Rastrigin | 2.1e+03 | no | 3.14e+03 |
| cec2017 | 30 | F7 | Shifted and Rotated Non-Continuous Rastrigin | 9.6e-01 | no | 888 |
| cec2017 | 30 | F8 | Shifted and Rotated Levy | 9.9e-01 | no | 7.23e+04 |
| cec2017 | 30 | F9 | Shifted and Rotated Schwefel | 2.9e-01 | no | 1.34e+04 |
| cec2017 | 30 | F10 | Hybrid Function 1 | 8.3e+04 | no | 8.37e+07 |
| cec2017 | 30 | F11 | Hybrid Function 2 | 8.4e-01 | no | 4.92e+10 |
| cec2017 | 30 | F12 | Hybrid Function 3 | 8.3e-01 | no | 1.31e+11 |
| cec2017 | 30 | F13 | Hybrid Function 4 | 5.0e+00 | no | 6.02e+08 |
| cec2017 | 30 | F14 | Hybrid Function 5 | 5.9e+00 | no | 4.32e+10 |
| cec2017 | 30 | F15 | Hybrid Function 6 | 1.8e+07 | no | 2.91e+04 |
| cec2017 | 30 | F16 | Hybrid Function 7 | 3.7e+12 | no | 7.85e+06 |
| cec2017 | 30 | F17 | Hybrid Function 8 | 8.3e+00 | no | 6.98e+09 |
| cec2017 | 30 | F18 | Hybrid Function 9 | 1.5e+09 | no | 5.37e+10 |
| cec2017 | 30 | F19 | Hybrid Function 10 | 2.9e+01 | no | 2.22e+03 |
| cec2017 | 30 | F20 | Composition Function 1 | 8.6e+01 | no | 0 |
| cec2017 | 30 | F21 | Composition Function 2 | 3.5e-01 | no | 0 |
| cec2017 | 30 | F22 | Composition Function 3 | 4.4e+01 | no | 0 |
| cec2017 | 30 | F23 | Composition Function 4 | 2.2e+01 | no | 0 |
| cec2017 | 30 | F24 | Composition Function 5 | 7.0e-01 | no | 0 |
| cec2017 | 30 | F25 | Composition Function 6 | 6.6e+00 | no | 0 |
| cec2017 | 30 | F26 | Composition Function 7 | 4.0e-01 | no | 0 |
| cec2017 | 30 | F27 | Composition Function 8 | 1.7e+00 | no | 0 |
| cec2017 | 30 | F28 | Composition Function 9 | 4.6e+13 | no | 0 |
| cec2017 | 30 | F29 | Composition Function 10 | 2.7e+07 | no | 0 |
| cec2022 | 10 | F1 | Shifted and full Rotated Zakharov | 1.1e+02 | no | 0 |
| cec2022 | 10 | F2 | Shifted and full Rotated Rosenbrock | 6.4e-16 | yes | 0 |
| cec2022 | 10 | F3 | Shifted and full Rotated Expanded Schaffer F6 | 2.9e-01 | no | 0 |
| cec2022 | 10 | F4 | Shifted and full Rotated Non-Continuous Rastrigin | 2.8e-01 | no | 0 |
| cec2022 | 10 | F5 | Shifted and full Rotated Levy | 9.7e-01 | no | 0 |
| cec2022 | 10 | F6 | Hybrid Function 1 | 1.2e+00 | no | 0 |
| cec2022 | 10 | F7 | Hybrid Function 2 | 2.4e+01 | no | 0 |
| cec2022 | 10 | F8 | Hybrid Function 3 | 5.0e+13 | no | 0 |
| cec2022 | 10 | F9 | Composition Function 1 | 4.6e+00 | no | 0 |
| cec2022 | 10 | F10 | Composition Function 2 | 1.8e-01 | no | 0 |
| cec2022 | 10 | F11 | Composition Function 3 | 6.3e+00 | no | 0 |
| cec2022 | 10 | F12 | Composition Function 4 | 5.6e-01 | no | 0 |
| cec2022 | 20 | F1 | Shifted and full Rotated Zakharov | 1.0e+00 | no | 0 |
| cec2022 | 20 | F2 | Shifted and full Rotated Rosenbrock | 5.9e-16 | yes | 0 |
| cec2022 | 20 | F3 | Shifted and full Rotated Expanded Schaffer F6 | 3.0e-01 | no | 0 |
| cec2022 | 20 | F4 | Shifted and full Rotated Non-Continuous Rastrigin | 5.1e-01 | no | 0 |
| cec2022 | 20 | F5 | Shifted and full Rotated Levy | 9.7e-01 | no | 0 |
| cec2022 | 20 | F6 | Hybrid Function 1 | 2.1e+00 | no | 0 |
| cec2022 | 20 | F7 | Hybrid Function 2 | 1.6e+01 | no | 0 |
| cec2022 | 20 | F8 | Hybrid Function 3 | 7.4e+11 | no | 0 |
| cec2022 | 20 | F9 | Composition Function 1 | 8.1e+00 | no | 0 |
| cec2022 | 20 | F10 | Composition Function 2 | 3.4e-01 | no | 0 |
| cec2022 | 20 | F11 | Composition Function 3 | 1.5e+01 | no | 0 |
| cec2022 | 20 | F12 | Composition Function 4 | 3.7e-01 | no | 0 |

