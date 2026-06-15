# Layer-Decoupled Stateful Kinetics (LDS-Kinetics) Experimental Results

This document contains the verified quantitative performance of LDS-Kinetics against baseline model merging frameworks. Evaluated across 5 random seeds inside the 14-layer, 192-dimensional Analytical Coordinate Sandbox.

## Orthogonal Manifolds

| Method | Homogeneous Acc (%) | Homogeneous Jitter | Heterogeneous Acc (%) | Heterogeneous Jitter |
| :--- | :---: | :---: | :---: | :---: |
| Expert Oracle | 66.46% ± 0.10% | 0.0302 ± 0.0000 | 67.10% ± 3.85% | 1.4894 ± 0.0734 |
| Uniform Merging (Static) | 66.10% ± 0.08% | 0.0000 ± 0.0000 | 66.73% ± 3.83% | 0.0000 ± 0.0000 |
| SABLE (Raw) | 66.33% ± 0.08% | 0.5059 ± 0.0218 | 66.98% ± 3.87% | 1.1801 ± 0.0426 |
| SABLE (SEP) | 66.27% ± 0.09% | 0.7767 ± 0.0375 | 66.94% ± 3.88% | 1.4073 ± 0.0644 |
| Static Layer-Wise Decay | 66.26% ± 0.08% | 0.2530 ± 0.0109 | 66.90% ± 3.85% | 0.5900 ± 0.0213 |
| Static Block-Wise Constant | 66.27% ± 0.08% | 0.2760 ± 0.0119 | 66.91% ± 3.86% | 0.6437 ± 0.0232 |
| Stateless PAC-ZCA | 66.33% ± 0.09% | 0.4525 ± 0.0250 | 66.97% ± 3.87% | 1.1849 ± 0.0568 |
| Heuristic ChemMerge | 66.30% ± 0.08% | 0.6054 ± 0.0251 | 66.96% ± 3.87% | 1.1717 ± 0.0452 |
| Global PAC-Kinetics (Trial 9) | 66.22% ± 0.12% | 0.1873 ± 0.0937 | 66.73% ± 3.90% | 0.8002 ± 0.1498 |
| Stateful ERM Global (M=1) | 66.22% ± 0.12% | 0.1877 ± 0.0946 | 66.69% ± 3.90% | 0.7315 ± 0.1677 |
| LDS-Kinetics (Tri-Block, M=3) | 66.22% ± 0.12% | 0.1873 ± 0.0915 | 66.77% ± 3.89% | 0.8740 ± 0.1245 |
| LDS-Kinetics (Fully Decoupled, M=11) | 66.22% ± 0.12% | 0.1897 ± 0.0842 | 66.79% ± 3.88% | 0.9269 ± 0.0998 |
| Decoupled ERM (Tri-Block, M=3) | 66.22% ± 0.12% | 0.1877 ± 0.0946 | 66.69% ± 3.90% | 0.7315 ± 0.1677 |
| Decoupled ERM (Tri-Block, M=3, Symmetry-Broken) | 66.22% ± 0.12% | 0.1870 ± 0.0951 | 66.68% ± 3.90% | 0.7292 ± 0.1680 |
| Decoupled ERM (Fully Decoupled, M=11) | 66.22% ± 0.12% | 0.1877 ± 0.0946 | 66.69% ± 3.90% | 0.7315 ± 0.1677 |
| Decoupled ERM (Fully Decoupled, M=11, Symmetry-Broken) | 66.22% ± 0.12% | 0.1879 ± 0.0951 | 66.68% ± 3.91% | 0.7300 ± 0.1695 |

## Overlapping Manifolds

| Method | Homogeneous Acc (%) | Homogeneous Jitter | Heterogeneous Acc (%) | Heterogeneous Jitter |
| :--- | :---: | :---: | :---: | :---: |
| Expert Oracle | 66.46% ± 0.10% | 0.0302 ± 0.0000 | 67.10% ± 3.85% | 1.4894 ± 0.0734 |
| Uniform Merging (Static) | 66.16% ± 0.09% | 0.0000 ± 0.0000 | 66.79% ± 3.83% | 0.0000 ± 0.0000 |
| SABLE (Raw) | 66.34% ± 0.09% | 0.5008 ± 0.0231 | 66.99% ± 3.86% | 1.1362 ± 0.0222 |
| SABLE (SEP) | 66.30% ± 0.09% | 0.7985 ± 0.0327 | 66.96% ± 3.87% | 1.3956 ± 0.0240 |
| Static Layer-Wise Decay | 66.29% ± 0.09% | 0.2504 ± 0.0115 | 66.93% ± 3.85% | 0.5681 ± 0.0111 |
| Static Block-Wise Constant | 66.30% ± 0.09% | 0.2732 ± 0.0126 | 66.94% ± 3.85% | 0.6197 ± 0.0121 |
| Stateless PAC-ZCA | 66.34% ± 0.08% | 0.4384 ± 0.0192 | 66.99% ± 3.86% | 1.1272 ± 0.0402 |
| Heuristic ChemMerge | 66.32% ± 0.09% | 0.6237 ± 0.0290 | 66.98% ± 3.87% | 1.1612 ± 0.0068 |
| Global PAC-Kinetics (Trial 9) | 66.25% ± 0.08% | 0.1379 ± 0.0327 | 66.81% ± 3.86% | 0.8460 ± 0.0577 |
| Stateful ERM Global (M=1) | 66.25% ± 0.08% | 0.1376 ± 0.0331 | 66.80% ± 3.86% | 0.8315 ± 0.0549 |
| LDS-Kinetics (Tri-Block, M=3) | 66.25% ± 0.08% | 0.1391 ± 0.0319 | 66.82% ± 3.86% | 0.8651 ± 0.0613 |
| LDS-Kinetics (Fully Decoupled, M=11) | 66.25% ± 0.08% | 0.1435 ± 0.0296 | 66.84% ± 3.86% | 0.8997 ± 0.0672 |
| Decoupled ERM (Tri-Block, M=3) | 66.25% ± 0.08% | 0.1376 ± 0.0331 | 66.80% ± 3.86% | 0.8315 ± 0.0549 |
| Decoupled ERM (Tri-Block, M=3, Symmetry-Broken) | 66.25% ± 0.08% | 0.1352 ± 0.0330 | 66.80% ± 3.86% | 0.8310 ± 0.0550 |
| Decoupled ERM (Fully Decoupled, M=11) | 66.25% ± 0.08% | 0.1376 ± 0.0331 | 66.80% ± 3.86% | 0.8315 ± 0.0549 |
| Decoupled ERM (Fully Decoupled, M=11, Symmetry-Broken) | 66.25% ± 0.08% | 0.1370 ± 0.0334 | 66.80% ± 3.86% | 0.8321 ± 0.0547 |

