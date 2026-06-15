# Empirical Evaluation Results: PAC-Kinetics vs. Baselines

This document provides the complete, rigorous empirical evaluation of **PAC-Kinetics** compared to standard ensembling and dynamic routing baselines inside our Analytical Coordinate Sandbox (ICS).

## Orthogonal Manifold Configurations

### Homo Batch Stream Serving (B=16, 5 seeds)

| Method | Joint Accuracy (%) | Routing Jitter |
| :--- | :---: | :---: |
| oracle | 95.04% &plusmn; 0.02% | 0.0060 &plusmn; 0.0000 |
| uniform | 32.68% &plusmn; 0.02% | 0.0000 &plusmn; 0.0000 |
| sable_raw | 93.76% &plusmn; 0.15% | 0.0697 &plusmn; 0.0049 |
| sable_sep | 87.53% &plusmn; 0.18% | 0.1132 &plusmn; 0.0023 |
| pac_zca | 94.03% &plusmn; 0.37% | 0.0617 &plusmn; 0.0160 |
| chemmerge | 94.56% &plusmn; 0.02% | 0.0060 &plusmn; 0.0001 |
| stateful_erm | 95.03% &plusmn; 0.02% | 0.0063 &plusmn; 0.0003 |
| pac_kinetics | 95.02% &plusmn; 0.03% | 0.0062 &plusmn; 0.0001 |
| pac_kinetics_rand | 33.67% &plusmn; 4.74% | 0.0260 &plusmn; 0.0211 |

### Hetero Batch Stream Serving (B=16, 5 seeds)

| Method | Joint Accuracy (%) | Routing Jitter |
| :--- | :---: | :---: |
| oracle | 95.04% &plusmn; 0.02% | 1.4879 &plusmn; 0.0142 |
| uniform | 32.68% &plusmn; 0.02% | 0.0000 &plusmn; 0.0000 |
| sable_raw | 93.76% &plusmn; 0.15% | 1.4790 &plusmn; 0.0135 |
| sable_sep | 87.53% &plusmn; 0.18% | 1.3968 &plusmn; 0.0129 |
| pac_zca | 94.03% &plusmn; 0.37% | 1.4794 &plusmn; 0.0136 |
| chemmerge | 70.59% &plusmn; 0.75% | 0.9130 &plusmn; 0.0302 |
| stateful_erm | 87.14% &plusmn; 5.47% | 1.3784 &plusmn; 0.0355 |
| pac_kinetics | 91.59% &plusmn; 0.70% | 1.3962 &plusmn; 0.0205 |
| pac_kinetics_rand | 31.03% &plusmn; 2.21% | 0.3856 &plusmn; 0.0828 |

## Overlapping Manifold Configurations

### Homo Batch Stream Serving (B=16, 5 seeds)

| Method | Joint Accuracy (%) | Routing Jitter |
| :--- | :---: | :---: |
| oracle | 95.08% &plusmn; 0.02% | 0.0060 &plusmn; 0.0000 |
| uniform | 37.65% &plusmn; 0.04% | 0.0000 &plusmn; 0.0000 |
| sable_raw | 92.81% &plusmn; 0.45% | 0.1024 &plusmn; 0.0120 |
| sable_sep | 87.04% &plusmn; 0.16% | 0.1145 &plusmn; 0.0052 |
| pac_zca | 93.89% &plusmn; 0.40% | 0.0719 &plusmn; 0.0149 |
| chemmerge | 94.55% &plusmn; 0.11% | 0.0096 &plusmn; 0.0056 |
| stateful_erm | 95.07% &plusmn; 0.02% | 0.0066 &plusmn; 0.0004 |
| pac_kinetics | 95.07% &plusmn; 0.02% | 0.0063 &plusmn; 0.0001 |
| pac_kinetics_rand | 33.93% &plusmn; 4.19% | 0.0305 &plusmn; 0.0106 |

### Hetero Batch Stream Serving (B=16, 5 seeds)

| Method | Joint Accuracy (%) | Routing Jitter |
| :--- | :---: | :---: |
| oracle | 95.08% &plusmn; 0.02% | 1.4879 &plusmn; 0.0142 |
| uniform | 37.65% &plusmn; 0.04% | 0.0000 &plusmn; 0.0000 |
| sable_raw | 92.81% &plusmn; 0.45% | 1.4690 &plusmn; 0.0107 |
| sable_sep | 87.04% &plusmn; 0.16% | 1.3515 &plusmn; 0.0133 |
| pac_zca | 93.89% &plusmn; 0.40% | 1.4769 &plusmn; 0.0112 |
| chemmerge | 70.49% &plusmn; 0.93% | 0.9052 &plusmn; 0.0400 |
| stateful_erm | 83.04% &plusmn; 1.54% | 1.2611 &plusmn; 0.0336 |
| pac_kinetics | 93.12% &plusmn; 0.52% | 1.4202 &plusmn; 0.0096 |
| pac_kinetics_rand | 31.95% &plusmn; 2.30% | 0.2675 &plusmn; 0.0740 |

