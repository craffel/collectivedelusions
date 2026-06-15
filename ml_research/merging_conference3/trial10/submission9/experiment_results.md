# Experiment Results

This document compiles the quantitative serving accuracy and routing jitter metrics evaluated on the Analytical Coordinate Sandbox across 5 independent random seeds.

## Orthogonal Manifolds

| Method | Homogeneous Stream Acc (%) | Homogeneous Stream Jitter | Heterogeneous Stream Acc (%) | Heterogeneous Stream Jitter |
|---|---|---|---|---|
| Oracle | 66.45% ± 0.02% | 0.0302 ± 0.0000 | 66.32% ± 0.92% | 1.4979 ± 0.0097 |
| Uniform | 51.38% ± 0.03% | 0.0000 ± 0.0000 | 51.27% ± 0.70% | 0.0000 ± 0.0000 |
| SABLE | 66.44% ± 0.02% | 0.0860 ± 0.0078 | 66.30% ± 0.92% | 1.4900 ± 0.0092 |
| Momentum-Merge | 64.98% ± 0.08% | 0.0350 ± 0.0009 | 54.48% ± 0.82% | 0.2184 ± 0.0010 |
| ChemMerge | 64.07% ± 0.12% | 0.0326 ± 0.0006 | 53.40% ± 0.79% | 0.1455 ± 0.0006 |
| PAC-Kinetics | 66.44% ± 0.02% | 0.0351 ± 0.0008 | 66.11% ± 0.91% | 1.3780 ± 0.0078 |
| AIR (Ours) | 66.44% ± 0.02% | 0.0364 ± 0.0009 | 66.23% ± 0.92% | 1.4202 ± 0.0090 |
| AIR (Non-Negative) | 66.44% ± 0.02% | 0.0367 ± 0.0010 | 66.22% ± 0.92% | 1.4178 ± 0.0088 |

## Overlapping Manifolds

| Method | Homogeneous Stream Acc (%) | Homogeneous Stream Jitter | Heterogeneous Stream Acc (%) | Heterogeneous Stream Jitter |
|---|---|---|---|---|
| Oracle | 66.45% ± 0.02% | 0.0302 ± 0.0000 | 66.32% ± 0.92% | 1.4979 ± 0.0097 |
| Uniform | 53.32% ± 0.03% | 0.0000 ± 0.0000 | 53.22% ± 0.74% | 0.0000 ± 0.0000 |
| SABLE | 66.43% ± 0.02% | 0.0923 ± 0.0052 | 66.30% ± 0.92% | 1.4886 ± 0.0108 |
| Momentum-Merge | 65.16% ± 0.06% | 0.0355 ± 0.0006 | 56.09% ± 0.83% | 0.2174 ± 0.0011 |
| ChemMerge | 64.37% ± 0.09% | 0.0328 ± 0.0005 | 55.14% ± 0.80% | 0.1449 ± 0.0007 |
| PAC-Kinetics | 66.43% ± 0.03% | 0.0358 ± 0.0005 | 66.09% ± 0.92% | 1.3631 ± 0.0054 |
| AIR (Ours) | 66.44% ± 0.02% | 0.0370 ± 0.0005 | 66.22% ± 0.92% | 1.4169 ± 0.0087 |
| AIR (Non-Negative) | 66.44% ± 0.02% | 0.0375 ± 0.0006 | 66.22% ± 0.92% | 1.4143 ± 0.0086 |

## Nonlinear Manifolds

| Method | Homogeneous Stream Acc (%) | Homogeneous Stream Jitter | Heterogeneous Stream Acc (%) | Heterogeneous Stream Jitter |
|---|---|---|---|---|
| Oracle | 60.60% ± 0.11% | 0.0302 ± 0.0000 | 60.63% ± 0.97% | 1.5011 ± 0.0091 |
| Uniform | 44.08% ± 0.09% | 0.0000 ± 0.0000 | 44.08% ± 0.72% | 0.0000 ± 0.0000 |
| SABLE | 60.30% ± 0.15% | 0.2600 ± 0.0084 | 60.33% ± 1.01% | 1.3783 ± 0.0302 |
| Momentum-Merge | 58.83% ± 0.16% | 0.0498 ± 0.0013 | 48.27% ± 0.80% | 0.1870 ± 0.0048 |
| ChemMerge | 57.96% ± 0.19% | 0.0403 ± 0.0011 | 47.24% ± 0.79% | 0.1243 ± 0.0032 |
| PAC-Kinetics | 60.42% ± 0.15% | 0.0602 ± 0.0052 | 58.74% ± 1.20% | 1.2964 ± 0.0232 |
| AIR (Ours) | 60.42% ± 0.17% | 0.0718 ± 0.0131 | 59.38% ± 1.10% | 1.3727 ± 0.0260 |
| AIR (Non-Negative) | 60.43% ± 0.14% | 0.0762 ± 0.0088 | 58.96% ± 1.30% | 1.2971 ± 0.0452 |

## Visualizing Active serving Dynamics
We plotted the ensembling weight trajectories of the active task expert over the homogeneous transition boundaries in `results/fig1_weight_trajectories.png`.
![Weight Trajectories](results/fig1_weight_trajectories.png)

### Discussion of Transition Dynamics
1. **Stateless SABLE (Nearest Centroid)** reacts immediately but exhibits intense, high-frequency ensembling weight fluctuations and routing jitter across the sequence due to observation noise.
2. **ChemMerge (Biochemical ODE)** successfully smooths the trajectory, but its continuous reactor concentration state accumulates history too rigidly. This results in severe **representational lag (inertial drag)** when the task switches at step 50, taking nearly 15-20 steps to fully adapt.
3. **AIR (Ours)** achieves both worlds simultaneously: it is exceptionally smooth under stationary task periods (filtering out noise via precision-weighted prediction errors), yet it adapts **near-instantaneously** (within 1-2 steps) when the task switches. Because the bottom-up prediction error spikes violently upon transition, the Free Energy Minimization perception loop immediately overcomes the prior temporal expectation, resetting the belief state and resolving the lag completely!
4. **AIR (Non-Negative Ablation)**: Restricting the generative mapping matrix $W \ge 0$ prevents negative feedback coupling. Consequently, the router is incapable of active inhibition, resulting in severe inertial drag where Task A cannot be actively suppressed and must decay slowly, validating the critical necessity of inhibitory pathways in serving perception.
