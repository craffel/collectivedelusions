# PHASE 2: EXPERIMENTAL RESULTS REPORT

## Homogeneous Joint Multi-Task Capabilities (k=14 for Routers)
| Method | MNIST | FashionMNIST | CIFAR-10 | SVHN | Joint Mean |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Uniform Merge (TA) | 58.13 ± 0.00% | 74.01 ± 0.00% | 54.50 ± 0.00% | 73.15 ± 0.00% | **64.95 ± 0.00%** |
| AdaMerging (SOTA Static) | 73.04 ± 0.00% | 76.48 ± 0.00% | 64.06 ± 0.00% | 75.66 ± 0.00% | **72.31 ± 0.00%** |
| Linear Router (Classical) | 99.75 ± 0.01% | 92.60 ± 0.00% | 96.07 ± 0.01% | 96.38 ± 0.01% | **96.20 ± 0.00%** |
| Linear Router (Reg - Ours) | 99.23 ± 0.03% | 92.28 ± 0.01% | 95.46 ± 0.02% | 95.83 ± 0.01% | **95.70 ± 0.01%** |
| QWS-Merge (SOTA Cosine) | 78.29 ± 0.75% | 83.64 ± 0.29% | 73.48 ± 0.19% | 83.42 ± 0.20% | **79.71 ± 0.25%** |
| BL-Router (Ours) | 86.78 ± 0.04% | 87.10 ± 0.01% | 81.59 ± 0.03% | 87.83 ± 0.02% | **85.82 ± 0.01%** |
| BL-Router (Ours - Reg) | 86.72 ± 0.03% | 87.04 ± 0.01% | 81.53 ± 0.04% | 87.76 ± 0.02% | **85.76 ± 0.01%** |
| GLS-Router (Ours) | 99.74 ± 0.01% | 92.60 ± 0.00% | 96.07 ± 0.01% | 96.39 ± 0.01% | **96.20 ± 0.00%** |
| GLS-Router (Ours - Reg) | 99.23 ± 0.03% | 92.28 ± 0.01% | 95.46 ± 0.02% | 95.83 ± 0.01% | **95.70 ± 0.01%** |
| BSigmoid-Router (Ours) | 85.50 ± 0.07% | 86.14 ± 0.08% | 80.36 ± 0.05% | 86.81 ± 0.04% | **84.70 ± 0.03%** |
| BSigmoid-Router (Ours - Reg) | 85.36 ± 0.07% | 86.03 ± 0.07% | 80.21 ± 0.06% | 86.68 ± 0.02% | **84.57 ± 0.03%** |

## Exhaustive Sweep of Hybrid-Router Partition Depth (k)
| Depth (k) | MNIST | FashionMNIST | CIFAR-10 | SVHN | Joint Mean | Latency | Overhead Reduction |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 58.13 ± 0.00% | 74.01 ± 0.00% | 54.50 ± 0.00% | 73.15 ± 0.00% | **64.95 ± 0.00%** | 0.00 ms | 100.0% |
| 1 | 62.49 ± 0.02% | 76.19 ± 0.02% | 58.46 ± 0.01% | 75.43 ± 0.01% | **68.14 ± 0.01%** | 0.75 ms | 92.7% |
| 2 | 66.71 ± 0.03% | 78.18 ± 0.03% | 62.33 ± 0.01% | 77.57 ± 0.02% | **71.20 ± 0.01%** | 1.48 ms | 85.6% |
| 4 | 74.46 ± 0.05% | 81.59 ± 0.05% | 69.58 ± 0.03% | 81.36 ± 0.03% | **76.75 ± 0.02%** | 2.95 ms | 71.3% |
| 12 | 85.66 ± 0.07% | 86.16 ± 0.07% | 80.51 ± 0.05% | 86.84 ± 0.03% | **84.79 ± 0.02%** | 8.81 ms | 14.3% |
| 14 | 85.36 ± 0.07% | 86.03 ± 0.07% | 80.21 ± 0.06% | 86.68 ± 0.02% | **84.57 ± 0.03%** | 10.28 ms | 0.0% |

## Heterogeneous Streaming Benchmark under Noise
| Method | B = 1 | B = 16 | B = 256 |
| :--- | :---: | :---: | :---: |
| Uniform Merge (TA) | 64.98 ± 1.00% | 64.98 ± 1.00% | 65.05 ± 1.30% |
| AdaMerging (SOTA Static) | 72.43 ± 1.40% | 72.43 ± 1.40% | 72.53 ± 1.57% |
| Linear Router (Classical) | 96.00 ± 0.59% | 67.75 ± 1.09% | 63.54 ± 1.32% |
| Linear Router (Reg - Ours) | 95.43 ± 0.60% | 68.33 ± 0.92% | 65.14 ± 1.50% |
| QWS-Merge (SOTA Cosine) | 79.65 ± 1.00% | 67.05 ± 1.29% | 66.13 ± 1.58% |
| BL-Router (Ours) | 85.93 ± 0.87% | 67.92 ± 1.46% | 66.72 ± 1.32% |
| BL-Router (Ours - Reg) | 85.83 ± 0.84% | 67.92 ± 1.49% | 66.76 ± 1.37% |
| GLS-Router (Ours) | 96.00 ± 0.59% | 67.75 ± 1.09% | 63.54 ± 1.32% |
| GLS-Router (Ours - Reg) | 95.43 ± 0.60% | 68.33 ± 0.92% | 65.14 ± 1.50% |
| BSigmoid-Router (Ours) | 84.65 ± 1.04% | 67.80 ± 1.54% | 66.59 ± 1.37% |
| BSigmoid-Router (Ours - Reg) | 84.55 ± 1.04% | 67.83 ± 1.54% | 66.63 ± 1.35% |
| Linear Router (Reg + DBF - Ours) | 95.43 ± 0.60% | 92.48 ± 0.54% | 93.77 ± 1.52% |
| BSigmoid-Router (Reg + DBF - Ours) | 84.50 ± 1.01% | 81.78 ± 1.60% | 83.18 ± 1.77% |

Successfully generated latency-vs-accuracy trade-off plots at 'latency_vs_accuracy.png' and 'results/fig1.png'.

## Calibration Dataset Size (|D_cal|) Ablation Sweep
| Calibration Size (|D_cal|) | k = 4 (Hybrid) | k = 12 (Hybrid) | k = 14 (Fully Dynamic) |
| :---: | :---: | :---: | :---: |
| 64 | 76.75 ± 0.02% | 84.79 ± 0.02% | 84.57 ± 0.03% |
| 256 | 76.80 ± 0.02% | 84.87 ± 0.02% | 84.65 ± 0.02% |
| 512 | 76.81 ± 0.02% | 84.88 ± 0.02% | 84.66 ± 0.02% |
| 1024 | 76.81 ± 0.02% | 84.88 ± 0.02% | 84.67 ± 0.02% |

## Detailed Runtime Latency Breakdown (Wall-clock, microseconds)
Profiling on device: cpu
| Operation Step | Latency (microsec) | Scaling Behavior | Description |
| :--- | :---: | :---: | :--- |
| 1. Feature Pooling & Logit Projection | 7.11 | O(1) | Computes routing logits from H_0 representation |
| 2. Coefficient Sigmoid Scaling | 6.81 | O(K) | Maps logits to independent sigmoidal coefficients |
| 3. Dynamic Weight Reconstruction (per layer) | 759.37 | O(P_layer) | Blends parameters: W_base + sum(alpha_k * V_k) |
| **Total Reconstruction (k = 4)** | 3051.40 | O(1 + K + k * P_layer) | Latency for 4 dynamic layers |
| **Total Reconstruction (k = 12)** | 9126.36 | O(1 + K + k * P_layer) | Latency for 12 dynamic layers |
| **Total Reconstruction (k = 14)** | 10645.10 | O(1 + K + k * P_layer) | Latency for 14 dynamic layers |
