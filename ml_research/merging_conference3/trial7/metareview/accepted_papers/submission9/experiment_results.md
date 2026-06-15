# SABLE Experimental Evaluation Results

## 1. Executive Summary
We evaluated **SABLE (Sample-wise Activation Blending of Low-Rank Experts)** against key dynamic model merging baselines in our 14-layer, 192-dimensional Analytical Coordinate Sandbox. SABLE completely eliminates heterogeneity collapse natively by blending activations in the forward pass on a per-sample basis, bypassing the need to average coefficients over the batch dimension and avoiding the complex, stateful dynamic sorting/grouping pipeline of Micro-Batch Homogenization (MBH).

## 2. Quantitative Performance Sweep
| Method | Homogeneous Batching (B=256) | Heterogeneous Batching (B=256) | Vectorization/Heterogeneity Collapse |
| :--- | :---: | :---: | :---: |
| **Expert Ceiling** | 78.80% | 78.80% | None |
| **Uniform Merging** | 35.60% | 35.60% | None (Static) |
| **Linear Router (Unreg)** | 55.50% | 54.00% | Severe (Collapse to Uniform) |
| **PFSR (No MBH)** | 71.70% | 56.30% | Severe (Collapse to Uniform) |
| **PFSR + MBH** | 71.70% | 67.20% | Partially Safeguarded (At latency/state cost) |
| **SABLE (Ours, Early Routing)** | 66.60% | 66.60% | Immune (0.00% collapse) |
| **SABLE (Ours, Late Adaptation)** | **68.10%** | **68.10%** | **Immune (0.00% collapse)** |

## 3. Key Findings & Discussion
- **Perfect Heterogeneity Robustness**: SABLE achieves identical, high performance (**68.10%** for Late Adaptation) under both homogeneous and fully heterogeneous streams. It does not suffer from any heterogeneity collapse because ensembling is done per-sample directly in activation space using low-rank LoRA adapters.
- **Bypassing the MBH Stateful Pipeline**: While PFSR+MBH successfully recovers performance in heterogeneous streams (raising accuracy to 67.20%), it requires a complex dynamic sorting and buffering wrapper. SABLE Late Adaptation achieves **68.10%**, which actually *outperforms* the complex PFSR+MBH systems pipeline under heterogeneous streams, while completely stripping away this complex stateful wrapper and executing in exactly a single unified forward pass of the backbone.
- **Minimal Compute Footprint**: By performing activation blending with small-rank ($r=8$) LoRA matrices, SABLE introduces completely negligible overhead while safeguarding the backbone network under extreme domain shift.

## 4. Performance Comparison Visualization
The plot below compares the Joint Mean accuracies of SABLE and standard baselines under both homogeneous and heterogeneous deployment streams.

![Performance Comparison Plot](results/fig1.png)
