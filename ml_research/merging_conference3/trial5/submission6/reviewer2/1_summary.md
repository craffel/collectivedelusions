# 1. Summary of the Paper

## Main Topic and Objective
The paper addresses the challenge of deploying multi-task networks created via weight-space model merging on resource-constrained edge devices. Specifically, it focuses on solving the critical deployment issues of **batch-dependency** and **heterogeneity collapse** that plague existing dynamic model merging methods (e.g., Linear Routers, QWS-Merge) when handling heterogeneous, mixed-task streaming inputs at inference time.

## Proposed Approach: SLD-Merge
The authors introduce **Sparse Low-Rank Dynamic Merging (SLD-Merge)**, a parameter-efficient, batch-independent dynamic weight merging framework. Unlike traditional dynamic model merging that reconstructs heavy dense weight matrices per batch on the fly, SLD-Merge operates sample-by-sample in the activation space. It consists of three primary components:
1. **Offline Task Vector SVD Factorization:** Singular Value Decomposition (SVD) is performed offline on specialized expert task vectors ($V_k = W_k - W_{base}$) to decompose them into lightweight, low-rank adapters ($B_k$ and $A_k$) of rank $r \ll D$ (e.g., $r = 8$). This reduces task-specific storage overhead by over **92.5%**.
2. **Bounded Cosine-Similarity Router:** A spatial token average pools activations into sample representation vectors $z(x)_b$. The router computes the cosine-similarity against task-specific routing basis vectors $\Phi_k$, bounding the scores in $[-1, 1]$ to act as a regularizer.
3. **Top-1 Sparse Gating & Parallel Execution:** For each sample, the router selects only the single best-aligned task adapter (Top-1 hard gating). The sample is processed batch-independently through this adapter in a parallelized PyTorch forward pass, preventing cross-sample leakage and batch-size dependency.
4. **Activation-Space Mean Initialization:** Sets the routing bases $\Phi_k$ to the empirical mean of activations on a tiny unlabeled calibration set, allowing high-quality dynamic routing completely **zero-shot** without gradient-descent calibration.
5. **Autonomous Classification Head Selection:** Computes average routing scores across late layers to autonomously route sample representations to task-specific classification heads without relying on privileged ground-truth labels.

## Key Findings
* **Heterogeneity Collapse and Soft Collapse:** Traditional dynamic merging approaches average routing coefficients across the batch dimension. In heterogeneous streams, this averages opposing signals, degrading performance to the level of static uniform merging (soft collapse, buffered by the pre-trained backbone capacity).
* **Batch-Independence:** SLD-Merge processes samples completely independently, yielding a stable, peak joint accuracy of **63.87%** across all batch sizes $B \in \{1, \dots, 256\}$, outperforming baselines by up to **+8.50%**.
* **Zero-Shot Efficacy:** The proposed zero-shot Activation-Space Mean Initialization achieves **63.87%** joint accuracy, recovering **99.5%** of the performance of a Straight-Through Estimator (STE) optimized router (**64.16%**).
* **High Efficiency:** SLD-Merge reduces task-specific parameter storage overhead by over **92.5%** (requiring only 0.295M additional parameters instead of 3.96M for 4 experts) and adds only **8.3%** computational overhead (FLOPs) at inference time.

## Explicitly Claimed Contributions and Evidence
1. **Characterization of Heterogeneity Collapse:** Documented in Figure 1 and Table 1, showing how traditional dynamic merging methods flatline at static uniform merging accuracy as batch size increases.
2. **SLD-Merge Framework:** Formulated mathematically in Section 3 and validated empirically in Section 4.2 and Table 1, achieving stable performance across all batch sizes.
3. **Activation-Space Mean Initialization:** Proved in the Ablation Studies (Section 4.4) to perform nearly identically to a gradient-calibrated router.
4. **Empirical Evaluation on a 4-Dataset Benchmark:** Demonstrated across MNIST, FashionMNIST, CIFAR-10, and SVHN on a Vision Transformer (ViT-Tiny) backbone, showing state-of-the-art results and extreme resource savings (detailed in Tables 1 and 2).
5. **Autonomous Head Selection:** Validated in Section 4.4, achieving **62.99%** accuracy (within **98.6%** of the oracle head baseline at **63.87%**) without leaking task labels.
