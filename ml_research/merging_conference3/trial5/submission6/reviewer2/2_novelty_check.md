# 2. Novelty Check and Delta from Prior Work

## Key Novel Aspects of SLD-Merge
1. **Activation-Space Routing vs. Parameter-Space Weight Reconstruction:** 
   Prior dynamic model merging techniques (e.g., AdaMerging, QWS-Merge, Linear Routers) dynamically compute merging coefficients and reconstruct a new, full-scale dense weight matrix in memory for each forward pass. SLD-Merge fundamentally departs from this by shifting dynamic adaptation from parameter-space weight reconstruction to **sample-wise activation-space routing**. 
2. **Combining Offline SVD with Model Merging:**
   While Singular Value Decomposition (SVD) and low-rank matrices are standard in parameter-efficient fine-tuning (PEFT/LoRA), performing offline SVD directly on specialized *expert task vectors* to create a dynamic, post-hoc Mixture-of-Experts (MoE) is a novel application. This avoids the massive pre-training cost of standard MoE models.
3. **Bounded Cosine-Similarity Router:**
   Instead of using unconstrained linear projection layers to compute routing scores, the authors map activation representations onto a bounded spherical cosine-similarity space. This suppresses high-frequency noise and acts as a strong regularizer, facilitating zero-shot calibration.
4. **Activation-Space Mean Initialization:**
   The paper proposes bypassing the non-differentiability of Top-1 hard gating during calibration by initializing the basis vectors $\Phi_k$ to the empirical activation centroids of a small, unlabeled calibration set. This provides high-quality routing completely zero-shot, representing a highly practical advancement.
5. **Autonomous Classification Head Selection:**
   Unlike prior model merging works that assume a privileged "oracle" at test time to route representations to the correct task-specific classification head, SLD-Merge introduces an autonomous layer-averaged routing score rule to select heads completely independently.

## Detailed Delta from Prior Work
* **Static Merging (Task Arithmetic, TIES-Merging, DARE):** Static approaches use flat linear combinations of expert weights. They have no runtime computational overhead but suffer from severe task interference and cannot adapt to dynamic input shifts. *Delta:* SLD-Merge dynamically routes inputs to specialized adapters at runtime.
* **Traditional Dynamic Merging (Linear Router, QWS-Merge):** These compute coefficients per batch and average them across the batch to reconstruct a single dense weight matrix. This causes catastrophic performance collapse in heterogeneous streams (heterogeneity collapse) and prediction shifts based on batch size. *Delta:* SLD-Merge computes sample-wise routing coefficients that are applied element-wise. Each sample is processed completely independently, ensuring perfect batch-independence and zero cross-sample leakage.
* **Mixture of Experts (MoE / Switch Transformer):** MoE models route tokens through specialized feed-forward paths, but are trained from scratch under massive compute budgets. *Delta:* SLD-Merge is a training-free, post-hoc merging framework that consolidates already fine-tuned, fully dense expert models on edge-device budgets.

## Characterization of Novelty
The novelty of SLD-Merge is **highly significant and exceptionally practical**. Rather than introducing complex mathematical abstractions, it cleverly synthesizes SVD decomposition, cosine-similarity routing, and low-rank activation-space parallel execution to solve a major real-world bottleneck (batch-dependency and heterogeneity collapse). From a system deployment standpoint, shifting the dynamic adaptation to activation-space adapters is a brilliant, highly scalable engineering choice.
