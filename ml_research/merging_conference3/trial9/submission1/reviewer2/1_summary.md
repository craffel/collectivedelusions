# Paper Summary: PAC-Bayesian Smooth Trajectory Merging (PAC-STM)

## Main Topic and Problem Statement
The paper addresses the challenge of **dynamic, sample-by-sample, layer-wise model ensembling (merging)** of task-specific parameter-efficient experts (specifically, Low-Rank Adaptation or LoRA experts) on a shared pre-trained backbone model during multi-task serving. 

While weight-space merging techniques combine experts into a static model, they are task-agnostic and suffer from **Heterogeneity Collapse** (destructive parameter interference under mixed batches of different tasks) and **Vectorization Collapse** (loss of GPU tensor parallelism and batched matrix multiplication when weights are updated dynamically per-sample). Activation-blending techniques (like SABLE or PAC-ZCA) avoid these collapses by dynamically routing and blending activations. 

However, activation-blending methods face a critical bottleneck: calibrating layer-wise routing parameters under ultra-low calibration data regimes (e.g., $N=16$ samples per task) via Empirical Risk Minimization (ERM) leads to severe **transductive overfitting**. This overfitting manifests as wild, high-frequency oscillations (spikes) in temperature/routing parameters across network depth, degrading generalization to unseen out-of-sample test streams.

## Proposed Approach: PAC-STM
To resolve this bottleneck, the paper proposes **PAC-Bayesian Smooth Trajectory Merging (PAC-STM)**. The core ideas include:
1. **Markovian Trajectory Prior ($P$):** Modeling ensembling log-temperatures ($\mathbf{w}_1, \dots, \mathbf{w}_L$) across deep network layers as a continuous autoregressive Gaussian random walk prior.
2. **Analytical Trajectory KL-Divergence:** Deriving and proving a closed-form theorem for the Kullback-Leibler (KL) divergence between the Markovian trajectory posterior $Q$ (with fixed posterior covariance) and the prior $P$.
3. **Deterministic Trajectory Optimization Objective:** Formulating a PAC-Bayesian deterministic training objective (such as Catoni-style linear objective or McAllester-style square-root objective) where the trajectory complexity maps exactly to a first-order finite-difference smoothness penalty. This eliminates the need for unstable stochastic variance optimization or expensive hyperparameter cross-validation under ultra-low data regimes.
4. **Unit-Norm PCA Subspace Projection (UN-PCA-SEP):** A robust preprocessing technique that unit-normalizes early hidden representations and projects them onto task-specific principal component bases, bounding feature coordinates in $[0, 1]$ to suppress heteroscedastic noise.
5. **Non-linear and Skip-aware Extensions:**
   - **UN-KPCA-SEP (Uncentered Kernel PCA):** A non-linear projection mapping onto infinite-dimensional RKHS to handle curved representational manifolds without discarding the crucial centroid identity.
   - **Skip-aware Prior Topologies:** Modeling residual connection dependencies using a DAG/multi-step prior that penalizes both consecutive and skip-level transitions.
   - **Sparse Top-$k$ Activation Blending:** A serving optimization for scalability to large libraries of experts ($K \gg 10$), proven to have bounded representation error.

## Key Findings and Claims (with Evidence)
- **Mitigation of Transductive Overfitting:** Under $N=16$ calibration samples, PAC-STM achieves $73.62\% \pm 1.48\%$ accuracy on mixed heterogeneous batches, outperforming unregularized Temp-Only ERM ($71.57\% \pm 1.50\%$) by **2.05%** in the Orthogonal Sandbox configuration. This pairwise improvement is shown to be highly statistically significant via a paired t-test ($p < 0.008$ across 5 seeds).
- **Absolute Collapse Immunity:** Activation blending in PAC-STM achieves absolute immunity to Heterogeneity Collapse, maintaining high accuracy ($73.15\% \pm 0.79\%$) on single-sample heterogeneous serving streams, whereas weight-space routers (QWS-Merge, Linear Router, PFSR) collapse completely to near-random levels ($\approx 39\%$).
- **Vectorization Retention:** PAC-STM keeps base and adapter weights frozen, allowing batched processing of heterogeneous inputs and avoiding Vectorization Collapse on GPUs.
- **Trajectory Smoothness Qualitative Validation:** Qualitative and quantitative trajectory analyses (illustrated in Figure 2 and verified on ViT-B/16 in Section 4.5) demonstrate that the derived KL regularizer successfully restricts high-frequency oscillations, matching the Oracle's smooth trajectory across network depth.
- **Kernel PCA for Curved Manifolds:** On simulated curved manifolds, uncentered UN-KPCA-SEP outperforms linear PCA by **+6.63%** ($51.98\% \pm 1.82\%$ vs. $45.35\% \pm 0.66\%$), with centered Kernel PCA failing catastrophically ($24.62\% \pm 0.79\%$).
- **Skip-Aware Prior Benefit:** Incorporating residual-skip prior topologies yields a **+1.05%** absolute accuracy gain ($65.70\% \pm 2.15\%$ vs. $64.65\% \pm 0.70\%$) and a 3.33% reduction in trajectory roughness.
