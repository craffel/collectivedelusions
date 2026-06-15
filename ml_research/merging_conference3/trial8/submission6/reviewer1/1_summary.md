# Evaluation Part 1: Summary of the Paper

## Main Topic
The paper addresses the challenge of **dynamic model serving and ensembling** for parameter-efficient fine-tuning (PEFT) experts, specifically Low-Rank Adaptation (LoRA) adapters. When serving multiple specialized tasks with a single frozen foundation model, the system must dynamically route each incoming query to the corresponding task expert on-the-fly. Current methods often introduce substantial systems and algorithmic complexity (such as high-dimensional centroids, multi-stage calibration, and Expectation-Maximization-fitted Gaussian Mixture Models). This paper proposes a return to mathematical simplicity by utilizing the intrinsic geometry of the low-rank adapter weights.

## Proposed Approach: LoRA Subspace Projection Routing (LSPR)
LSPR is a co-designed joint training-and-routing framework that operates in two main stages:
1. **Training (Joint Classification-Reconstruction):** Individual adapters are trained using a joint loss objective combining standard classification cross-entropy and a lightweight subspace autoencoding reconstruction loss ($\mathcal{L} = \mathcal{L}_{\text{classification}} + \lambda \mathcal{L}_{\text{reconstruction}}$). This autoencoding constraint mathematically guides the down-projection weight matrix $A_k$ to span the task's activation subspace.
2. **Offline Initialization:** The system performs a microsecond-level, closed-form QR decomposition of the first adapter block's down-projection matrix ($A_k = Q_k R_k$) to extract an orthonormal basis $Q_k$ representing the task's intrinsic representational subspace.
3. **Online Routing (Subspace Energy Routing):** The shared backbone executes task-agnostically up to an early routing block (e.g., Block 4). Early-stage activation vectors $h_b$ are projected onto the task-specific orthonormal bases ($Q_k$). The scale-invariant alignment score $u_{k, b} = \frac{\|h_b Q_k\|_2}{\|h_b\|_2}$ (the cosine of the angle between the activation and the task subspace) is computed.
4. **OOD Rejection and Ensembling:** If the maximum alignment score is below a threshold $\gamma_{\text{OOD}}$, the query is flagged as out-of-distribution (OOD) and bypasses the adapters entirely. Otherwise, the scores are softmax-scaled to derive dynamic ensembling coefficients ($\alpha_{k, b}$), which are then frozen and reused to blend activations across all subsequent layers in a single parallel forward pass.
5. **Extensions:** 
   - **Post-Hoc Warm Alignment:** A localized post-hoc adaptation phase that fine-tunes only $A_k$ of the first adapter layer while freezing $B_k$ and other parameters to align public, unaligned adapters.
   - **Sparse-LSPR:** A Top-$M$ sparse gating mechanism that computes projection scores for all $K$ experts but only executes the top $M$ experts, decoupling serving latency from the registry size.

## Key Findings and Empirical Evidence
The framework is evaluated within a fully-trained PyTorch multi-task environment called the **Isolating Coordinate Sandbox (ICS)**:
- **Accuracy Recovery:** Under both homogeneous and heterogeneous streams, LSPR recovers **85.81% Joint Mean Accuracy**, perfectly matching the Expert Ceiling and SPS-ZCA SOTA, while exceeding Uniform Merging by 61.85%.
- **Immunity to Heterogeneity Collapse:** In highly mixed heterogeneous batches, classic parametric routers (e.g., Linear Router, QWS-Merge) experience "heterogeneity collapse" where ensembling coefficients average to a uniform blend, plummeting accuracy to 23.96%. LSPR maintains flat, robust accuracy (85.81%) by performing sample-specific parallel ensembling.
- **OOD Detection:** LSPR's zero-shot projection energy detects out-of-distribution tasks with an outstanding **AUROC of 0.9755**, outperforming classification-head similarity thresholds (SABLE) and matching EM-fitted GMM density models (SPS-ZCA).
- **Physical Latency Scaling:** On resource-constrained edge CPUs, LSPR delivers highly efficient, flat physical latency scaling by loading weights exactly once and executing a single parallel pass, avoiding the linear latency scaling of sequential micro-batching (MBH).

## Explicitly Claimed Contributions
1. **Mathematical Simplicity:** Replacing highly over-engineered pipelines (such as offline calibration splits, IDC dispersion metrics, and EM-fitted parametric GMMs) with a single, elegant QR decomposition and projection.
2. **Robust Domain-Shifted Performance:** Recovering 85.81% Joint Mean accuracy under continuous representation overlap with zero trainable parameters and zero task-specific calibration data at deployment.
3. **Head-Free, Zero-Shot OOD Rejection:** Outperforming or matching parametric density models in detecting out-of-distribution tasks with an AUROC of 0.9755.
4. **Computational Efficiency:** Flat, highly efficient physical execution latency on host CPUs by serving mixed batches in a single parallel pass.
