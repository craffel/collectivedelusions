# 1. Summary of the Paper

## Main Topic
The paper addresses the challenge of dynamically routing incoming queries to task-specific parameter-efficient fine-tuning (PEFT) adapters, specifically Low-Rank Adaptation (LoRA) experts, when serving multiple tasks on a shared foundation model backbone. It aims to solve both the task-routing problem and the out-of-distribution (OOD) rejection problem training-free at deployment, without relying on complex offline calibration, parameter-heavy density models (like GMMs), or classification-head similarities.

## Proposed Approach: LoRA Subspace Projection Routing (LSPR)
LSPR is a co-designed joint training-and-routing framework that operates in two stages:
1. **Training (Co-design Phase):** Task-specific LoRA adapters are trained using a joint classification and reconstruction (subspace autoencoding) objective:
   $$\mathcal{L} = \mathcal{L}_{\text{classification}} + \lambda \mathcal{L}_{\text{reconstruction}}$$
   where:
   $$\mathcal{L}_{\text{reconstruction}} = \frac{1}{B} \sum_{b=1}^B \|h_b - h_b A_k B_k\|_2^2$$
   This reconstruction objective forces the down-projection matrix $A_k \in \mathbb{R}^{D \times r}$ to span the principal components of the task's activation distribution.
2. **Offline Initialization (Microsecond QR):** A standard, closed-form QR decomposition is performed on the first-adapter block's down-projection matrix $A_k = Q_k R_k$ to extract an orthonormal basis $Q_k$ representing the task's representational subspace.
3. **Online Routing (Subspace Energy Routing & OOD Rejection):**
   - Incoming early activations $h_b$ are projected onto the task-specific subspaces to calculate scale-invariant geometric alignment scores (cosines of the angle between activation and subspace):
     $$u_{k, b} = \frac{\|h_b Q_k\|_2}{\|h_b\|_2}$$
   - Zero-shot, head-free OOD rejection is performed by checking if $\max_j u_{j, b} < \gamma_{\text{OOD}}$. If so, the adapters are bypassed entirely.
   - For in-distribution queries, ensembling coefficients are derived via a temperature-scaled Softmax:
     $$\alpha_{k, b} = \frac{\exp(u_{k, b}/\tau)}{\sum_j \exp(u_{j, b}/\tau)}$$
   - Parallel experts are executed and dynamically blended in a single parallel vectorized forward pass.

## Key Findings & Empirical Claims
- **Classification Accuracy:** In a synthetic "Isolating Coordinate Sandbox (ICS)" PyTorch environment with continuous domain-shifted task overlap, LSPR achieves **85.81% Joint Mean Accuracy**, perfectly recovering the Expert Ceiling and matching SPS-ZCA SOTA while outperforming Uniform Merging (23.96%).
- **Resilience to Heterogeneity Collapse:** Under highly mixed heterogeneous batches, other parametric routers (Linear Router, QWS-Merge) average ensembling coefficients over the batch and collapse to Uniform Merging (23.96%). LSPR maintains 85.81% accuracy because it performs sample-specific ensembling on-the-fly.
- **OOD Rejection:** LSPR achieves an outstanding zero-shot OOD rejection AUROC of **0.9755**, outperforming SABLE and matching EM-fitted GMMs (SPS-ZCA) without fitting any parametric density model.
- **Physical Latency:** Standard micro-batching solutions (such as MBH) scale linearly in latency with batch size on edge CPUs due to sequential forward passes. LSPR maintains flat latency scaling because it serves mixed batches in a single parallel pass.

## Claimed Contributions
1. **Mathematical Simplicity:** Proposes a clean, closed-form linear algebra projection (QR decomposition) that replaces complex data-dependent pipelines (such as offline calibration datasets, GMM parameter fitting, UNC, and IDC).
2. **Robust Multi-Task Performance:** Matches SOTA accuracy and recovering 100% of the Expert Ceiling in a trained PyTorch environment under continuous domain shifts, requiring zero trainable parameters or offline calibration at deployment.
3. **Head-Free OOD Rejection:** Utilizes scale-invariant projection energy to detect out-of-distribution tasks with 0.9755 AUROC, surpassing head-dependent similarity baselines without fitting density models.
4. **Systems and Computational Efficiency:** Enables single-pass vectorized execution layout on CPUs, avoiding DRAM weight-loading and multi-pass overhead, and scaling flatly across mixed batch configurations.
