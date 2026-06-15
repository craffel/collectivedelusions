# Evaluation Phase 1: Summary of the Paper

## Main Topic and Motivation
The paper addresses the challenge of on-the-fly routing and dynamic ensembling for multiple Low-Rank Adaptation (LoRA) experts in parameter-efficient fine-tuning (PEFT) frameworks. As multi-task serving proliferates, existing dynamic merging and routing systems introduce substantial algorithmic and engineering complexity. This complexity includes the use of offline calibration datasets, classification-head dependencies, unit-norm/dispersion calibration parameters, and parametric Gaussian Mixture Models (GMMs) for out-of-distribution (OOD) rejection. 

The authors advocate for a "return to mathematical simplicity" based on the principle of Occam's razor. They argue that the intrinsic geometric properties of the low-rank weights themselves are sufficient to resolve routing and OOD detection on-the-fly without the need for auxiliary parametric gating models or classification heads.

## Proposed Approach: LoRA Subspace Projection Routing (LSPR)
LSPR is a co-designed joint training-and-routing framework that operates in two primary stages:
1. **Offline Stage (Subspace Extraction):** For each task-specific expert $k$, a standard QR decomposition is performed on the first adapter layer's down-projection matrix $A_k \in \mathbb{R}^{D \times r}$, yielding $A_k = Q_k R_k$. Here, $Q_k \in \mathbb{R}^{D \times r}$ is a semi-orthogonal matrix ($Q_k^T Q_k = I_r$) representing an orthonormal basis for the task's intrinsic representational subspace.
2. **Online Stage (Subspace Energy Routing - SER):** 
   - Activations are executed task-agnostically up to an early routing block ($L_{\text{route}} = 3$).
   - The early activation vector $h_b \in \mathbb{R}^{1 \times D}$ for each sample $b$ is projected orthogonally onto the representational subspaces: $P_k(h_b) = h_b Q_k Q_k^T$.
   - A scale-invariant subspace alignment score is computed as:
     $$u_{k, b} = \frac{\| h_b Q_k \|_2}{\| h_b \|_2}$$
     which corresponds to the cosine of the angle between the activation vector $h_b$ and the $r$-dimensional subspace spanned by $Q_k$.
   - **Zero-Shot, Head-Free OOD Rejection:** If $\max_j u_{j, b} < \gamma_{\text{OOD}}$, the query is flagged as OOD and executed solely by the base backbone.
   - **Dynamic Blending:** For in-distribution queries, ensembling coefficients are derived using a temperature-scaled Softmax:
     $$\alpha_{k, b} = \frac{\exp(u_{k, b} / \tau)}{\sum_{j=1}^K \exp(u_{j, b} / \tau)}$$
     where $\tau$ is a routing temperature (default $\tau = 0.01$).
   - **Single-Pass Parallel Execution:** Dynamic blending of parallel expert activations is executed in a single parallel pass without micro-batching.

### Crucial Co-Design: Joint Loss Objective
The authors emphasize that standard LoRA fine-tuning does not align the column space of $A_k$ with the activation distribution because $B_k$ is initialized to zero and the gradient updates to $A_k$ are negligible. To resolve this representational disconnect, LSPR employs a joint training objective for the first adapter block:
$$\mathcal{L} = \mathcal{L}_{\text{classification}} + \lambda \mathcal{L}_{\text{reconstruction}}$$
where the reconstruction loss is a subspace autoencoding objective:
$$\mathcal{L}_{\text{reconstruction}} = \frac{1}{B} \sum_{b=1}^B \| h_b - h_b A_k B_k \|_2^2$$
This constraint guides the column space of $A_k$ to converge to the principal components of the activation distribution of task $k$.

## Explicitly Claimed Contributions and Reported Evidence
1. **Mathematical Simplicity:** LSPR replaces complex pipelines (UNC, IDC, GMMs, EM parameter-fitting, classification-head routing) with a simple closed-form QR decomposition and projection. 
2. **Robust Multi-Task Performance under Domain Shifts:** In a fully-trained PyTorch synthetic sandbox environment (the *Isolating Coordinate Sandbox*), LSPR recovers **85.81% Joint Mean accuracy** under both homogeneous and heterogeneous streams, perfectly recovering the expert ceiling and matching the state-of-the-art SPS-ZCA while requiring zero offline calibration data.
3. **Immunity to Heterogeneity Collapse:** In mixed heterogeneous batches, classic parametric routers collapse to uniform ensembling (23.96% accuracy) due to batch-level coefficient averaging. LSPR computes sample-specific coefficients inside a parallel forward pass, maintaining 85.81% accuracy.
4. **Head-Free, Zero-Shot OOD Rejection:** LSPR's projection score detects OOD queries with an **AUROC of 0.9755** in the sandbox environment, matching parametric GMMs without fitting any parametric density model.
5. **Serving Efficiency / Flat Latency Scaling:** Wall-clock CPU latency profiling shows that LSPR completely avoids sequential micro-batching and multiple weight-reloads, maintaining highly efficient physical execution and flat latency scaling.
6. **Workflow Solutions (Warm Alignment and Sparse-LSPR):**
   - **Post-Hoc Warm Alignment:** Enables compatibility with standard public adapters by freezing $B_k$ and fine-tuning $A_k$ with the reconstruction loss for 50--100 steps on representative queries. This is shown to increase Task 0's alignment score from 0.0975 to 0.4076, recovering Joint Mean Accuracy to 66.02%.
   - **Sparse-LSPR:** Top-$M$ gating matches full LSPR performance (85.81% accuracy) while achieving flat serving latency scaling with respect to expert registry size $K$.
