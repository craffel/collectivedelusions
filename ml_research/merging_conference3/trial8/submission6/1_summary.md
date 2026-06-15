# 1. Summary of the Paper: LoRA Subspace Projection Routing (LSPR)

## 1.1 Core Objective and Problem Context
With the proliferation of Parameter-Efficient Fine-Tuning (PEFT) experts, specifically Low-Rank Adaptation (LoRA), serving multiple specialized downstream tasks simultaneously on a single shared foundation model backbone has become a critical operational paradigm. At runtime, the serving framework must dynamically route heterogeneous queries to their corresponding specialized experts and perform robust out-of-distribution (OOD) rejection on-the-fly. 

Prior training-free, zero-shot ensembling and serving methods (such as SPS-ZCA, SABLE, and PFSR) have introduced heavy systems and algorithmic complexity:
- **SPS-ZCA** requires pre-computing high-dimensional centroid statistics from offline calibration splits, Unit-Norm Calibration (UNC), Intra-Task Dispersion Calibration (IDC) variance scaling factors, entropy-dependent temperatures, and Expectation-Maximization-fitted Gaussian Mixture Models (GMMs) for OOD rejection.
- **SABLE and PFSR** depend on calculating cosine similarities to frozen classification-head weights, which are fragile (incompatible with head-free layers, autoregressive decoder blocks, or embedding extraction) and suffer from the **Early-Layer Routing Paradox** (where task-specific adapters must run across all layers before routing can be determined at the head).

To resolve these limitations, this paper proposes **LoRA Subspace Projection Routing (LSPR)**, a minimalist joint training-and-routing framework that leverages the intrinsic geometric properties of LoRA weights and early-layer activations.

---

## 1.2 Proposed Methodology
LSPR co-designs the training and serving phases to achieve mathematical simplicity and systems efficiency. It operates in two main stages:

### A. Offline Stage: Subspace Orthonormal Basis Extraction
1. At model initialization, LSPR retrieves the low-rank down-projection matrix $A_k \in \mathbb{R}^{D \times r}$ from the very first block containing adapters (e.g., Block 4) for each expert $k$.
2. It computes a standard, closed-form QR decomposition:
   $$A_k = Q_k R_k \quad \forall k \in \{1, \dots, K\}$$
   where $Q_k \in \mathbb{R}^{D \times r}$ is a semi-orthogonal matrix ($Q_k^T Q_k = I_r$), whose columns form an orthonormal basis for the column space of $A_k$. This offline step takes microseconds and is performed once.

### B. Online Stage: Subspace Energy Routing (SER) & Blending
1. **Shared Representation Pass:** Run incoming heterogeneous queries through the shared, adapter-free early layers (Blocks 1 to 3, i.e., $L_{\text{route}} = 3$) of the base model to extract early-stage activations $h_b \in \mathbb{R}^{1 \times D}$.
2. **Subspace Projection:** Project the activation vector $h_b$ onto each expert subspace: $P_k(h_b) = h_b Q_k Q_k^T$. Due to $Q_k$'s semi-orthogonality, the L2-norm of the projection simplifies to $\| P_k(h_b) \|_2 = \| h_b Q_k \|_2$.
3. **Scale-Invariant Similarity Score:** Compute the geometric alignment coordinate:
   $$u_{k, b} = \frac{\| h_b Q_k \|_2}{\| h_b \|_2}$$
   representing the exact cosine of the angle between the activation vector $h_b$ and the $r$-dimensional task subspace spanned by $Q_k$.
4. **OOD Shield:** If the maximum score falls below a threshold ($\max_j u_{j, b} < \gamma_{\text{OOD}}$), the sample is flagged as OOD and bypasses all adapters, executing solely on the base model backbone.
5. **Dynamic Activation Blending:** For in-distribution queries, ensembling coefficients $\alpha_{k, b}$ are derived via a sharp, temperature-scaled Softmax:
   $$\alpha_{k, b} = \frac{\exp(u_{k, b} / \tau)}{\sum_{j=1}^K \exp(u_{j, b} / \tau)}$$
   For all subsequent adapter layers, activations are blended on-the-fly inside a single, parallel forward pass:
   $$h_b^{(l)} = h_b^{(l-1)} W_{\text{base}}^{(l)} + \sum_{k=1}^K \alpha_{k, b} \left( h_b^{(l-1)} A_k^{(l)} B_k^{(l)} \right)$$

---

## 1.3 Key Theoretical and Technical Insights
- **The Adapter Sensitivity Theorem:** The authors prove that the magnitude of an adapter's parameter-efficient update is upper-bounded by the projection energy of the input activation vector onto the column space of $A_k$. If the activation is completely orthogonal to the column space of $A_k$, the adapter output is exactly zero, theoretically justifying why weight column spaces align with activation distributions.
- **Joint Training Objective:** Standard LoRA fine-tuning does not align the column space of $A_k$ with the activation manifold because the up-projection $B_k$ is initialized to zero, causing $A_k$'s gradients to remain small. LSPR solves this by introducing a joint classification-reconstruction loss:
   $$\mathcal{L} = \mathcal{L}_{\text{classification}} + \lambda \mathcal{L}_{\text{reconstruction}}$$
   where $\mathcal{L}_{\text{reconstruction}} = \frac{1}{B} \sum_{b=1}^B \| h_b - h_b A_k B_k \|_2^2$ forces the low-rank path to reconstruct the input activations, aligning the columns of $A_k$ with the principal components of the task's activations.
- **Layer-Wise Freezing:** To preserve downstream expert capacity and reduce training overhead, the reconstruction loss is applied solely to the first adapter layer (Block 4). The computed routing coefficients $\alpha_{k, b}$ are frozen and re-used for all downstream layers (Blocks 5 to 12).
- **Workflow Adaptations:**
  - **Post-Hoc Warm Alignment:** Enables compatibility with unaligned public adapters. By freezing $B_k$, subsequent adapters, and classification heads, and only fine-tuning $A_k$ on the reconstruction loss for 50–100 steps, public adapters are aligned in under a minute with 0% downstream degradation.
  - **Split-Rank LoRA:** Partitions the bottleneck rank into dedicated routing and task-specific channels ($r = r_{\text{route}} + r_{\text{task}}$) to completely decouple downstream task performance from the joint autoencoding constraint.
  - **Sparse-LSPR:** Restricts active adapter computation to the Top-$M$ expert pathways ($M \ll K$), decoupling serving latency from the expert registry size $K$.
  - **Anisotropic Calibration:** Resolves representation collapse in high-dimensional spaces by deriving an expected random projection energy under spherical assumptions ($\mathbb{E}[u^2] = r/D$) and adjusting it for practical anisotropy ($\sqrt{r/d_{\text{dom}}}$) using a task-agnostic calibration set.

---

## 1.4 Empirical Results
LSPR is evaluated inside a fully-trained PyTorch multi-task environment (the **Isolating Coordinate Sandbox**) representing domain-shifted task distributions. The main findings are:
1. **Multi-Task Performance:** LSPR achieves **85.81% Joint Mean Accuracy** under both homogeneous and heterogeneous streams, perfectly recovering the Expert Ceiling and matching the SPS-ZCA SOTA (85.94%) while requiring zero trainable parameters, zero classification heads, and zero calibration data.
2. **Resilience to Heterogeneity Collapse:** While classic parametric routers (Linear Router, QWS-Merge) experience "heterogeneity collapse" under mixed batches (dropping to the Uniform merging baseline of 23.96%), LSPR maintains flat, robust accuracy (85.81%).
3. **OOD Rejection:** LSPR's zero-shot projection energy score achieves an outstanding AUROC of **0.9755** for domain shift detection, outperforming SABLE and matching GMM-based density models.
4. **Physical Latency:** By avoiding sequential loops and DRAM weight-reloads required by Micro-Batch Homogenization (MBH) SOTA (139.02 ms), LSPR serving in a single parallel pass runs in **49.46 ms** (a **2.81$\times$ speedup**).
