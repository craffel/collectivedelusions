# Evaluation Part 4: Experimental Evaluation Check

## Experimental Setup and Datasets
The evaluation is conducted within a controlled, synthetic PyTorch multi-task environment called the **Isolating Coordinate Sandbox (ICS)**:
*   **Backbone:** Frozen linear feature extractor mapping $D_{\text{in}} = 64$ to $D = 192$.
*   **Experts:** $K = 3$ task-specific Low-Rank Adaptation (LoRA) adapters with rank $r = 8$, trained via backpropagation using Adam.
*   **OOD Evaluation:** A 4th domain-shifted task is held out to evaluate out-of-distribution (OOD) rejection performance.
*   **Streams:** Evaluated under homogeneous and highly mixed heterogeneous streams of batch size $B = 256$.

### Major Limitation of the Setup
The primary limitation of this experimental setup is that it is a **simplified synthetic sandbox (proof-of-concept)** rather than a large-scale real-world benchmark (like GLUE on Llama-3 or ImageNet on ViT). A single linear projection layer does not capture the multi-layer, multi-head attention dynamics, residual connections, layer normalization, or token-level routing of modern Transformer architectures.

### Author's Rigor and Scaling Analysis
The authors are remarkably honest and transparent about this limitation in Section 4.1. Rather than hand-waving it, they address the scale gap head-on by analyzing LSPR's geometric properties in high-dimensional spaces using **random projection theory**:
1.  **Spherical Isotropic Scaling:** In their low-dimensional sandbox ($D = 192, r = 8$), the expected squared projection energy of an OOD query is $r/D = 8/192 \approx 0.0417$. In a full-scale LLM like Llama-3-8B ($D = 4096$) with rank $r = 8$, this expected squared score drops drastically to $8/4096 \approx 0.00195$, corresponding to an expected OOD alignment score of only $u \approx 0.0441$. This proves that higher dimensions actually provide **stronger, more distinct geometric separation** than the sandbox.
2.  **Anisotropic Scaling:** Even under severe representation collapse (anisotropy) where activations are concentrated in a dominant subspace of dimension $d_{\text{dom}} = 200$ (a 95% reduction), the expected OOD score is $\sqrt{8/200} \approx 0.20$, which is still extremely low and easily calibrated via their task-agnostic hybrid calibration split.

This mathematical scaling analysis is exceptionally rigorous and successfully bridges the gap between their sandbox proof-of-concept and large-scale deployment.

## Baselines
The paper compares LSPR against an exceptionally strong and representative set of baselines:
1.  **Expert Ceiling:** The absolute upper bound of perfect routing.
2.  **Uniform Merging:** Static weight averaging (the standard fallback).
3.  **Linear Router (Reg):** Labeled data-dependent parametric routing.
4.  **QWS-Merge SOTA:** Trainable dynamic merging SOTA.
5.  **PFSR + MBH SOTA:** Parameter-free, head-dependent projection routing served using Micro-Batch Homogenization to avoid mixed-batch conflicts.
6.  **SPS-ZCA SOTA:** The training-free SOTA utilizing 64-sample calibration splits, UNC, IDC variance calibrations, and EM-fitted GMM density models.

This selection of baselines is comprehensive and covers all major paradigms of PEFT ensembling.

## Do the Results Support the Claims?
Yes, the empirical results provide robust and convincing support for all central claims:

*   **Claim 1: High Accuracy with Zero Parameters/Data:** Supported by Table 1. LSPR achieves **85.81% Joint Mean Accuracy**, perfectly recovering the Expert Ceiling and matching the data-dependent SPS-ZCA SOTA (85.94%), while requiring zero trainable parameters and zero task-specific calibration data.
*   **Claim 2: Immunity to Heterogeneity Collapse:** Supported by Table 1 and Figure 1. Under highly mixed heterogeneous batches, batch-level parametric routers (Linear Router, QWS-Merge) average ensembling coefficients over the batch, causing them to collapse to Uniform Merging (23.96%). LSPR maintains a flat, optimal 85.81% accuracy across all batch configurations due to sample-specific parallel ensembling.
*   **Claim 3: Head-Free, High-Fidelity OOD Rejection:** Supported by Figure 3. LSPR's zero-shot projection energy score achieves an outstanding **AUROC of 0.9755** under continuous feature overlap. This outclasses SABLE and matches or exceeds SPS-ZCA's EM-fitted parametric GMMs without fitting any density model.
*   **Claim 4: Computational and Latency Efficiency:** Supported by Figure 4. Micro-batching (PFSR+MBH) scales linearly in latency due to multiple sequential PyTorch execution passes and weight-reloads. LSPR maintains flat, predictable latency scaling by executing a single parallel pass, delivering massive physical speedups on edge CPUs.

## Thoroughness of Ablations
The ablation studies in this paper are **remarkably comprehensive and deep**:
1.  **Routing Temperature ($\tau$ in Figure 6):** Sweeping $\tau \in [10^{-4}, 1.0]$ shows that a sharp temperature ($\tau \le 0.01$) is critical to enforce crisp routing and prevent multi-expert representation dilution.
2.  **OOD Threshold ($\gamma_{\text{OOD}}$):** Showing that LSPR is highly robust to the choice of $\gamma_{\text{OOD}}$ in the range $[0.30, 0.40]$.
3.  **Necessity of Joint Loss (Failure Mode):** Showing that standard LoRA (cross-entropy alone) has random weight-activation alignment (e.g., $u = 0.0975$), which collapses ensembling accuracy to 19.79%. This verifies the joint classification-reconstruction objective as a vital requirement.
4.  **Verification of Post-Hoc Warm Alignment:** Empirically showing that fine-tuning only $A_k$ of standard LoRA for 60 steps on reconstruction loss increases subspace alignment from 0.0975 to **0.4076** (4.1$\times$ improvement), restoring ensembling accuracy to **66.02%** (3.3$\times$ improvement) with zero downstream capacity loss.
5.  **Verification of Sparse-LSPR:** Showing that Top-2 gating matches full parallel ensembling accuracy (85.81%) while decoupling physical latency from registry size (Figure 5).
6.  **Split-Rank Capacity Trade-offs:** Verifying that Split-Rank LoRA maintains performance within 0.40% of Joint LoRA while preserving high-fidelity subspace alignment of 0.5447 on dedicated routing columns.
7.  **Loss Coefficient ($\lambda$) and Layer-Wise Freezing:** Confirming that applying reconstruction loss solely to Block 4 and freezing coefficients for subsequent layers recovers 100% of the Expert Ceiling and outperforms layer-wise recomputation by 22.66%.
