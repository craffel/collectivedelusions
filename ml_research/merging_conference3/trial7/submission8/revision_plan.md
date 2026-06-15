# Revision Plan: Addressing Twenty-Fourth Round Mock Review Feedback for CGHR + MBH

Following our **Empiricist** persona, we welcome rigorous critiques of our experimental assumptions and systems-level trade-offs. The mock reviewer rated our latest version as a strong **Accept (5)**, praising the scientific rigor, theoretical elegance, and thoroughness of our additions. To achieve absolute top-tier quality, we have updated our revision plan to incorporate and document our responses to the advanced systems-level and memory-capacity trade-off critiques raised by the reviewer:

1.  **Offline Estimation and Caching Cost for SVD Projections (Weakness 1)**:
    *   **Weakness**: The computational complexity of SVD projection operators scales as $\mathcal{O}(D \cdot \min(D^2, N_{\text{act}}^2))$, making on-the-fly calculation prohibitive. Pre-computed projection matrices $P_k \in \mathbb{R}^{D \times D}$ require substantial memory overhead (e.g., 64MB each in FP32 for $D=4096$).
    *   **Plan**: We will expand the SVD subspace discussion in Section 5.1 of `submission/sections/05_conclusion.tex` to perform a concrete memory footprint analysis. We will prove that for $K=64$ experts under LLaMA-7B, storing all matrices requires an additional 4GB of GPU and host RAM. We will propose low-rank projection parameterizations factoring $P_k$ into low-rank components $A_k B_k$ ($A_k \in \mathbb{R}^{D \times r}, B_k \in \mathbb{R}^{r \times D}$ for $r \ll D$) to compress storage to only 3.125% of the full matrix footprint, or sharing projection matrices across expert clusters organized under H-MBH.

2.  **Hardware-Native Parallel Occupancy & Batch Padding Trade-off (Weakness 2)**:
    *   **Weakness**: Mixed execution streams in GPU Segmented-BGEMM kernels suffer from thread-warping divergence and load imbalances under highly skewed task streams (e.g., one task has 250 samples and another has 6), leading to GPU under-occupancy.
    *   **Plan**: We will update Appendix D.1 in `submission/example_paper.tex` to discuss how the serving pipeline can introduce **Batch Padding** (padding smaller micro-batches with zero/dummy inputs to the nearest warp boundary of 32) to ensure balanced thread-block sizing and synchronized execution. We will detail the resulting latency-throughput trade-off: padding increases warp occupancy and lowers peak latency, but slightly reduces maximum throughput by consuming extra GPU FLOPS on dummy elements.

---

## Completed Solutions for Each Critical Flaw (Summary of Modifications)

1.  **`submission/sections/05_conclusion.tex`**:
    *   Added a detailed memory footprint analysis of storing $D \times D$ projection matrices in FP32, showing that LLaMA-7B matrices take 64MB each (totaling 4GB for $K=64$).
    *   Formulated low-rank parameterization ($A_k B_k$) and clustered projection-sharing as advanced mitigations to reduce the storage overhead to $\mathcal{O}(D \cdot r)$ (just 2MB per expert under $r=64$).
2.  **`submission/example_paper.tex`**:
    *   Updated the hardware-native Triton implementation outline in Appendix D.1 to discuss thread-warping divergence and load imbalances under skewed task streams.
    *   Characterized the Batch Padding design pattern and its core throughput-latency trade-off (increasing warp occupancy vs. consuming extra GPU FLOPS on dummy elements).

---

## Forty-Third Round Review Update (Acceptance and Verification)
A fresh iteration of the automated mock critic returned a pristine, top-tier **6: Strong Accept** recommendation with "Excellent" ratings across Soundness, Presentation, Significance, and Originality. The reviewer concluded that our paper is a tour de force, with zero major or minor critical flaws, completely validating our extensive mathematical, empirical, and systems-level enhancements (including semantic class correlation normalization, data-free calibration priors, dynamic homogeneity bypasses, SVD projection sweeps, and LRU caching). No further revisions were requested, and all final deliverables have been fully compiled and synchronized across all deployment targets.
