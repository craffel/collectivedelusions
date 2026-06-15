# Evaluation Task 3: Soundness and Methodology

## Clarity of Description and Mathematical Rigor
The mathematical exposition is clean, rigorous, and highly detailed. The paper does a commendable job of grounding the heuristic "Fisher-Weighted Cosine Similarity" in information geometry:
- It defines the diagonal empirical Fisher Information coordinate ($F_{k, c, j}$) as the inverse coordinate noise variance under a conditional Gaussian assumption.
- The dual-space alignment between activation coordinate centroids and classification weights under $L_2$-regularized softmax cross-entropy training is formally bounded (Appendix). This is a crucial step that justifies using classifier weights as proxies for activation centroids.
- The asymptotic behavior of the smoothed and power-scaled Fisher regularizer under different extreme limits ($\beta \to \infty, \gamma \to 0$ and $\beta \to 0, \gamma \to 1$) is clearly derived.

## Appropriateness of Methods (Practitioner's Lens)
From a systems and deployment perspective, the choice of **diagonal empirical Fisher Information** is highly appropriate. Estimating full covariance matrices or Kronecker-factored curvature during online streaming inference would be computationally prohibitive and mathematically underdetermined on microscopic calibration splits. dFIM has a linear storage and computational complexity ($O(d)$), taking only **4.05 milliseconds** to compute, making it exceptionally well-suited for low-overhead test-time adaptation.

## Critical Technical Flaws and Deployment Limitations
Despite its theoretical elegance, several significant engineering and scalability challenges limit the practical utility of the proposed methodology in production environments:

1. **The Computational Redundancy of MBH under High Heterogeneity**:
   Micro-Batch Homogenization (MBH) partitions a batch of size $B$ into $G \le K$ homogeneous micro-batches. If a test stream is highly heterogeneous and contains samples from all $K$ expert domains in a single batch, MBH must perform $G = K$ separate, sequential forward passes. 
   From an engineering and FLOPs perspective, executing $K$ sequential forward passes with $K$ distinct merged models is **computationally equivalent to (and practically slower than) simply running the original, unmerged specialized expert models** on their respective samples. Running the original experts is more efficient because it avoids the overhead of dynamic weight assembly, routing coefficient calculation, and scatter-gather operations. Thus, the core computational benefit of test-time model merging is lost in diverse streaming environments unless the active experts are strictly bounded.

2. **Rotated and Non-Axis-Aligned Noise (The Diagonal Assumption Collapse)**:
   Section 4.6 demonstrates that when the noise in the representation space is rotated and correlated, the standard diagonal Fisher formulation collapses and performs worse than standard unweighted cosine similarity. While the authors propose an on-the-fly covariance alignment mechanism (`FIOSR-Online`) that performs an eigenvalue decomposition (EVD) on a shrinkage covariance matrix directly from the calibration split, running a full online EVD during test-time is computationally expensive ($O(d^3)$ complexity) and does not scale to high-dimensional representation spaces (e.g., $d \ge 1024$).

3. **Memory and Storage Overhead for Massive Vocabularies**:
   The class-conditional Fisher matrices require storing $K \times C \times d$ parameters. For modern Large Language Models (LLMs) with vocabulary sizes of $C \approx 32\text{K}$ to $128\text{K}$ and hidden dimensions of $d \approx 4096$, storing these Fisher coefficients introduces substantial storage and memory bandwidth overhead. Although compression strategies (Class-Grouped Pooling, Low-Rank FIM) are proposed in the appendix, they are discussed purely conceptually and lack empirical validation.

4. **Statistical Phase Transition of Calibration Size ($N_c$)**:
   The sensitivity analysis (Appendix) reveals a sharp statistical phase transition. When the calibration split is extremely scarce ($N_c \le 4$), FIOSR performs **significantly worse than the flat baseline** due to highly unstable variance estimators. The method requires a minimum threshold of $N_c \ge 8$ samples per task to stabilize, which limits its applicability in zero-shot or highly dynamic, uncalibrated streaming contexts.

5. **Asymmetrical Alternative-Hypothesis Penalty in CSC**:
   The Class-Size Scaling Calibration (CSC) normalizer $\sqrt{2\log C_k / d}$ is derived under null-hypothesis conditions. However, under alternative-hypothesis conditions (a genuine positive match), it introduces an asymmetrical penalty that unfairly penalizes experts with larger vocabularies, even if they achieve an identical genuine class prototype match as a smaller vocabulary expert.

6. **Closed-World Assumption**:
   The framework assumes all incoming samples belong to one of the $K$ expert domains, offering no safeguards (e.g., OOD rejection thresholds) to handle out-of-distribution data safely.
