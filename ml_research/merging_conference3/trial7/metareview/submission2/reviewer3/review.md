# Peer Review: Fisher-Information Optimal Subspace Routing (FIOSR)

## Summary of the Paper
This paper proposes **Fisher-Information Optimal Subspace Routing (FIOSR)**, a training-free and parameter-free dynamic ensembling framework for test-time model merging. Existing dynamic routing methods either rely on over-parameterized parametric routers that overfit on tiny calibration sets (the "Dynamic Routing Paradox") and fluctuate wildly under single-sample sequential streams (the "Vectorization Collapse"), or rely on parameter-free projections that assume a flat, isotropic Euclidean weight space. 

FIOSR resolves these issues by modeling the parameter space as a Riemannian manifold. Using the diagonal empirical Fisher Information Matrix (dFIM) computed over a microscopic calibration split, FIOSR constructs a local Riemannian metric tensor that warps the representation space, suppressing noisy dimensions and amplifying highly discriminative task features via a **Fisher-Weighted Cosine Similarity**. Combined with Class-Size Scaling Calibration (CSC) and Micro-Batch Homogenization (MBH), FIOSR bypasses the need for test-time optimization. On a 192-dimensional synthetic Analytical Coordinate Sandbox, FIOSR recovers near-perfect routing stability (~100% MNIST/FashionMNIST) and outperforms unweighted Cosine PFSR by 8.56% and parametric baselines by up to 40.7%. The authors also validate the framework on a simulated LoRA activation space and a physical pre-trained ResNet-18 backbone.

---

## Strengths

1. **Optimization-Free Test-Time Adaptation**: Bypassing test-time gradient optimization is a massive practical benefit. It eliminates the computational latency, memory overhead, and code-complexity of running backpropagation during online inference, making it highly appealing for real-world deployments.
2. **Complete Immunity to Overfitting and Volatility**: By avoiding parameter optimization, FIOSR is entirely immune to the Dynamic Routing Paradox and maintains absolute, flat-line ensembling stability across all batch sizes ($B=1$ to $512$). This is a critical property for low-latency streaming applications where batch sizes fluctuate down to $B=1$.
3. **High Computational Efficiency of dFIM**: Calculating and smoothing diagonal empirical Fisher Information coordinates has a linear computational and storage complexity ($O(d)$). In practice, this takes only **4.05 milliseconds** to compute, introducing negligible overhead during deployment.
4. **Outstanding Presentation and Rigor**: The paper is exceptionally well-written, clearly structured, and mathematically rigorous. The authors display a high level of scientific integrity by including a comprehensive and self-critical analysis of their assumptions, limitations, and potential failure modes in Appendix B.
5. **Practical Systems-Level Gating**: Proposing and evaluating Top-$M$ expert gating directly addresses a critical practical latency ceiling of batch partitioning under Micro-Batch Homogenization (MBH).

---

## Weaknesses & Critical Deployment Challenges

Despite its theoretical elegance, several significant engineering, scalability, and empirical challenges limit the immediate utility of the proposed methodology in production settings:

1. **The "Reality Gap" and Modest Real-World Performance Gains**:
   There is a substantial performance discrepancy between the synthetic sandbox and actual physical models. While FIOSR outperforms unweighted Cosine PFSR by a massive **8.56%** absolute accuracy in the synthetic homogeneous sandbox ($76.86\%$ vs $68.30\%$), this gap narrows considerably in more realistic settings:
   - In the high-fidelity simulated LoRA activation space, the joint classification accuracy improvement of FIOSR over PFSR is **6.67%** ($77.00\%$ vs $70.33\%$).
   - In the **physical end-to-end ResNet-18 deployment** (evaluated on real MNIST, FashionMNIST, and SVHN features), the performance gains are **highly modest**:
     - Routing accuracy: 56.33% (PFSR) vs. 59.00% (FIOSR) (+2.67%).
     - Joint ensembling accuracy: 50.67% (PFSR) vs. 52.00% (FIOSR) (+1.33%).
   
   This dramatic reduction in empirical gains suggests that in realistic, high-dimensional activation spaces, the assumption of coordinate-aligned noise is violated due to dense off-diagonal coordinate correlations. Under these conditions, the simple diagonal Fisher Information Matrix provides only marginal utility over standard unweighted cosine similarity. A practitioner must question whether a mere 1.33% joint accuracy improvement in an actual deployment justifies the additional complexity of dFIM estimation, pre-calibration mean-centering, and extreme-value normalization.

2. **Computational Redundancy of MBH under High Stream Heterogeneity**:
   Micro-Batch Homogenization (MBH) partitions a heterogeneous batch of size $B$ into $G \le K$ homogeneous micro-batches. If a test stream is highly heterogeneous and contains samples from all $K$ expert domains in a single batch, MBH must perform $G = K$ separate, sequential forward passes.
   From a systems and execution-latency perspective, executing $K$ sequential forward passes with $K$ distinct merged models is **computationally equivalent to (and practically slower than) simply running the original, unmerged specialized expert models** on their respective samples. Running the original experts is more efficient because it avoids the overhead of dynamic weight assembly, routing coefficient calculation, and scatter-gather operations. Thus, the core computational benefit of test-time model merging is lost in diverse streaming environments unless the active experts are strictly bounded.

3. **Memory and Storage Overhead for Modern LLM Vocabularies**:
   The class-conditional Fisher matrices require storing $K \times C \times d$ parameters. For modern Large Language Models (LLMs) with vocabulary sizes of $C \approx 32\text{K}$ to $128\text{K}$ and hidden dimensions of $d \approx 4096$, storing these Fisher coefficients introduces substantial storage and memory bandwidth overhead. Although the authors propose compression strategies (Class-Grouped Pooling, Low-Rank FIM) in the appendix, these are discussed purely conceptually and lack any empirical validation.

4. **Online Covariance Scaling Bottleneck under Non-Axis-Aligned Noise**:
   Under rotated or correlated noise, standard diagonal Fisher collapses. The proposed training-free solution (`FIOSR-Online`) relies on on-the-fly eigenvalue decomposition (EVD) of a shrinkage-regularized covariance matrix estimated from the calibration split. However, running a full online EVD during test-time is computationally expensive ($O(d^3)$ complexity) and does not scale well to modern high-dimensional representation spaces ($d \ge 1024$), creating a severe latency bottleneck.

5. **Statistical Phase Transition of Calibration Size ($N_c$)**:
   The sensitivity analysis in the appendix reveals a sharp statistical phase transition. At extremely scarce calibration regimes ($N_c \le 4$), the variance estimator is unstable and mathematically underdetermined. At $N_c = 2$, FIOSR achieves **55.97%** accuracy, which represents a massive **-9.48%** absolute loss compared to the flat baseline. This means that if calibration data is extremely limited, the proposed method actually hurts performance significantly compared to simple, unweighted cosine similarity, presenting a serious operational risk.

6. **Asymmetrical Alternative-Hypothesis Penalty in CSC**:
   The Class-Size Scaling Calibration (CSC) normalizer $\sqrt{2\log C_k / d}$ is derived under null-hypothesis conditions. Under alternative-hypothesis conditions (a genuine positive match), it introduces an asymmetrical penalty that unfairly penalizes experts with larger vocabularies, even if they achieve an identical genuine class prototype match as a smaller vocabulary expert.

---

## Detailed Ratings

### 1. Soundness: Good
The paper is mathematically rigorous and carefully formulated. The dual-space alignment proof and the asymptotic behavior of the smoothed Fisher regularizer are solid. However, the diagonal assumption collapses under rotated noise, and the proposed training-free EVD alignment (`FIOSR-Online`) scales poorly ($O(d^3)$) and provides only marginal gains over the flat baseline ($67.68\%$ vs $67.50\%$). Furthermore, the method is highly sensitive to microscopic calibration sizes ($N_c \le 4$), where it catastrophically underperforms.

### 2. Presentation: Excellent
The paper is beautifully written, clear, and extremely well-structured. The narrative flow is highly logical, and the notations are consistent. The figures are clean, and the tables are informative. The authors are highly commended for their exceptional intellectual honesty in Appendix B, where they exhaustively detail and critique seven of their own key assumptions and limitations.

### 3. Significance: Good
The paper introduces a highly elegant information-geometric framework that connects Riemannian manifold curvature with test-time model merging. It addresses an important problem in dynamic model merging and successfully bypasses test-time parameter optimization. However, its practical significance is constrained by the modest gains on actual physical models (+1.33% joint accuracy improvement) and the sequential forward pass latency of MBH under heterogeneous streams, which represent significant barriers to real-world deployment.

### 4. Originality: Excellent
The dynamic, online use of the diagonal empirical Fisher Information Matrix as a coordinate-warping metric tensor to replace flat Euclidean/cosine similarity in test-time projection is highly original and represents a creative synthesis of information geometry and modular deep learning.

---

## Overall Recommendation

**Rating: 4 (Weak Accept)**

**Justification:** 
This is a technically solid, mathematically rigorous, and exceptionally well-written paper that advances the sub-area of test-time model merging by introducing a novel, training-free information-geometric framework. By bypassing test-time parameter optimization, the proposed FIOSR framework successfully and elegantly resolves critical, well-documented failure modes (few-shot overfitting and sequential stream instability) of parametric routers. 

However, several critical practical weaknesses limit its real-world impact and deployment viability: the "reality gap" where joint accuracy gains drop from +8.56% (synthetic sandbox) to a highly modest +1.33% (physical ResNet-18); the worst-case computational redundancy of Micro-Batch Homogenization (MBH) where executing $K$ sequential passes is computationally equivalent to/slower than running the original specialized models; the storage footprint of class-conditional Fisher coordinates for large vocabularies; and the expensive $O(d^3)$ scaling of online covariance alignment under rotated noise. These practical bottlenecks limit its immediate applicability in high-throughput production pipelines, making Weak Accept the most appropriate recommendation.

---

## Questions and Suggestions for the Authors

1. **Addressing the "Reality Gap"**: In Section 4.8, the joint accuracy improvement of FIOSR over unweighted Cosine PFSR on the physical ResNet-18 backbone is only 1.33%. Real-world activation representations have dense, non-axis-aligned coordinate correlations. How do you plan to scale the information-geometric warping to capture off-diagonal correlations (e.g., K-FAC or low-rank covariance) in high-dimensional spaces ($d \ge 1024$) without incurring the prohibitive $O(d^3)$ cost of eigenvalue decomposition during inference?
2. **Computational Redundancy of MBH**: In a highly heterogeneous stream containing samples from all $K$ active expert domains, MBH partitions the batch into $K$ micro-batches and runs $K$ sequential forward passes. Since running $K$ sequential forward passes with $K$ differently merged weights is computationally equivalent to simply executing the original $K$ unmerged expert models on their respective samples (and avoids weight assembly and scatter-gather overhead), why would a practitioner choose test-time model merging over simple hard task routing? Please discuss this practical system-level trade-off more prominently in the main text.
3. **Empirical Validation of LLM Compression**: For a vocabulary size of $C \approx 32\text{K}$ to $128\text{K}$ in LLMs, storing class-conditional Fisher matrices is a major memory bottleneck. Could you provide a small empirical validation of your proposed "Class-Grouped Pooling" or "Low-Rank FIM Factorization" compression strategies (even on a small model) to prove they can preserve routing accuracy while reducing the storage footprint?
4. **Dynamic CSC Calibration**: To address the asymmetrical penalty of the CSC divisor $\sqrt{2\log C_k / d}$ on large-vocabulary experts under true positive matches, have you considered a dynamic calibration divisor that adjusts its scaling based on whether the similarity represents a null-hypothesis noise score or an alternative-hypothesis genuine match?
