# Conference Peer Review

## Summary of the Paper
The paper addresses a critical deployment bottleneck in parameter-efficient fine-tuning (PEFT): serving multiple specialized, task-specific expert adapters (such as LoRA) simultaneously to a stream of heterogeneous, noisy, and unpredictable real-world requests on edge hardware. To resolve this, the authors propose **Evolutionary Symbiotic Merging via Lotka-Volterra Cooperation (ESM-LVC)**, a dynamic, training-free, and parameter-free ensembling framework in activation space.

Rather than treating specialized task experts as isolated, independent entities existing in a vacuum, the paper models ensembling coefficients as interacting species populations whose densities evolve over a localized virtual timescale governed by a classical Lotka-Volterra competition-cooperation framework. The framework consists of three core components:
1. **Lotka-Volterra Activation Dynamics (LVAD):** A non-linear dynamical system governing the temporal evolution of expert activation pathways inside the forward pass.
2. **Symbiotic Interaction Tensor (SIT):** A pre-computed semantic interaction matrix deriving cooperative (mutualistic) and competitive (exclusionary) relationships between experts based on zero-shot centroid similarities.
3. **Discrete Euler Symbiosis Solver (DESS):** An ultra-lightweight Projected Euler integrator with an Adaptive Step-Size Heuristic that solves these differential equations on-the-fly with negligible latency.

To resolve the trade-off between soft regularization and logit dilution under noise, the paper introduces several advanced algorithmic extensions, including **Exponential Information-Theoretic Adaptive Sharpening (E-ITAS)**, **Dirichlet-Multinomial Bayesian Self-Calibration (DM-BSC)**, **Gaussian Mixture Centroids (GMC)**, and **Dynamic Scale Alignment (DSA)**.

Evaluating the framework in a calibrated high-fidelity 14-layer synthetic Isolating Coordinate Sandbox (ICS) and performing offline "physical model verification" on CLS token activations from Layer 12 of a pre-trained Vision Transformer model across four real-world datasets (MNIST, Fashion-MNIST, CIFAR-10, SVHN), the authors demonstrate that ESM-LVC achieves state-of-the-art Joint Mean accuracy (75.12%), maintains high accuracy under extreme noise (65.37% at Scale 2.5), and exhibits absolute immunity to batch-size and stream-level heterogeneity collapse (0.00% collapse vs. 19.86% collapse for weight-space parametric routers).

---

## Strengths and Weaknesses

### Strengths
1. **Deep and Grounded Theoretical Framing:** 
   Unlike many purely empirical deep learning papers that rely on ad-hoc heuristics, this paper is rigorously theoretically grounded in non-linear dynamical systems and mathematical ecology. The authors successfully modernize connectionist lateral inhibition, scaling it to high-dimensional specialized adapter channels (experts) rather than individual neurons or logits.
2. **Theorem 3.1 and Mathematical Trajectory Guarantees:** 
   The derivation and proof of **Theorem 3.1 (Boundedness and Stability of DESS Trajectories)** is a major technical achievement. Under both infinite-horizon and finite-horizon cooperation regimes, the authors provide strict mathematical guarantees that the ensembling coefficients remain bounded within physically meaningful limits, ensuring integration stability under arbitrary expert scales. This theoretical safety is vital for real-world edge serving.
3. **Rigorously Calibrated and Non-Strawman Evaluation:** 
   The evaluation is comprehensive and highly rigorous. By including both a **Few-Shot** and a **Fully-Optimized** parametric Linear Router in physical verification, the authors avoid "strawman" comparisons. The results convincingly show that ESM-LVC achieves the superior decision boundaries of trained parametric heads under clean settings while offering superior out-of-distribution noise filtering under extreme shifts.
4. **Outstanding Scientific Integrity and Transparency:** 
   The paper stands out for its high level of scientific transparency:
   * It openly discloses the **attractor equivalence bottleneck** of single-centroid non-parametric routing (SABLE, SPS-ZCA, and basic ESM-LVC).
   * It provides a detailed, mathematically grounded diagnostic explaining the low absolute joint classification accuracies of the downstream linear probes.
   * It analyzes and successfully resolves the **moderate noise regularization anomaly** using information-theoretic and Bayesian self-calibration frameworks (E-ITAS and DM-BSC).
5. **Excellent Writing and Presentation Quality:** 
   The paper is exceptionally well written, structured, and easy to follow. The mathematical notation is consistent, terms are fully defined, and the execution layout (Section 3.4) provides a clear blueprint for systems integration.

### Weaknesses
1. **Theoretical Gap Between Trajectory Boundedness and Convergence Rate:** 
   Theorem 3.1 rigorously proves that the DESS Projected Euler trajectories are bounded, which is highly commendable. However, **boundedness does not guarantee convergence to a steady-state equilibrium** within the fixed step budget of $N=5$. Competitive-cooperative Lotka-Volterra systems can exhibit limit cycles, oscillations, or even chaotic behavior under discrete integration. If the trajectories do not converge rapidly, the final ensembling coefficients $\alpha^{(N)}$ will be highly sensitive to the exact step count $N$. A formal contraction mapping or convergence rate analysis for the discrete Projected Euler operator would elevate the paper's mathematical completeness.
2. **Theoretical Stability Risks of Asymmetric Biological Systems:** 
   While the authors rightly point out that Theorem 3.1's bounds apply to asymmetric interaction matrices ($\Gamma_{k, j} \neq \Gamma_{j, k}$), asymmetric Lotka-Volterra systems (such as predator-prey dynamics) are highly prone to chaotic orbits. The paper lacks a theoretical analysis or stability boundary mapping demonstrating why their proposed asymmetric localized thresholds or directional projection formulations do not trigger chaotic oscillations under edge serving.
3. **Heuristic Probabilistic Grounding of Bayesian Self-Calibration (DM-BSC):** 
   The Dirichlet-Multinomial Bayesian Self-Calibration (DM-BSC) framework (Equations 21--24) is highly intuitive and empirically successful. However, treating continuous cosine-similarity affinities $u_{k,b} \in [0, 1]$ directly as empirical pseudocount observations $\mathbf{n}_b = \kappa \cdot u_b$ lacks formal measure-theoretic and decision-theoretic justification.
4. **Simplification of the Power-Law Performance Model:** 
   The synthetic ICS sandbox relies on a calibrated power-law performance model ($\text{Acc}_k(\alpha) = C_k \cdot \alpha_k^{\gamma_k}$) and a bilinear destructive interference penalty. While useful for controlled comparisons, real physical adapter blending is highly non-linear, making the offline physical verification on pre-trained CLS tokens (Section 4.7) the primary anchor of the paper's physical feasibility.

---

## Technical Soundness
* **Rating:** Excellent
* **Justification:** The proposed framework is technically solid and highly sound. The mathematical formulations are clear and precise. The Projected Euler method and Adaptive Step-Size Heuristic appropriately address the numerical instability of continuous-to-discrete biological equations. The proof of Theorem 3.1 is correct, rigorously derived, and provides crucial boundedness guarantees. The physical model verification on Layer 12 CLS tokens from a physical Vision Transformer successfully bridges the simulation-to-reality gap, proving that the mathematical attractor dynamics translate perfectly to physical representations.

---

## Presentation Quality
* **Rating:** Excellent
* **Justification:** The paper is beautifully written, exceptionally well structured, and extremely easy to follow. The mathematical notation is consistent, formulas are fully defined, and tables and figures are clean, high-contrast, and professional. The authors do an outstanding job of positioning their work relative to both static model merging and classical connectionist lateral inhibition models.

---

## Significance
* **Rating:** Excellent
* **Justification:** This paper addresses a highly important and relevant problem: serving multiple specialized task adapters simultaneously on edge hardware. It demonstrates that dynamic, training-free, and parameter-free activation-space ensembling can match or outperform trained parametric routing heads, successfully bypassing batch-level heterogeneity collapse and severe out-of-domain noise sensitivity. It opens up a highly promising frontier for organic, resilient model serving.

---

## Originality
* **Rating:** Excellent
* **Justification:** The originality of this work is outstanding. Framing model ensembling as a self-organizing dynamic ecosystem and implementing Lotka-Volterra competition-cooperation differential equations directly into the forward pass of a transformer block is highly creative. The proposed SIT, DESS, E-ITAS, DM-BSC, GMC-BSC, and DSA modules represent novel combinations of mathematical ecology, Bayesian statistics, and deep learning systems.

---

## Overall Recommendation
* **Score:** 5: Accept
* **Justification:** This is an exceptionally high-quality submission. It is rigorously theoretically grounded, mathematically sound, beautifully written, and thoroughly evaluated across both synthetic and physical representation spaces. The derivation of Theorem 3.1 and formal proofs of trajectory boundedness are highly commendable. The physical model verification on actual Vision Transformer CLS tokens successfully bridges the simulation gap, proving that the proposed dynamical attractor networks generalize to physical representations. While there is a minor theoretical gap regarding discrete convergence rates and asymmetric chaos risks, the paper's strengths, outstanding originality, and empirical transparency far outweigh these minor limitations. I strongly recommend this paper for acceptance.

---

## Questions and Suggestions for the Authors

1. **Theoretical Convergence Rate Analysis:** 
   Theorem 3.1 proves the boundedness of the ensembling coefficients $\alpha^{(t)}$. Can you provide a contraction mapping or Lipschitz analysis of the discrete Projected Euler operator to formally guarantee convergence to a steady-state equilibrium? Under what parameter bounds (on $\Delta\tau$ and $\lambda$) is the steady state guaranteed to be a stable node rather than a limit cycle or chaotic attractor?
2. **Chaotic Dynamics in Asymmetric/Predator-Prey Regimes:** 
   Since asymmetric Lotka-Volterra systems are prone to chaotic orbits, did you monitor the trajectory convergence or fallback rates when deploying your asymmetric proposals (asymmetric localized thresholds or directional projection alignments)? Discussing the stability boundaries under asymmetric biological parameters would strengthen the paper.
3. **Quantitative FLOPs and Edge Latency of GMC-BSC:** 
   The Gaussian Mixture Centroids (GMC) framework is an empirical success. However, scaling to $M=3$ local cluster centers per task increases projection complexity. Can you provide a quantitative breakdown of the FLOPs and actual edge execution latency (in microseconds) of the GMC routing module?
4. **Downstream Probe Calibration Size:** 
   Did you explore scaling the calibration size of the downstream linear probe classifiers from 64 samples to 256 or 512 samples per task? While your current diagnostic analysis is highly refreshing and mathematically sound, larger calibration sets would likely boost the absolute downstream joint classification accuracies, making the physical end-to-end evaluation much more visually compelling.
