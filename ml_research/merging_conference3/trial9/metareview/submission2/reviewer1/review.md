# Peer Review of Conference Submission

**Paper Title:** Resource-Budgeted Top-$M$ Expert Serving: Dynamic Activation-Space Gating for Low-Power Edge Model Merging

---

## Review Summary
This paper presents **Resource-Budgeted Top-$M$ Expert Serving (RB-TopM)**, a training-free, hardware-aware dynamic ensembling framework designed to optimize the serving path of task-specific Low-Rank Adaptation (LoRA) experts under volatile edge compute budgets. Governing the serving path is a real-time hardware resource budget coefficient $C_{\text{budget}} \in [0, 1]$, which dynamically scales the ensembling capacity limit ($M(C_{\text{budget}})$) and the active expert pruning threshold ($\theta(C_{\text{budget}})$). In addition, the framework integrates an early Coordinate diagonal Gaussian Mixture Model (GMM) safety shield to reject out-of-distribution (OOD) inputs, saving downstream computation. To scale to large registries, a Hierarchical Macro-Domain GMM Routing (HMD-GMM) architecture is proposed.

The paper is exceptionally well-written, clearly positioned, and demonstrates a rare level of mathematical and systems-level rigor. It provides a formal theoretical derivation of the "activation dilution" phenomenon, proving why dynamic pruning acts as a hard-thresholding covariance regularizer—which beautifully explains the empirical observation of non-monotonic serving accuracy.

---

## Strengths

1. **Rigorous Theoretical Grounding:** Unlike many purely empirical papers in the model ensembling and merging literature, this work provides a solid mathematical foundation. It formalizes representation-space environmental noise and models "activation dilution" as an inflation of the ensembled representation's covariance. Proving that dynamic thresholding acts as a hard-thresholding operator that eliminates this covariance penalty is an elegant and highly satisfying contribution.
2. **Exhaustive Systems-ML Co-Design:** The paper exhibits deep systems-level awareness. It features a detailed Roofline Model analysis to prove that expert serving is strictly memory-bandwidth-bound ($\text{OI}_{\text{expert}} = 0.5$ FLOPs/byte) due to DRAM-to-SRAM weight transfer limitations, and analyzes LPDDR5 shared-memory bus queue occupancy. This co-design justifies why a 78.5% reduction in expert DRAM fetches translates directly to a simulated 17.5% overall system serving latency reduction.
3. **Training-Free, Microsecond-Scale Adaptability:** Since the closed-loop control equations are closed-form and arithmetic-only, the system-level controller can adjust $C_{\text{budget}}$ in microsecond scales. This allows the model serving pipeline to respond instantly to physical OS-level interrupts (e.g., thermal throttle warnings or low-battery broadcasts) without graph re-compilation or virtual memory paging.
4. **Comprehensive Evaluation and Strong Baselines:** The framework is validated across 10 random seeds on a 14-layer Analytical Coordinate Sandbox (ICS) and on a TVM-compiled compiler-simulation pilot using MobileNetV3-Large on DomainNet. It is compared against a comprehensive set of baselines, including Expert Oracle, Uniform Merging, SABLE SOTA, SPS-ZCA, Q-SPS, and state-of-the-art static parameter-space merging techniques (TIES-Merging and DARE), outperforming the static alternatives by up to 8% in accuracy.

---

## Weaknesses and Detailed Technical Critiques

Despite its exceptional quality, we identify several subtle mathematical simplifications and structural assumptions that should be addressed to maximize the submission's mathematical soundness:

1. **Shared Noise Coupling in the Activation Dilution Proof:**
   In Appendix A.1, Equation (18) expresses the covariance of the ensembled representation $Y^{(l)}$ as:
   $$\text{Cov}\left( Y^{(l)} \right) \approx \alpha_{k^*}^2 \text{Cov}(y_{k^*}^{(l)}) + \sum_{k \neq k^*} \left( \bar{\alpha}_k^2 + \text{Var}(\delta_k) \right) \left[ \sigma_{\text{inter}}^{(l)2} I_D + \sigma_{\text{env}}^2 (A_k^{(l)} B_k^{(l)}) (A_k^{(l)} B_k^{(l)})^T \right]$$
   This formulation assumes that the secondary expert terms under the summation are mutually independent. However, because all secondary expert pathways process the *same* perturbed input representation $h^{(l-1)} = h_0^{(l-1)} + \epsilon_l$, they are coupled through the shared environmental noise vector $\epsilon_l \sim \mathcal{N}(0, \sigma_{\text{env}}^2 I_D)$. 
   Thus, there are non-zero cross-covariance terms for any $j \neq k$ ($j, k \neq k^*$):
   $$\text{Cov}\left( \alpha_j y_j^{(l)}, \alpha_k y_k^{(l)} \right) \approx \bar{\alpha}_j \bar{\alpha}_k \sigma_{\text{env}}^2 \left( A_j^{(l)} B_j^{(l)} \right) \left( A_k^{(l)} B_k^{(l)} \right)^T$$
   Neglecting these cross-covariance terms mathematically understates the true activation dilution penalty when multiple secondary experts are active simultaneously. Under high environmental noise $\sigma_{\text{env}}$, this coupled noise term scales quadratically with the number of active un-gated experts. Addressing this noise coupling in the appendix would significantly strengthen the authors' theoretical case for dynamic pruning.

2. **Discrete Latency and Capacity Jitter in the Control Loop:**
   The dynamic top-$M$ capacity limit $M(C_{\text{budget}}) = \max(1, \lfloor M_{\max} \cdot C_{\text{budget}} \rfloor)$ is modeled as a step function. Because the system budget $C_{\text{budget}}$ can fluctuate continuously due to transient active OS scheduling and physical temperature changes, a step function introduces discrete jump-discontinuities. In real-time closed-loop control systems, such step-function transitions can induce sudden "latency jitter" or computational spikes on boundary values (e.g., when $C_{\text{budget}}$ oscillates around $0.25$ or $0.50$). Discussing or analyzing a smoother, continuous soft-thresholding function (e.g., sigmoid-gated top-$M$ selection or temperature-controlled smooth gating) would be theoretically superior to prevent control-loop destabilization.

3. **Bounded Support Approximation in GMM OOD Detection:**
   The Coordinate GMM safety shield is fitted over cosine similarity coordinates $u'_b \in [-1, 1]^K$. Fitting a standard multivariate Gaussian Mixture Model (which assumes infinite support on $\mathbb{R}^K$) over a bounded hypercube $[-1, 1]^K$ is a mathematical approximation. While the mass outside the boundary is small, the density estimation near the hypercube boundaries remains theoretically uncalibrated. Although the authors provide a convincing systems-level defense of the diagonal GMM's microsecond-scale integer compatibility over the transcendental complexity of von Mises-Fisher (vMF) Bessel functions, this mathematical bounded support approximation should be explicitly stated as a limitation.

4. **Heuristic Constraints in Hierarchical GMM Routing (HMD-GMM):**
   The partition of $K$ tasks into $G = \lceil K / 4 \rceil$ macro-domains via Agglomerative Clustering is a heuristic. The choice of grouping up to 4 tasks per domain is arbitrary. The paper lacks a theoretical proof or bound showing that Level-1 OOD misclassification is strictly bounded under arbitrary task overlaps and scaling dimensions.

---

## Ratings

- **Soundness:** **Good**
  The paper is theoretically and empirically highly solid. However, minor mathematical simplifications in the covariance derivation and the step-function control loop prevent a rating of "excellent."
- **Presentation:** **Excellent**
  The paper is beautiful, highly polished, and structured. Figures, tables, pseudo-code, and glossaries are of outstanding professional quality.
- **Significance:** **Excellent**
  The serving bottleneck of PEFT models on low-power devices is a critical and immediate problem. The proposed training-free solution holds substantial practical and research significance.
- **Originality:** **Excellent**
  The combination of hardware budget monitors with dynamic ensembling, the formalization of activation dilution, and the HMD-GMM hierarchical scaling are highly creative and novel.

---

## Overall Recommendation

**Rating:** **5: Accept**

This is an exceptionally strong, publication-ready paper that successfully bridges systems-ML engineering and deep learning theory. The mathematical derivation of activation dilution as a representation covariance penalty is highly elegant. The proposed closed-loop control and OOD safety shield are practical, training-free, and highly optimized for edge deployment. Addressing the noise-coupling terms in the covariance proof, analyzing smooth alternatives to the step-function capacity limit, and explicitly noting the bounded support approximation of the GMM would elevate this to a top-tier submission.

---

## Questions and Clarifications for the Authors

1. **Noise Coupling in Covariance Derivation:** In Appendix A.1, Equation (18), the covariance of the secondary experts is summed independently. Since they all process the same perturbed activation vector $h^{(l-1)} = h_0^{(l-1)} + \epsilon_l$, the terms are coupled through the shared environmental noise $\epsilon_l$. Have the authors analyzed the magnitude of the cross-covariance terms $\text{Cov}(\alpha_j y_j^{(l)}, \alpha_k y_k^{(l)})$ for $j \neq k$? Correcting this coupling would strengthen the activation dilution argument.
2. **Continuous Gating Alternatives:** Did the authors evaluate any continuous alternatives to the step-function capacity limit $M(C_{\text{budget}})$ (e.g., a sigmoid-scaled soft top-$M$ gate)? How does the physical control loop manage latency jitter when $C_{\text{budget}}$ oscillates rapidly around a boundary threshold?
3. **Quantization Noise Impact:** How does low-precision quantization (e.g., INT8/INT4 in Q-SPS) affect the coordinate-space geometry of the ZCA projection? Does the coordinate-shift induced by quantization noise trigger increased false positives in the GMM safety shield, and can Intra-Task Dispersion Calibration (IDC) mathematically correct this shift?
4. **HMD-GMM Theoretical Bounds:** Is it possible to construct a theoretical bound or proof on the Level-1 macro-domain OOD rejection rate under HMD-GMM as the task overlap or number of macro-domains $G$ scales?
