# Peer Review

**Paper Title:** Information-Geometric Subspace Routing: A Provably Stable Parameter-Free Framework for Test-Time Model Merging

---

## 1. Summary of the Paper
This paper addresses the challenge of test-time model merging (parameter ensembling) of specialized modular domain-expert models (such as LoRAs) sharing a pre-trained backbone. It proposes **Fisher-Information Optimal Subspace Routing (FIOSR)**, a training-free and parameter-free dynamic ensembling framework that treats the parameter representation space as a Riemannian manifold. 

Rather than utilizing flat Euclidean geometric assumptions (such as unweighted cosine similarity, which assumes $\mathbf{g} = \mathbf{I}$), FIOSR warps the coordinate space using a local Riemannian metric tensor constructed from a smoothed and power-scaled diagonal empirical Fisher Information Matrix (dFIM) of the expert heads. By projecting test representations onto expert class prototypes using a **Fisher-Weighted Cosine Similarity**, FIOSR naturally suppresses noisy or task-irrelevant coordinates (high variance, low Fisher Information) and amplifies highly informative task features (low variance, high Fisher Information). 

To ensure stability across sequential streams and asymmetric vocabularies, the framework incorporates **Class-Size Scaling Calibration (CSC)**—correcting for extreme value maximum bias—and **Micro-Batch Homogenization (MBH)**—preventing heterogeneity collapse through batch partitioning.

---

## 2. Strengths (Soundness, Significance, Originality, Presentation)

### A. Soundness & Mathematical Rigor
* **Information-Geometric Framing:** The paper is exceptionally theoretically grounded. Rather than employing empirical coordinate-weighting heuristics, it formally derives that the diagonal Fisher Information coordinate represents the inverse coordinate noise variance ($F_j = 1/\sigma_j^2$) under a conditional Gaussian assumption, providing a principled information-geometric justification.
* **Dual-Space Relationship Bounding:** The authors proactively bridge a potential conceptual gap—applying representation-derived metrics to warp classifier weights—by proving a formal dual-space relationship under regularized softmax cross-entropy training. They formally bound the finite-sample directional misalignment on the unit sphere as $\le C_0/\sqrt{N_c} = \epsilon$, which mathematically justifies using classification weights as proxies for class centroids.
* **Non-Gaussian Robustness Proof:** The paper includes a complete derivation of dFIM under non-Gaussian rectified (ReLU) activations (Appendix 1.2), proving that the inverse-variance relationship $F_j \propto 1/\sigma_j^2$ dominates coordinate sensitivity even under non-negative sparsity and severe noise.
* **Empirical Integrity:** The authors are highly transparent, explicitly detailing their synthetic coordinate sandbox, its noise characteristics, and evaluating all results across 10 independent random seeds. They successfully bridge the external validity gap by validating the framework end-to-end on a physical pre-trained ResNet-18 backbone with real image datasets (MNIST, FashionMNIST, SVHN).

### B. Originality & Significance
* **Novel Test-Time Application:** This is the first work to utilize Fisher Information dynamically at test-time as a coordinate-warping metric tensor for parameter-free subspace routing, reframing modular ensembling on Riemannian representation manifolds.
* **Optimization Bypass:** By completely bypassing test-time parameter optimization, FIOSR is immune to the *Dynamic Routing Paradox* (few-shot overfitting) and *Vectorization Collapse* (sequential stream instability), establishing state-of-the-art results on parameter-free ensembling and dynamic merging.

### C. Presentation Quality
* **Clarity & Structure:** The writing is exceptionally clear, precise, and dense with technical rationale. The figures and tables are professional and self-contained. The authors provide a structured mathematical notation reference table in Appendix Table 1, which greatly aids readability.

---

## 3. Weaknesses (Areas for Improvement)

### A. Definite Mathematical Sign Error in Appendix 1.2
While verifying the derivations in the appendix, we identified a standard sign error in integration by parts. 
In Equation (36), the continuous component of the rectified Gaussian coordinate is integrated:
$$\int_{-\mu_j/\sigma_j}^\infty t^2 \phi(t) dt = \left[ -t\phi(t) \right]_{-\mu_j/\sigma_j}^\infty + \int_{-\mu_j/\sigma_j}^\infty \phi(t) dt$$
Let us evaluate the boundary term $\left[ -t\phi(t) \right]_{-\mu_j/\sigma_j}^\infty$:
* The upper limit is $\lim_{t \to \infty} -t \phi(t) = 0$.
* The lower limit is evaluated at $t = -\mu_j/\sigma_j$.
* Therefore:
$$\left[ -t\phi(t) \right]_{-\mu_j/\sigma_j}^\infty = (0) - \left( - \left(-\frac{\mu_j}{\sigma_j}\right) \phi\left(-\frac{\mu_j}{\sigma_j}\right) \right) = -\frac{\mu_j}{\sigma_j} \phi\left(-\frac{\mu_j}{\sigma_j}\right)$$

However, in Equation (37), the authors write:
$$I_{\text{continuous}} = \frac{1}{\sigma_j^2} \left[ 1 - \Phi\left(-\frac{\mu_j}{\sigma_j}\right) + \frac{\mu_j}{\sigma_j} \phi\left(-\frac{\mu_j}{\sigma_j}\right) \right]$$
which incorrect propagates a positive sign on the boundary term. This error is carried over to the total Fisher Information expression in Equation (39):
$$F_{j} = \frac{1}{\sigma_j^2} \left[ 1 - \Phi\left(-\frac{\mu_j}{\sigma_j}\right) + \frac{\mu_j}{\sigma_j} \phi\left(-\frac{\mu_j}{\sigma_j}\right) + \frac{\phi(-\mu_j/\sigma_j)^2}{\Phi(-\mu_j/\sigma_j)} \right]$$

**Correction Required:**
The boundary term should have a negative sign. The mathematically correct total Fisher Information is:
$$F_{j} = \frac{1}{\sigma_j^2} \left[ 1 - \Phi\left(-\frac{\mu_j}{\sigma_j}\right) - \frac{\mu_j}{\sigma_j} \phi\left(-\frac{\mu_j}{\sigma_j}\right) + \frac{\phi(-\mu_j/\sigma_j)^2}{\Phi(-\mu_j/\sigma_j)} \right]$$
*Impact:* Crucially, since the term $\frac{\mu_j}{\sigma_j}\phi(-\mu_j/\sigma_j)$ still vanishes as $\sigma_j \to \infty$, the asymptotic result $F_j \propto 1/\sigma_j^2$ is unaffected. However, this sign error must be corrected to maintain mathematical soundness.

### B. External Validity Gap in Physical Deployments
In the end-to-end physical ResNet-18 deployment (Section 4.8), FIOSR's joint ensembling accuracy improvement over the flat Cosine baseline is $+1.33\%$ (52.00% vs 50.67%), which is noticeably modest compared to the $+8.56\%$ gap in the simulated coordinate-aligned sandbox (76.86% vs 68.30%). 
* **Critique:** This narrower gap suggests that while diagonal Fisher is highly powerful in simulated, coordinate-aligned settings, actual physical activation spaces are highly complex with dense, non-axis-aligned covariance structures, where diagonal Fisher's advantage is dampened unless full block-diagonal (K-FAC) or shrinkage EVD alignment is employed. The authors should explicitly highlight and discuss this external validity gap in the main text.

### C. Asymmetrical Alternative-Hypothesis Penalty of CSC
The CSC normalization divisor $\sqrt{2\log C_k / d}$ is derived under the null hypothesis (expected maximum of independent random variables).
* **Critique:** Under the alternative hypothesis (a true positive match), this introduces an asymmetrical penalty on larger class vocabularies. If a 10-class task and a 4-class task both achieve an identical, genuine prototype match of similarity $0.8$, the 10-class task's score will be divided by a larger divisor, creating an artificial bias favoring smaller-vocabulary tasks when true matches occur. The authors should acknowledge this limitation.

---

## 4. Specific Section Evaluation

### Soundness: Excellent
The paper is exceptionally sound, supported by both rigorous mathematical derivations (modulo the intermediate sign error) and comprehensive empirical validations. The authors' pooled within-class variance estimator successfully isolates pure coordinate noise from class centroid spread, demonstrating exceptional soundness.
* **Rating:** Excellent

### Presentation: Excellent
The paper is beautifully structured, highly readable, and exceptionally detailed. The notation is consistent, and the inclusion of Appendix Table 1 is extremely helpful.
* **Rating:** Excellent

### Significance: Good
Test-time model merging is an active, hot topic in deep learning. Reframing ensembling through information geometry represents a highly valuable, principled advance, although the modest gain in the physical ResNet-18 evaluation indicates some real-world latency-accuracy trade-offs.
* **Rating:** Good

### Originality: Excellent
By treating the parameter space as a Riemannian manifold and using dFIM dynamically at test-time, the paper distinguishes itself sharply from previous static merging and unweighted dynamic ensembling heuristics.
* **Rating:** Excellent

---

## 5. Overall Recommendation
**Score: 5: Accept**

**Justification:**
This is an outstanding, mathematically rigorous, and highly complete paper. By reframing parameter-free subspace routing as coordinate-warping on Riemannian representation manifolds, it provides a beautiful information-geometric foundation for test-time model ensembling. It is completely immune to the Dynamic Routing Paradox and Vectorization Collapse, outperforming state-of-the-art baselines across all streaming regimes. While we have identified a minor mathematical sign error in Appendix 1.2, a translation bias limitation, and a narrower accuracy gain in the physical deployment, these do not detract from the paper's overall exceptional quality. This paper is highly suitable for acceptance, provided the authors correct the sign error in Appendix 1.2 and incorporate a discussion of the identified limitations.
