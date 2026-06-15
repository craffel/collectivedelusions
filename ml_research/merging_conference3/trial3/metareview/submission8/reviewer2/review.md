# Peer Review

## Summary of the Paper
This paper addresses the task of **test-time parameter-space model merging** (TTA model merging), which combines multiple task-specific expert neural networks (fine-tuned from the same base model) into a single multi-task network at test time using small, unlabeled calibration streams. 

The authors first identify the **Overfitting-Optimizer Paradox**: unconstrained first-order optimization of layer-wise merging coefficients (e.g., standard AdaMerging) on small test-time calibration streams (e.g., $N \le 64$) aggressively fits transductive noise, causing highly volatile, high-frequency spatial oscillations in learned layer coefficients and catastrophic generalization collapse on unseen target test data (specifically on SVHN).

To resolve this, the paper introduces **GP-BayesMerge**, a mathematically rigorous Gaussian Process PAC-Bayes framework that reframes test-time model merging as a Bayesian inference problem. Placing a continuous Gaussian Process (GP) prior over layer-wise merging coefficients as a function of normalized network depth, and minimizing Alquier's linear PAC-Bayes generalization bound, mathematically derives a unified **quadratic precision-matrix regularization** ($\Sigma_{\ell}^{-1}$). This operator simultaneously enforces distance-from-initialization (diagonal proximity decay) and continuous spatial smoothness (off-diagonal finite-difference Laplacian).

The paper introduces several elegant extensions and theoretical results:
1. **Ornstein-Uhlenbeck (OU) Kernel:** Under an OU prior, the precision matrix is strictly tridiagonal, allowing exact analytical inversion in linear time $O(L)$ rather than cubic time $O(L^3)$, making the method highly scalable to ultra-deep architectures.
2. **Non-Stationary Block-Wise GP Prior:** Dynamically scales down covariance across distinct functional stages (e.g. attention vs. MLP blocks) via a decoupling factor $\rho$ to prevent over-smoothing.
3. **Kronecker Multi-Task GP Prior:** Models cross-task conflicts using a joint prior $B^{-1} \otimes \Sigma_{\ell}^{-1}$, where the task-correlation matrix $B$ is estimated data-free and online using activation Centered Kernel Alignment (CKA) with shrinkage on calibration samples.
4. **Theorem 1 (Surrogate-to-Target Risk Bound):** Mathematically bridges the unsupervised surrogate-to-target risk gap, bounding true classification error by normalized prediction entropy under semi-supervised assumptions (Margin-Preserving Support and Classifier Calibration).

The authors evaluate GP-BayesMerge using a high-fidelity non-convex simulation calibrated to a 12-layer Vision Transformer (ViT-B/16), and validate it on actual physical weight merging of pre-trained CLIP ViT-B/32 and CLIP ViT-L/14 models across 8 diverse datasets. Under both settings, GP-BayesMerge completely resolves the Overfitting-Optimizer Paradox, achieving state-of-the-art accuracy, exceptional stability across seeds, rapid optimization convergence (fewer than 50 steps), and minimal computational overhead.

---

## Strengths and Weaknesses

### Major Strengths (Originality and Significance)
- **Elegant Conceptual Breakthrough:** This is a highly original, paradigm-shifting work. Instead of proposing another empirical heuristic for model merging or test-time adaptation, the paper builds a complete, first-principles probabilistic science. Reframing test-time adaptation as a Bayesian inference problem and deriving a unified spatial regularizer directly from Alquier's linear PAC-Bayes bound is a brilliant conceptual leap.
- **Unification of Disjoint Constraints:** The derivation showing that the GP precision matrix $\Sigma_{\ell}^{-1}$ naturally acts as both a proximity weight-decay penalty (on the diagonal) and an adjacent-layer Laplacian smoother (on the off-diagonals) is mathematically beautiful and highly intuitive. This provides a rigorous first-principles justification for constraints that prior works treated as disconnected, heuristic penalties.
- **Computational and Scalability Brilliance (OU Prior):** By adapting the spatial prior to an OU process, the authors achieve a strictly tridiagonal precision matrix with an exact analytical inverse. This scales linearly in $O(L)$ time and adds zero online adaptation latency. This is of outstanding practical importance for ultra-deep foundation models.
- **Addressing Structural and Semantic Complexity:** The non-stationary block-wise prior and the multi-task Kronecker joint prior (using online, data-free activation CKA with shrinkage) demonstrate that the authors have deeply and successfully modeled the structural and cross-task realities of deep neural weight interpolation.
- **Exemplary Scientific Rigor and Transparency:** The paper exhibits a level of intellectual honesty and completeness that is rarely seen. The authors explicitly analyze and discuss:
  - *The surrogate-to-target risk gap* (and prove Theorem 1 to bridge it).
  - *The randomized-to-deterministic PAC-Bayes discrepancy* (and empirically show that randomized posterior evaluation cut ECE in half).
  - *The inherent design bias of their simulation sandbox* (and resolve it via extensive validation on actual physical weight merging across 8 diverse datasets, showing the same hyperparameter and stability behaviors).

### Weaknesses and Constructive Suggestions
The paper is exceptionally strong, and there are no major technical flaws. However, the following constructive suggestions are provided to further expand and enrich this ambitious framework:
- **Scaling to Decoder-Only Large Language Models (LLMs):** While the Appendix details a latency benchmark for an 80-layer architecture (corresponding to LLaMA-70B), the paper would benefit from a future empirical study on physical LLM weight-merging across conversational and linguistic tasks (e.g., LLaMA-8B or LLaMA-70B). The linear-time OU exact inversion makes this extremely tractable, and seeing this validated in LLM weight-merging would be of outstanding interest.
- **Dynamic / Learned Covariance Kernels:** Currently, functional blocks are partitioned statically, and the lengthscale $\ell$ is held constant or scaled inversely by depth. A highly ambitious future direction would be to learn or adapt local lengthscales $\ell_l$ or block boundaries dynamically on-the-fly based on running activation distances or activation covariance matrices computed from the test-time stream.
- **Overhead of Stochastic Randomized Posterior Evaluation:** The authors show that ensembling 10 sampled coefficient profiles cuts SVHN ECE in half. However, running 10 forward passes introduces a $10\times$ computational overhead during test-time inference. Exploring single-pass uncertainty propagation or closed-form variance approximations to achieve this calibration boost would be highly valuable.

---

## Soundness and Methodology

### Rating: Excellent
The paper is technically flawless and highly rigorous. Every theoretical claim is backed by a solid mathematical derivation or proof. The assumptions behind Alquier's bound and Theorem 1 are clearly stated, and the authors go to great lengths to address potential corner cases (e.g. boundary truncation bias, truncated Gaussian paradox, and small-sample CKA variance). The empirical evaluation is exhaustive, including statistical significance (means and standard deviations across 3 independent seeds), hyperparameter sweeps, budget-convergence studies, and low-sample limits.

---

## Presentation

### Rating: Excellent
The writing is exceptionally clear, precise, and articulate. The authors strike an outstanding balance between dense theoretical proofs and practical implementation details. Visualizations are of publication-grade quality: Figure 6 strikes a powerful visual chord by contrasting the highly volatile, jagged unconstrained AdaMerging coefficients against the smooth, optimal GP-BayesMerge trajectories. The tables are extremely clean and include proper statistical metrics.

---

## Significance

### Rating: Excellent
This paper has the potential to fundamentally shift how the machine learning community approaches test-time model adaptation and parameter-space interpolation. By turning model merging from an empirical, heuristic-driven trial-and-error process into a principled, certifiable probabilistic science, GP-BayesMerge could inspire a new wave of research in foundation model adaptation, federated learning, and resource-efficient multi-task edge deployment.

---

## Originality

### Rating: Excellent
The paper is outstandingly original. The combination of PAC-Bayes generalization theory, continuous GP spatial priors over depth, linear-time OU exact inversion, and online multi-task Kronecker priors is highly creative and novel. The "delta" from prior heuristic and unconstrained test-time adaptation methods is massive, representing a major conceptual and theoretical leap.

---

## Questions for the Authors (to be addressed during rebuttal)
1. **Behavior of Online activation CKA under Severe OOD Shift:** Since $B_{\text{online}}$ is estimated dynamically using activations of calibration samples, how does this task-correlation matrix behave under extreme, adversarial, or noisy covariate shifts where intermediate features might be highly distorted? Does the shrinkage operator ($\epsilon = 0.1$) completely shield the joint prior from task correlation errors in these severe regimes?
2. **Continuous / Running Activation-Driven Covariance:** Under the OU prior, the precision matrix is strictly tridiagonal with closed-form entries. Is it possible to dynamically update these tridiagonal entries online based on the running feature variance or gradient norms of the layers on the calibration stream?

---

## Overall Recommendation

### 6: Strong Accept
**Justification:** This is a technically flawless, exceptionally written, and highly original paper that addresses a crucial and emerging paradigm in machine learning. By framing test-time model merging as a Bayesian inference problem, the authors successfully derive a unified, mathematically rigorous, and computationally lightweight framework that completely resolves the Overfitting-Optimizer Paradox. Supported by both a rigorous simulation sandbox and comprehensive physical weight-merging validation across 8 diverse datasets and deep models, the paper sets an exemplary standard for both theoretical and empirical research. It represents a major conceptual and paradigm-shifting contribution that the machine learning community will undoubtedly build upon.
