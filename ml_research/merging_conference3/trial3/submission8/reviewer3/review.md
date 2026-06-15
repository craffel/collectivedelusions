# Peer Review

## Summary of the Paper
The paper addresses a fundamental vulnerability in existing optimization-based test-time model merging methods (such as AdaMerging), which the authors term the **Overfitting-Optimizer Paradox**. At test time, unconstrained first-order optimization of layer-wise merging coefficients on small, unlabeled calibration batches aggressively fits transductive noise. This causes the learned coefficients to exhibit highly volatile, high-frequency spatial oscillations across adjacent layers, leading to catastrophic generalization collapse on unseen test data (particularly on the SVHN dataset).

To resolve this, the authors introduce **GP-BayesMerge**, a mathematically rigorous Gaussian Process PAC-Bayes framework for robust test-time model merging. By reformulating test-time adaptation as a Bayesian inference problem, the authors apply Alquier's linear PAC-Bayes bound to derive a Kullback-Leibler (KL) complexity penalty over the coefficient space. Placing a continuous Gaussian Process prior over normalized network depth allows the KL penalty to simplify to a quadratic form governed by the GP precision matrix $\Sigma_{\ell}^{-1}$. This operator mathematically unifies distance-from-initialization (proximity/diagonal elements) and spatial smoothness (off-diagonal elements) constraints.

The authors present several key extensions:
1. **MT-GP-BayesMerge:** A joint multi-task prior modeled using a Kronecker product ($\Sigma_{\text{joint}} = B \otimes \Sigma_{\ell}$), where the task correlation matrix $B$ is estimated online and data-free using activation Centered Kernel Alignment (CKA) similarities on target calibration streams.
2. **Ornstein-Uhlenbeck (OU) Kernel:** Proving that the OU kernel provides a strictly tridiagonal precision matrix with an exact closed-form analytical inverse, enabling linear-time $O(L)$ assembly and bypassing expensive dense matrix inversions.
3. **Non-Stationary Block-Wise GP Prior:** Decoupling correlations across functional block boundaries to capture sharp, localized architectural transitions.
4. **Randomized Posteriors Evaluation:** Sampling coefficients from the posterior distribution at test time to serve as a representation-space dropout and improve calibration.

The framework is evaluated under a high-fidelity non-convex simulation sandbox calibrated to replicate a 12-layer Vision Transformer (ViT-B/16), as well as on actual physical weight merging of pre-trained CLIP ViT-B/32 and ViT-L/14 models across 8 diverse visual classification benchmarks.

---

## Strengths

1. **Rigorous and Elegant Theoretical Formulation:**
   The paper stands out for its strong theoretical foundation. Instead of relying on empirical, hyperparameter-heavy trial-and-error, the authors derive their spatial regularizer directly from first-principles PAC-Bayes generalization theory. The simplification of the KL divergence under a continuous GP depth prior into a single quadratic precision-matrix form $\Sigma_{\ell}^{-1}$ is mathematically elegant and highly principled.

2. **Unified Dual-Purpose Spatial Operator:**
   By showing that the precision matrix $\Sigma_{\ell}^{-1}$ naturally acts as both a localized weight-decay proximity penalty (via positive diagonal elements) and a finite-difference Laplacian smoother (via negative adjacent off-diagonal elements), the paper elegantly unifies previously disconnected empirical constraints under a single, cohesive mathematical operator.

3. **Exceptional Empirical Performance and Stability:**
   The proposed GP-BayesMerge and MT-GP-BayesMerge frameworks achieve outstanding classification accuracies under both simulated and physical weight-merging deployments (e.g., reaching $82.35\%$ and $82.68\%$ average accuracy on physical weights). Crucially, the continuous spatial prior completely resolves the Overfitting-Optimizer Paradox, eliminating SVHN accuracy collapse and dramatically reducing seed-to-seed standard deviation (e.g., physical SVHN standard deviation drops from $\pm 1.84\%$ under unconstrained AdaMerging to just $\pm 0.35\%$).

4. **Multi-Task Kronecker Prior with Online CKA Estimation:**
   The joint multi-task extension is highly innovative. Estimating task correlations dynamically online using activation CKA is a clever, training-free, and data-free approach that captures functional task relationships and resolves representational conflicts without requiring original training data.

5. **Linear-Time Computational Scalability ($O(L)$):**
   The derivation of the closed-form analytical inverse for the Ornstein-Uhlenbeck kernel precision matrix is a monumental contribution for scalability. Assembling a strictly tridiagonal precision matrix in $O(L)$ linear time completely bypasses the $O(L^3)$ dense covariance inversion cost, enabling perfect scalability to ultra-deep networks (as verified in the 2048-layer latency benchmark).

6. **Rigorous Calibration Boost via Randomized Evaluation:**
   The authors successfully bridge the theoretical-to-empirical discrepancy by evaluating the randomized PAC-Bayes classifier. Sampling merging coefficients from the optimized posterior distribution ($\Lambda \sim Q$) provides a substantial calibration cushion, cutting Expected Calibration Error (ECE) on physical SVHN in half ($8.45\% \to 4.12\%$), which represents a brilliant empirical result.

---

## Weaknesses

1. **Inherent Design Bias in the Simulation Sandbox:**
   The simulation sandbox generates the ground-truth optimal coefficient configurations using a decaying spatial covariance matrix ($\Sigma_{\text{true}}$ with parameter $0.5^{|l-l'|}$). Because the ground truth is pre-defined to be spatially smooth, this setup incorporates an inherent design bias that naturally favors spatially-smooth regularizers like GP-BayesMerge. While the physical weight experiments are successful and address this concern, the simulated results are somewhat circular.

2. **Inadequate Seed Count for Statistical Rigor:**
   The authors report means and standard deviations across **only 3 random seeds** (42, 100, 2026). While a low seed count is standard for expensive deep training runs, test-time adaptation in model merging is computationally extremely cheap and fast. The authors state that GP-BayesMerge converges in fewer than 50 steps, taking less than 0.15 seconds per adaptation run. Since running the entire pipeline takes a fraction of a second, evaluating the framework on only 3 seeds represents a missed opportunity for empirical rigor. Running at least 10 or 20 random seeds would provide much more robust statistics and allow for rigorous significance testing (e.g., t-tests).

3. **Complete Lack of Empirical Validation for the CCV Tuning Algorithm:**
   The authors propose **Calibration Cross-Validation (CCV)** as a fully unsupervised test-time hyperparameter tuning protocol and outline its pseudo-code in Algorithm 1 (Appendix D.4). However, **there is absolutely no empirical data or results in the paper validating this algorithm**. It is unclear whether CCV can actually find optimal hyperparameters in practice, what performance is achieved under CCV-tuned parameters, or how it compares to default/oracle configurations. Proposing an unsupervised tuning algorithm without reporting any empirical experiments to verify its effectiveness is a notable gap.

4. **Absence of Evaluation on Non-Vision Modalities (Decoder-only LLMs):**
   Although the authors benchmark GP Covariance Inversion Latency up to 80 layers (referencing LLaMA-70B) in Appendix C.2, all actual physical weight merging experiments are restricted to Vision Transformer (ViT-B/32 and ViT-L/14) image encoders. Parameter-space model merging is an exceptionally prominent paradigm in the NLP and LLM community. Validating the framework on language generation, instruction tuning, or reasoning tasks under decoder-only LLM weight-merging setups would significantly strengthen the empirical generalizability of the method.

5. **Missing Baseline Hyperparameter Tuning Details:**
   The paper is silent on how the hyperparameters for the baseline methods (such as the disjoint smoothing parameters of RegCalMerge and the polynomial degree of PolyMerge) were selected and tuned. For a fair empirical comparison, the baselines should be optimally tuned on the same calibration batches to ensure a fair evaluation.

---

## Questions and Constructive Suggestions for the Authors

1. **Empirical Validation of CCV (Algorithm 1):**
   Can the authors provide empirical results demonstrating the classification performance of GP-BayesMerge when lengthscale $\ell$ and regularization $\alpha$ are tuned online and unsupervised using the Calibration Cross-Validation (CCV) algorithm?

2. **Statistical Significance over Additional Seeds:**
   Given the exceptionally low adaptation latency ($<0.15$ seconds), can the authors re-evaluate the physical weight-merging benchmarks over 10 or 20 independent random seeds, and conduct statistical significance tests (e.g., paired t-tests) comparing GP-BayesMerge against the best baselines?

3. **Decoder-only LLM Weight Merging:**
   Are there plans to evaluate GP-BayesMerge on physical weight merging of autoregressive decoder-only LLMs (such as merging fine-tuned LLaMA or Mistral experts)? Doing so would greatly expand the impact of the paper.

4. **Baseline Hyperparameter Tuning:**
   Can the authors clarify how the hyperparameters of RegCalMerge and PolyMerge were tuned? Were they swept and optimized for each dataset to ensure a fair comparison?

---

## Ratings

* **Soundness:** **Good**
  The mathematical derivations, proofs, and physical weight-merging evaluations are highly sound and convincing. However, the rating is bounded at "Good" rather than "Excellent" due to the circular design of the simulation, the low seed count (3 seeds) for a fast adaptation method, and the complete lack of empirical validation for the proposed CCV tuning algorithm.
* **Presentation:** **Excellent**
  The paper is exceptionally well-written, mathematically precise, and engaging. The figures are high-quality, and the foundational proofs and remarks are beautifully organized in the Appendix.
* **Significance:** **Excellent**
  The paper addresses a highly relevant problem (test-time model merging) and provides a highly efficient, training-free adaptation paradigm with zero storage overhead, which has high potential impact for resource-constrained edge devices.
* **Originality:** **Excellent**
  The transition from high-dimensional weight-space PAC-Bayes bounds to a low-dimensional merging coefficient control space, along with the continuous GP prior formulation, Kronecker multi-task joint prior, and analytical OU tridiagonal precision matrix, are highly original and creative.

## Overall Recommendation
**5: Accept**
This is a technically solid, highly original, and well-written paper that addresses an important problem in test-time model merging. It unifies disparate empirical heuristics under a continuous GP prior derived from first-principles PAC-Bayes theory, and provides strong empirical gains and stability. While there are some empirical gaps (the low seed count, the lack of LLM evaluation, and the lack of empirical validation for the unsupervised CCV tuning algorithm), the paper's theoretical elegance and successful physical weight-merging results on Vision Transformers make it a very strong addition to the conference.
