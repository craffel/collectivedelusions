# Peer Review: FlatQ-Merge

## 1. Summary of the Paper
The paper presents **FlatQ-Merge** (Flatness-Aware Quantization-Aware Model Merging), a systematic investigation into how the loss landscape flatness of task-specific expert neural networks (pre-trained with Sharpness-Aware Minimization, or SAM) governs their resilience to post-training quantization (PTQ) and test-time blending coefficient optimization. 

The framework consists of four main phases:
1. Pre-training task-specific experts using SAM across 5 perturbation radii ($\rho \in \{0.0, 0.01, 0.05, 0.1, 0.2\}$).
2. Merging the experts linearly using layer-wise dynamic coefficients $\Lambda \in [0, 1]^{L \times K}$ initialized uniformly at $0.3$.
3. Applying per-channel symmetric uniform post-training quantization (8-bit or 4-bit) to the merged weights, using the Straight-Through Estimator (STE) during the backward pass.
4. Optimizing the coefficients $\Lambda$ at test-time via joint entropy minimization using a small, unlabeled calibration batch.

The paper uncovers a precision-dependent "Flatness-Robustness Synergy" where flatness yields negligible gains under 8-bit quantization but provides a substantial **+7.44%** absolute multi-task accuracy improvement under extreme 4-bit quantization. It also documents an "Over-Perturbation Threshold" at $\rho \ge 0.1$ where expert task vectors lose their specialized identity due to "representation convergence," causing performance to collapse. Additionally, the authors show that a simple static uniform merging of flat experts (NaiveUniform, $\rho = 0.05$) outperforms complex test-time adaptation on sharp SGD-trained experts ($\rho = 0.0$) by **+6.03%** absolute accuracy, demonstrating that pre-merging landscape geometry is significantly more critical to success than downstream adaptation complexity.

---

## 2. Strengths and Weaknesses

### Strengths:
* **High Practical and Systems Relevance**: Bridging model merging and post-training quantization is a highly practical and relevant pursuit for edge AI deployment. The demonstrated $8\times$ peak RAM reduction during adaptation compared to post-hoc optimization is a notable systems-level advantage.
* **Extensive and Rigorous Empirical Evaluation**: The paper evaluates its method across 3 independent random seeds and provides incredibly detailed multi-axial sweeps. The inclusion of several highly competitive baselines and advanced ablations (DARE, Stochastic Weight Averaging, Softmax combination, TENT-style high-dimensional adaptation, and direct empirical weight-space flatness measurements) is exemplary and far exceeds the typical standard for empirical machine learning submissions.
* **Intelligent Geometric Profiling**: The analysis of the "Over-Perturbation Threshold" using the pairwise cosine similarity of task vectors is highly insightful and provides a satisfying, original explanation (representation convergence) for why excessively large SAM radii collapse performance.
* **Exceptional Transparency**: Section 5.1 is highly commendable for its thoroughness and scientific honesty, proactively discussing constraints regarding data scale, absolute accuracy gaps, activation quantization, task incongruence, and architectural generalization.

### Weaknesses:
* **Significant Theoretical Gaps and Logical Gaps**:
  * **Hessian Mismatch**: In Section 3.1, the paper attempts to prove that minimizing the weight-space Hessian via SAM bounds the coefficient-space Hessian $H_{\Lambda}$ using the projection $H_{\Lambda} = T^T H_{\theta} T$. However, the Hessian $H_{\theta}$ in the projection represents the second derivative of the *test-time joint prediction entropy loss* evaluated at the *merged, quantized parameters*. In contrast, SAM pre-training minimizes the Hessian of the *supervised task-specific training loss* evaluated at the *individual expert parameters*. The paper treats these two Hessians as interchangeable, which is a major logical gap. Because of the non-convexity of deep network loss landscapes, linear interpolation, and non-linear rounding, these Hessians can be completely different.
  * **Taylor Approximation of Discrete Noise**: The paper relies on a local, second-order Taylor expansion to justify the benefits of flatness under quantization noise. However, weight quantization (especially 4-bit) is a large, non-local, discrete coordinate-wise rounding noise. An infinitesimal Taylor expansion is heuristically informative but mathematically insufficient to serve as a rigorous guarantee without bounding the higher-order Taylor remainder or performing a non-local analysis.
* **Lack of Convergence Analysis under STE**: The optimization of coefficients in the piecewise-constant, non-differentiable quantized loss landscape via the Straight-Through Estimator is a pure heuristic. While Section 3.4 provides a good empirical discussion of stability, the paper lacks any analytical treatment of convergence or error bounds under STE gradient mismatch in this non-convex, discontinuous setting.
* **Low-Data and Scale Constraints**: While well-documented as a limitation, the empirical evaluation remains confined to a tiny sandbox (ViT-Tiny trained on only 512 images per task). The resulting absolute multi-task accuracies are low ($\sim 30\%$ under 4-bit), making it difficult to assess whether these findings and thresholds hold under full-scale, fully converged models.

---

## 3. Soundness
* **Rating**: **Fair**
* **Justification**:
  The empirical methodology is exceptionally sound. The authors have carefully run multi-seed evaluations with standard deviations, compared against strong baselines, and validated their architectural design choices (e.g., independent clipping vs. Softmax combination).

  However, from a theoretical perspective, the soundness is limited. The claim of providing a "rigorous mathematical connection" that "proves" SAM pre-training flattens the test-time adaptation landscape is mathematically flawed due to the incongruence between the training-time supervised task Hessians and the test-time unsupervised joint entropy Hessian. Furthermore, using a local second-order Taylor expansion to model large, discrete 4-bit quantization rounding noise is technically inappropriate without bounding the remainder term. To elevate the soundness to "Good" or "Excellent," the paper must address these mathematical gaps, qualify its theoretical claims as heuristic justifications, and explicitly state the strong, unstated assumptions required for their derivations to hold.

---

## 4. Presentation
* **Rating**: **Excellent**
* **Justification**:
  The paper is beautifully written, exceptionally structured, and easy to follow. The mathematical notation is clean and consistent. Figures and tables are beautifully designed, with self-contained, highly informative captions. The authors do an outstanding job of presenting complex empirical sweeps and accompanying them with clear, deep physical explanations (e.g., the explanation of representation convergence in Section 4.4 and the SWA vs. SAM comparison in Section 4.8). The limitations section is exemplary in its depth and academic transparency.

---

## 5. Significance
* **Rating**: **Good**
* **Justification**:
  The paper addresses a highly important and active problem: deploying merged models onto resource-constrained edge hardware. The insight that pre-merging expert loss landscape geometry is far more critical to low-precision merging success than the complexity of downstream adaptation algorithms represents a valuable paradigm shift for future research.

  However, the significance is currently bottlenecked by the toy-scale nature of the experiments (ViT-Tiny, 512 images per task). Because the absolute accuracies are very low, practitioner-level utility is limited. Showing that these findings scale to larger architectures (e.g., ResNets, larger ViTs) or standard-scale pre-training would dramatically enhance the paper's significance.

---

## 6. Originality
* **Rating**: **Good**
* **Justification**:
  While model merging in flat minima and post-training quantization in flat minima have been explored separately, the intersection and specific study of their precision-dependent interaction under extreme compression (4-bit) is highly original. The investigation of the "Over-Perturbation Threshold" and its geometric explanation via "representation convergence" is a highly creative and insightful contribution. The empirical comparisons isolating SAM's adversarial objective from simple trajectory averaging (SWA) and verifying weight-space flatness directly via weight perturbations are also highly original and well-conceived.

---

## 7. Overall Recommendation
* **Rating**: **3: Weak Reject**
* **Justification**:
  This paper has significant merits, particularly in its exceptionally thorough, honest, and highly ablated empirical evaluation. The discovery that pre-merging flatness dominates downstream adaptation and the geometric profiling of the over-perturbation threshold are outstanding contributions.

  However, as a paper that places significant emphasis on its "Theoretical Foundation" (Section 3.1), it falls short of the necessary mathematical rigor. The logical gap of treating training-time task-specific supervised Hessians at expert points as equivalent to test-time joint unsupervised entropy Hessians at merged quantized points is a critical flaw in their "proof." Additionally, relying on local Taylor expansions for non-infinitesimal 4-bit rounding noise is technically loose.

  Because these theoretical assertions are central to the paper's methodology and framing, they must be revised and corrected. I recommend a **Weak Reject** in its current form to allow the authors to:
  1. Soften their theoretical claims, framing the derivations as heuristic projection guidelines rather than "rigorous mathematical proofs."
  2. Explicitly discuss the mathematical gaps and outline the strong, simplifying assumptions required for the Hessian projections to hold.
  3. Formally address the non-local nature of quantization noise and explain the limitations of local Taylor expansions in the extreme compression regime.
  4. Perform a validation experiment on at least one moderately larger backbone or dataset subset to demonstrate the scalability of the observed thresholds.

  If the authors can successfully execute these revisions and reconcile their mathematical derivations with the physical realities of the optimization pipeline, this paper would be a highly competitive and strong candidate for acceptance.
