# Mock Review: Parameter-Free Task-Space Projection for Dynamic Model Merging

## Overall Rating & Summary
*   **Overall Recommendation:** **5: Accept** (Bordering on **6: Strong Accept**)
*   **Soundness:** **Excellent**
*   **Presentation:** **Excellent**
*   **Significance:** **Excellent**
*   **Originality:** **Excellent**

### Summary of the Paper
This paper presents a minimalist, training-free, and data-free paradigm for sample-wise dynamic model ensembling and model merging. Guided by the principle of Occam's razor, the authors introduce **Parameter-Free Task-Space Projection (PFSR)**, which extracts task-specific centroids from pre-trained expert classifier weights using Singular Value Decomposition (SVD). Input features are projected onto these SVD-derived centroids, and a temperature-scaled Softmax gating function computes dynamic ensembling coefficients. 

To investigate whether decoupling task coordinates can eliminate cross-talk under task overlap, the authors explore an advanced extension, **Löwdin-Orthogonalized Task-Space Projection (OTSP)**, which uses Löwdin Symmetric Orthogonalization to create an orthonormal task coordinate basis in closed form offline.

Through a highly rigorous 10-seed simulation study and mathematical analysis, the authors prove:
1. Under symmetric task layouts, OTSP and PFSR make *exactly identical* routing decisions, rendering orthogonalization redundant.
2. Under isotropic representation noise, the coordinate-difference Signal-to-Noise Ratio (SNR) of OTSP and PFSR is mathematically identical: the margin-expansion benefit of OTSP is exactly cancelled by its online noise-amplification factor.
3. In asymmetric task environments under noise, OTSP systematically *underperforms* PFSR by 0.2% to 1.6% across sweeps due to the **Noise Amplification Penalty** (arising from near-singular overlap matrices) and the **Noise Spillover Penalty** (where noise from corrupted experts leaks into clean coordinate axes).

Additionally, the authors deconstruct baseline behaviors, revealing the **Orthogonal Masking Effect** (which flattens joint classification metrics in disjoint orthogonal setups), characterizing **Vectorization Collapse** under sample-wise vectorized ensembling ($B=1$), and identifying that simple zero-initialization of Softmax routing weights acts as an optimal maximum-entropy regularizer. Crucially, the draft successfully incorporates **Top-$k$ Sparse Gating** to preserve systems-level sparse VRAM execution, validates the methods on a real-world **ResNet-18 ImageNet-1K manifold**, and resolves anisotropic noise issues using **offline Mahalanobis covariance whitening**.

---

## Key Strengths

1. **Outstanding Theoretical Rigor:** The paper is mathematically mature and exceptionally robust. The closed-form proofs of Symmetric Equivalence (Section 3.7) and SNR Equivalence (Section 3.8) are elegant, correct, and provide a deep geometric explanation of why orthogonalization fails to yield routing benefits under symmetric noise.
2. **Intellectual Honesty & Self-Critical Deconstruction:** Unlike typical ML papers that propose a complex algorithm (such as OTSP) and try to force-fit its superiority, this paper uses OTSP as a theoretical lens to prove why the simpler, raw unorthogonalized projection (PFSR) is actually superior and more robust under noise. This "anti-novelty" approach is refreshing and represents a major win for Occam's razor.
3. **Meticulous Experimental Methodology:** The experiments are executed with exceptional statistical rigor (averaged over 10 seeds). The authors evaluate multiple deployment configurations ($B=256$, $B=1$), conduct dense sweeps over overlaps and noise scales, and test on skewed asymmetric layouts to isolate ensembling dynamics.
4. **Excellent Additions for Real-World Alignment:** The inclusion of Top-$k$ sparse gating (Section 4.5), the real-world ResNet-18 ImageNet-1K manifold evaluation (Section 4.6), and the anisotropic covariance whitening toy simulation (Section 4.7) completely resolve major potential criticisms regarding the practical evaluation gap, the VRAM trade-offs under soft gating, and the spherical noise assumption. This elevates the work to a very high level of maturity.

---

## Suggestions for Minor Revisions

The paper is exceptionally solid, theoretically complete, and ready for publication. We offer several constructive suggestions to further polish the manuscript for the camera-ready version:

### 1. Discuss and Explore Alternative Centroid Formulations
The SVD-based centroid extraction operates directly on the frozen classifier weights $W_k$. While this successfully avoids the sum-to-zero prototype cancellation (which causes the Naive Mean Centroid baseline to collapse to near-random guessing, as successfully demonstrated in Table 1), it assumes classifier weights are accessible.
*   **Actionable Suggestion:** The authors should discuss or briefly compare how centroids extracted via SVD on weights compare to centroids computed directly on sample representation activations (e.g., using K-Means or mean-normalized representations on the 64-sample calibration split). This would help generalize the method to settings where classifier weights are not directly available (e.g., intermediate layers in model merging).

### 2. Discuss the Interaction of Top-$k$ Gating and Self-Calibrated Temperature Scheduling
In Section 4.5, the authors introduce Top-$k$ sparse gating, and in Section 3.5, they propose self-calibrated temperature scheduling ($\tau_b = \gamma \cdot \text{std}_k(u'_{b})$).
*   **Actionable Suggestion:** The authors should briefly discuss how these two components interact in practice. Does Top-$k$ sparse gating require a different scaling multiplier $\gamma$ compared to standard Softmax, or does the self-calibrated temperature scheduling seamlessly adapt to the subset of top-$k$ coordinate distributions? The current brief explanation in Section 4.5 is helpful but could be expanded slightly to assist practitioners trying to deploy the complete pipeline.

### 3. Scale to Transformer Architectures in Future Work
While the ResNet-18 proof-of-concept on ImageNet-1K successfully demonstrates real-world manifold generalization, modern post-hoc ensembling and model merging are typically applied to large-scale Vision Transformers (ViTs) or Large Language Models (LLMs) with LoRA adapters.
*   **Actionable Suggestion:** Explicitly acknowledge in the conclusion or future work section that scaling PFSR/OTSP to massive transformer registries (e.g., merging dozens of LLM adapters on multi-task benchmarks like GLUE or MMLU) remains an exciting next step, referencing the proposed Data-Free Centroid Representation (DFCR) as the primary theoretical pathway.

---

## Detailed Assessment of Review Criteria

### Soundness: Excellent (Rate: Excellent)
The paper is technically flawless. The SVD-based centroid extraction is mathematically sound (preventing sum-to-zero cancellation), the Löwdin orthogonalization is correctly derived, and the absolute projection represents a methodologically crucial step to handle symmetric class margins. The mathematical proofs of symmetric equivalence and SNR cancellation are robust, complete, and verified by empirical results.

### Presentation: Excellent (Rate: Excellent)
The paper is exceptionally well-written, clearly structured, and easy to follow. The mathematical notation is clean and consistent. The authors' intellectual honesty regarding the Orthogonal Masking Effect, the Hard Gating Penalty, and the simulation sandbox significantly elevates the clarity and scholarly quality of the manuscript.

### Significance: Excellent (Rate: Excellent)
Simplifying dynamic model ensembling is a highly important and active problem. By showing that elegant closed-form linear algebra (PFSR) can match or exceed the routing accuracy of over-engineered, parameter-heavy parametric networks with zero training, the paper makes a highly valuable conceptual contribution. The addition of Top-$k$ sparse gating and the real-world ResNet-18 POC establish high practical utility for large-scale expert registries.

### Originality: Excellent (Rate: Excellent)
The SVD centroid formulation is elegant, and the application of Löwdin Symmetric Orthogonalization to representation-space task projection is highly creative and novel. Crucially, the originality lies in the deep deconstructive analysis: proving that the simpler method (PFSR) is mathematically and empirically superior to the complex extension (OTSP) due to noise amplification. This is a highly refreshing and original contribution in a literature dominated by hyper-parameterization.

---

## Questions and Clarifications for the Authors
1. **Heterogeneous Class Cardinalities:** If Expert $A$ classifies 2 classes (e.g., binary sentiment) and Expert $B$ classifies 1000 classes (e.g., ImageNet), the variance of their prototype distributions will be wildly different. How does the SVD centroid extraction perform under such high class imbalance, and how would you normalize the coordinates to prevent routing bias?
2. **Covariance Whitening Scalability:** The offline Mahalanobis covariance whitening ($\hat{\Sigma}^{-1/2}$) successfully resolves the anisotropic noise penalty in your toy simulation. In large-scale online settings, how expensive is it to estimate and invert the empirical covariance matrix $\hat{\Sigma}$? Is it possible to use a pre-computed diagonal covariance matrix instead?
