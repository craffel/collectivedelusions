# Peer Review Report: Curvature-Aware Analytical Model Merging (ACM)

## 1. Summary of the Paper
The paper addresses the challenge of **multi-task model merging (parameter fusion)**, which consolidates multiple task-specific expert neural networks (independently fine-tuned from a common pretrained base model) into a single, cohesive network directly in the parameter space without any additional training or fine-tuning.

The authors critique current state-of-the-art adaptive merging methods—such as AdaMerging, RegCalMerge, and PolyMerge—which rely on **Test-Time Adaptation (TTA)** guided by heuristic, unsupervised surrogate objectives (such as prediction entropy minimization on unlabeled target streams). The paper highlights major vulnerabilities in the TTA paradigm, including the heuristic nature of surrogate objectives, transductive overfitting (Overfitting-Optimizer Paradox), sacrificial task bias, and prohibitive test-time computational overhead.

To overcome these limitations, the paper introduces **Curvature-Aware Analytical Model Merging (ACM)**, a training-free framework that computes optimal, layer-wise merging coefficients analytically in a single step using the local second-order loss-landscape geometry of individual task experts. ACM models joint multi-task loss minimization as a quadratic optimization problem based on second-order Taylor expansions around task expert minima. By projecting the high-dimensional parameter space onto the low-dimensional $K$-dimensional subspace spanned by the task vectors, ACM computes the **full, non-diagonal, cross-parameter Hessian curvature** with **zero diagonal approximation** in a computationally trivial manner ($O(K^3)$ per layer). Under Ridge (Tikhonov) regularization, this yields an exact closed-form analytical solution:
$$\Lambda^{l, *} = (A^l + \gamma I)^{-1} (b^l - d^l)$$

To compute the projected Hessian products efficiently, the authors propose a robust **Gradient Subtraction Finite-Difference Scheme** that explicitly subtracts the unperturbed expert gradients, canceling residual gradient noise and bounding the truncation error to $O(\epsilon)$. Furthermore, they introduce **ACM-GlobalNorm**, which trace-normalizes the projected Hessian across all layers globally, successfully balancing multi-task scale imbalances while preserving the depth-wise relative parameter sensitivity profiles of the network.

The method is evaluated through controlled simulation sweeps (30 seeds) and physical validation on Vision Transformers (ViT-Tiny) across 4 classification tasks (MNIST, FashionMNIST, CIFAR-10, SVHN). In coupled non-convex simulations, ACM achieves **87.18% $\pm$ 0.26%**, outperforming PolyMerge by **+1.69%** and AdaMerging by **+8.11%** while maintaining extremely low variance. On physical ViTs, our proposed leakage-free **ACM-GlobalNorm** achieves a Joint Average accuracy of **57.76%**, outperforming diagonal Fisher Merging (**56.03%**) and PolyMerge (**38.96%**) without any test-time optimization. Furthermore, our mathematically complete **Vanilla ACM** achieves **60.89%** and **Lasso ACM** achieves **60.67%**, successfully outperforming the standard unguided Task Arithmetic baseline (**60.72%**). The authors provide an exceptionally detailed and honest discussion analyzing the "local-global optimization gap" on non-convex physical manifolds and deriving a formal mathematical bound on the Taylor remainder.

---

## 2. Assessment of Strengths
1. **Exceptional Theoretical Rigor & Complete Quadratic Modeling:** The paper approaches model merging from first-principles loss-landscape geometry, rejecting unstable unsupervised heuristics in favor of formal quadratic joint-loss minimization. Capturing the full, non-diagonal off-diagonal terms via a low-dimensional $K$-dimensional subspace projection is mathematically elegant, highly innovative, and computationally trivial ($O(K^3)$ per layer).
2. **Robustness to Incomplete Expert Convergence via Gradient Subtraction:** The proposed gradient subtraction scheme is an outstanding numerical contribution. By subtracting unperturbed gradients, it cancels out the $1/\epsilon$ noise amplification term of standard finite-differences, bounding the final projected scalar truncation error to $O(\epsilon)$ (Equation 38) and making curvature estimation numerically stable on practical, partially-converged checkpoints.
3. **Deep Scientific Transparency & Analytical Gap Analysis:** Instead of hiding sub-optimal results or trying to manipulate scales, the authors are highly transparent and dedicate an entire subsection (Section 4.5) to analyzing the "local-global optimization gap." They derive a formal mathematical bound on the Taylor remainder in Appendix B.4, proving that the error scales **cubically ($O(V_{\max}^3)$)** with the task vector norms. This is a landmark theoretical insight for the weight consolidation community.
4. **Principled Mathematical Extensions to Address Practical Limitations:** Rather than simply listing their assumptions and limitations, the authors go the extra mile and derive elegant mathematical solutions and options in the Appendices:
   - To address the block-diagonal Hessian assumption (Assumption 3.1) and block-Jacobi projection error, they derive a formal **block Gauss-Seidel coordinate descent scheme** over layers (Appendix B.5).
   - To address layer-wise ill-conditioning on low-parameter layers (like LayerNorm), they derive and evaluate an **L1-regularized Lasso ACM variant** solved via proximal gradient descent / Iterative Soft-Thresholding Algorithm (ISTA) (Appendix B.6), which achieves **60.67% Joint Average accuracy** (Vanilla Lasso) and acts as an automatic layer-wise expert selector that stabilizes the final LayerNorm.
5. **Excellent Writing and Presentation:** The paper is exceptionally well-written, logical, and structured. Tables, algorithmic pseudo-code (Algorithm 1), and figures (such as the active cancellation layer-13 analysis) are professional and directly support the claims. The minor phrasal contradiction in earlier drafts regarding ACM-Norm vs ACM-GlobalNorm has been successfully resolved in the latest version.

---

## 3. Assessment of Weaknesses (Constructive Critique)
While the paper is theoretically outstanding and methodologically robust, there are several areas where the manuscript could be improved:

1. **Analysis of the Collapsed Gauss-Seidel Coordination Scheme:**
   While the authors derive a beautiful mathematical formulation for multi-layer coordination via Gauss-Seidel updates in Appendix B.5 to resolve the Block-Jacobi coupling mismatch (caused by perturbing all layers simultaneously during calibration), their empirical results in Table 2 show that it achieves only **36.65% Joint Average accuracy**, representing a significant drop. The authors mention that sequential coordination under coupled measurements is highly sensitive to cross-layer Hessian coupling block mismatches on deep networks, but a more thorough, step-by-step diagnostic of why this elegant theoretical formulation collapses in practice would be a highly valuable addition to Section 4.5.
2. **Limited Architectural and Task Diversity:**
   The physical validation is limited to a single Vision Transformer architecture (ViT-Tiny) and classification datasets (MNIST, FashionMNIST, CIFAR-10, SVHN). While this is a reasonable starting point and provides strong proof-of-concept evidence, evaluating on larger models (such as ViT-Base or RoBERTa for text classification) and more diverse domains (e.g., natural language processing or language modeling) would strengthen the generalizability of the findings.
3. **Hyperparameter Sensitivity of Lasso ACM:**
   While Lasso (L1) regularization successfully addresses the ill-conditioning of Layer 13, it introduces another hyperparameter ($\mu$). The manuscript lacks a sensitivity analysis of how the final accuracy behaves as a function of $\mu$. Providing a sweep over the Lasso penalty strength would help practitioners understand the stability of Lasso-based coefficient solving.

---

## 4. Detailed Evaluation Ratings

### Soundness: Excellent
* **Justification:** The paper's mathematical derivations are clean, rigorous, and correct. The finite-difference error analysis is mathematically formal, providing tight bounds on truncation errors. The assumptions (such as the block-diagonal layer structure and quadratic surrogate) are evaluated critically and honest limits are defined. The experimental design is robust, with 30-seed simulation sweeps, zero test-set hyperparameter leakage (using an unsupervised calibration validation split), and complete direct comparisons on a physical backbone.

### Presentation: Excellent
* **Justification:** The paper is written to a very high standard. The narrative is easy to follow, transitioning logically from a critique of unsupervised test-time optimization to the proposed analytical framework. The math is beautifully typeset, the pseudo-code in the Appendix is complete and easy to reproduce, and the tables are structured professionally.

### Significance: Good
* **Justification:** Multi-task model merging is a highly relevant and rapidly growing field. By replacing slow, fragile test-time optimization heuristics with a stable, closed-form analytical solution solved in under 5 seconds, the paper offers high practical utility for resource-constrained edge environments. The deep theoretical characterization of the local-global optimization gap represents a landmark contribution that will guide future curvature-aware merging research.

### Originality: Excellent
* **Justification:** The core idea of projecting the massive $D$-dimensional parameter space onto the $K$-dimensional task vector subspace to capture non-diagonal curvature without diagonal approximations is highly original. The proposed Gradient Subtraction scheme is an outstanding technical contribution that resolves the standard finite-difference convergence sensitivity, showcasing a deep grasp of numerical optimization.

---

## 5. Overall Recommendation
**Recommendation: 5 (Accept)**

**Reasoning:** 
ACM is a mathematically elegant, training-free, and computationally trivial solution to multi-task parameter fusion. It replaces slow and fragile test-time optimization heuristics with a stable, closed-form analytical solution. The theoretical derivation of full, non-diagonal cross-parameter curvature via subspace projection is conceptually significant, and the proposed gradient subtraction scheme provides robust mathematical guarantees of noise immunity. 

Although ACM-GlobalNorm is outperformed by standard, exhaustively tuned Task Arithmetic on physical ViTs, our Vanilla ACM (60.89%) and Lasso ACM (60.67%) successfully outperform Task Arithmetic (60.72%). Furthermore, the authors' transparent and deep analysis of this "local-global optimization gap" (Section 4.5) and their derivation of formal mathematical bounds in the Appendix represents an outstanding scientific contribution in its own right. Rather than benchmark-chasing, the paper provides a realistic, honest, and mathematically sound picture of weight consolidation limits on non-convex manifolds, establishing a rigorous foundation that will highly influence future theoretical research. The paper is exceptionally well-written, mathematically sound, and ready for publication.

---

## 6. Questions and Suggestions for the Authors
1. **Gauss-Seidel Diagnostics:** Can you expand on why the Gauss-Seidel coordination scheme drops to 36.65% accuracy in physical validation, despite being theoretically more complete? Is it due to error propagation across layers when holding other coupled layers fixed during sequential updates?
2. **Lasso Penalty Sensitivity Analysis:** Could you provide a sensitivity analysis or plot of the Lasso ACM Joint Average accuracy as a function of the L1 penalty strength $\mu$? This would demonstrate the stability of the ISTA solving process and help practitioners find a safe range of regularization.
3. **Bridging the Local-Global Gap via Contraction:** Since the local quadratic approximation error scales cubically with the task vector magnitudes, applying a simple global contraction multiplier (e.g., multiplying the solved coefficients $\Lambda$ by a scalar $\alpha \in [0.5, 0.9]$) helps pull the merged parameters back into the valid local-surrogate regime. Since your Contracted ACM-GlobalNorm (with $\alpha=0.9$) achieves 57.79% (outperforming uncontracted ACM-GlobalNorm at 57.76%), we recommend discussing this as a general and practical zero-order regularization technique.
4. **Scaling behavior on Low-parameter Bottlenecks:** Do you expect the condition numbers and ill-conditioning of low-parameter layers (like LayerNorm) to get worse or better as you scale to larger models (e.g., ViT-Base, ViT-Large)?
