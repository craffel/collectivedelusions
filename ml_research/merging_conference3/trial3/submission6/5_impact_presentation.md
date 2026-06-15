# Intermediate Review Task 5: Impact and Presentation (`5_impact_presentation.md`)

## 1. Major Strengths of the Paper
1. **Exceptional Theoretical Rigor:** The paper approaches model merging from first-principles loss-landscape geometry, rejecting unstable unsupervised heuristics in favor of formal quadratic joint-loss minimization. The mathematical formulation, proofs of strict convexity/uniqueness, and truncation error bounds are highly rigorous and correct.
2. **Innovative Subspace Projection:** By restricting parameter search to the $K$-dimensional subspace of task vectors, the authors brilliantly resolve the $O(D^2)$ space and $O(D^3)$ time complexity barriers of modeling full Hessian curvature. This allows them to compute the complete, non-diagonal, cross-parameter curvature with zero diagonal approximation in a computationally trivial manner ($O(K^3)$ per layer).
3. **Robust Gradient Subtraction Scheme:** The introduction of the gradient subtraction scheme is an outstanding contribution. It mathematically cancels out unperturbed residual expert gradients, resolving the $1/\epsilon$ noise amplification bottleneck of standard finite-differences and ensuring complete numerical stability regardless of checkpoint convergence.
4. **Outstanding Scientific Transparency:** The paper is incredibly honest and thorough in analyzing its limitations. Section 4.5 provides a masterclass in scientific analysis, detailing the local-global optimization gap, layer-wise ill-conditioning risks, and the block-Jacobi coupling mismatch. This transparency elevates the paper from a simple "benchmark-chasing" submission to a foundational theoretical study.
5. **Excellent Writing and Presentation:** The paper is superbly written, logical, and easy to follow. The LaTeX formatting, tables, and figures are professional and well-structured.

---

## 2. Areas for Improvement and Constructive Feedback
1. **Incorporate Sparse or L1 Regularization to Handle Ill-Conditioning:**
   The layer-wise analysis reveals extreme coefficient blowups at Layer 13 (the final LayerNorm, where SVHN solved to $-95.691$ and FashionMNIST to $-40.076$). The authors correctly attribute this to numerical ill-conditioning on low-parameter layers (which have extreme condition numbers). While they use Ridge (L2) regularization to stabilize the system, they also propose and evaluate Lasso (L1) regularization in Appendix B.6. L1 regularization successfully introduces sparsity, driving non-essential coefficients at Layer 13 to exactly **0.000** across all tasks. This prevents active cancellation blowups and serves as a mathematically sound layer-wise expert task selector. Integrating Lasso ACM more prominently in the main body would strengthen the manuscript.
2. **Empirical Validation of the Gauss-Seidel Coordinate Descent Scheme:**
   The authors note that perturbing the entire model simultaneously during calibration pollutes the layer-wise independent solvers with cross-layer Hessian terms, acting like a single-step block-Jacobi solver. While they elegantly detail a block Gauss-Seidel coordinate descent scheme in Appendix B.5 and include it in their physical experiments (Table 2), it achieves only **36.65% Joint Average accuracy**. This indicates that sequential coordination under coupled measurements is highly sensitive to cross-layer Hessian coupling block mismatches on deep networks. The paper would benefit from a deeper discussion analyzing why this elegant theoretical coordination collapses in practice.
3. **Phrasal Contradiction regarding ACM-Norm vs ACM-GlobalNorm:**
   In Section 4.3, under "Resolution of Layer-wise Sensitivity Flattening," the authors write: *"As a result, ACM-GlobalNorm outperforms ACM-Norm (58.89%) and achieves a Joint Average accuracy of 57.76%"*. This is a phrasal contradiction since 57.76% is lower than 58.89%. The authors should rephrase this to clarify that while ACM-GlobalNorm achieves a slightly lower overall average due to its global scaling constraint, it successfully outperforms ACM-Norm on individual tasks where global relative depth-wise sensitivity preservation is key (such as FashionMNIST at 70.02% vs 69.24% and CIFAR-10 at 77.05% vs 76.27%).

---

## 3. Overall Presentation Quality and Narrative Flow
The presentation quality is **excellent** (Excellent rating).
- **Structure:** The paper follows the standard, logical structure of high-quality ML conference submissions.
- **Narrative:** The overall narrative is compelling, transitioning from the critique of test-time adaptation heuristics, through the rigorous mathematical derivation of ACM, to the empirical validation and deep, insightful characterization of its limitations.
- **Clarity:** The writing style is professional, clear, and direct. The mathematical notation is consistent and elegant.
- **Visuals:** The tables are clear, and the figure captions provide immediate context.

---

## 4. Potential Impact and Significance
The significance of this contribution is **high** (Good to Excellent rating).
- **Advancing Capabilities:** ACM establishes a formal, mathematically rigorous foundation for multi-task parameter fusion, moving the field away from purely empirical, slow, and fragile test-time adaptation heuristics.
- **Influence on Future Research:** The detailed discussion of limitations (specifically the local-global optimization gap and block-Jacobi coupling mismatch) provides clear, actionable directions and "landmark insights" that are highly likely to influence future research in weight consolidation.
- **Practical Utility:** ACM is training-free and solves for optimal coefficients in under 5 seconds on a single GPU. This represents a massive speedup (hundreds of times faster) over Test-Time Adaptation methods like AdaMerging, which require extensive test-time forward-backward optimization. This makes ACM highly viable for real-time, resource-constrained edge environments.
