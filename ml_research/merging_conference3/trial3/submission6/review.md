# Peer Review Report: Curvature-Aware Analytical Model Merging (ACM)

## 1. Summary of the Paper
The paper addresses the task of **multi-task model merging (parameter fusion)**, which consolidates multiple task-specific expert neural networks (fine-tuned from a common pretrained base) into a single unified network directly in parameter space without training. It critically analyzes existing **Test-Time Adaptation (TTA)** methods (e.g., AdaMerging, RegCalMerge, PolyMerge) that utilize unsupervised entropy minimization at deployment, highlighting major vulnerabilities: lack of joint loss guarantees, transductive overfitting, sacrificial task bias, and high computational latency.

To resolve these, the paper introduces **Curvature-Aware Analytical Model Merging (ACM)**, a training-free framework that derives layer-wise merging coefficients analytically in a single step. ACM models joint multi-task loss minimization as a quadratic optimization problem based on second-order Taylor expansions around task expert minima. By projecting the high-dimensional parameter space onto the low-dimensional $K$-dimensional subspace spanned by the task vectors, ACM computes the **full, non-diagonal, cross-parameter Hessian curvature** with zero diagonal approximation. Under Ridge (Tikhonov) regularization, this yields an exact closed-form analytical solution:
$$\Lambda^{l, *} = (A^l + \gamma I)^{-1} (b^l - d^l)$$

To compute the projected Hessian products, the paper proposes a robust **Gradient Subtraction Finite-Difference Scheme** that explicitly subtracts the unperturbed expert gradients, canceling residual gradient noise and bounding the truncation error to $O(\epsilon)$.

The method is evaluated through extensive simulations (30 seeds) and physical validation on Vision Transformers (ViT-Tiny) across 4 datasets (MNIST, FashionMNIST, CIFAR-10, SVHN). In simulations, ACM outperforms AdaMerging by up to +8.11% in coupled non-convex settings. On physical ViTs, our proposed **ACM-GlobalNorm** (57.30%) outperforms diagonal Fisher Merging (55.62%) and PolyMerge (55.40%). Crucially, standard Task Arithmetic (with scale 0.5 tuned on test set) achieves 62.28%, and the authors provide an exceptionally detailed and honest discussion analyzing this "local-global optimization gap" alongside other structural limitations.

---

## 2. Assessment of Strengths
1. **Exceptional Theoretical Rigor:** The paper approaches model merging from first-principles loss-landscape geometry, rejecting unstable unsupervised heuristics in favor of formal quadratic joint-loss minimization. The mathematical formulation, proofs of strict convexity/uniqueness, and truncation error bounds are highly rigorous and correct.
2. **Innovative Subspace Projection:** By restricting parameter search to the $K$-dimensional subspace of task vectors, the authors brilliantly resolve the $O(D^2)$ space and $O(D^3)$ time complexity barriers of modeling full Hessian curvature. This allows them to compute the complete, non-diagonal, cross-parameter curvature with zero diagonal approximation in a computationally trivial manner ($O(K^3)$ per layer).
3. **Robust Gradient Subtraction Scheme:** The introduction of the gradient subtraction scheme is an outstanding contribution. It mathematically cancels out unperturbed residual expert gradients, resolving the $1/\epsilon$ noise amplification bottleneck of standard finite-differences and ensuring complete numerical stability regardless of checkpoint convergence.
4. **Outstanding Scientific Transparency:** The paper is incredibly honest and thorough in analyzing its limitations. Section 4.5 provides a masterclass in scientific analysis, detailing the local-global optimization gap, layer-wise ill-conditioning risks, and the block-Jacobi coupling mismatch. This transparency elevates the paper from a simple "benchmark-chasing" submission to a foundational theoretical study.
5. **Excellent Writing and Presentation:** The paper is superbly written, logical, and easy to follow. The LaTeX formatting, tables, and figures are professional and well-structured.

---

## 3. Assessment of Weaknesses (Constructive Critique)
While the paper is technically excellent and methodologically robust, there are several areas where the theoretical analysis and evaluation could be strengthened:

1. **Lack of Formal Bounds on the Local-Global Optimization Gap:**
   The authors transparently discuss the "local-global optimization gap," noting that on highly non-convex, fully converged physical neural manifolds, the local quadratic Taylor approximation (taken around individual task expert minima) breaks down over the large step sizes required to reach the merged parameter state. However, the paper lacks a formal mathematical bound on this approximation error. Deriving a bound on the Taylor remainder as a function of the task vector norms $\|v_k\|_2$ would formalize this gap and provide a clear theoretical condition for when curvature-aware methods are expected to underperform uniform interpolation.
2. **Layer Block-Diagonal Assumption and Block-Jacobi Projection Error:**
   Assumption 3.1 assumes that cross-layer second-order interactions are negligible (block-diagonal Hessian). In deep networks, layers are highly coupled, and early-layer shifts modify late-layer representation distributions. The authors note that their calibration scheme perturbs the entire model simultaneously (capturing cross-layer coupling) but solves for coefficients independently, acting as a single-step block-Jacobi solver. However, the paper would benefit from a formal analysis or coordinate-descent correction (e.g., Gauss-Seidel updates over layers) to fully resolve this projection error.
3. **Ill-Conditioning and Large Scaling on Low-Parameter Layers:**
   The layer-wise analysis reveals extreme coefficient blowups at Layer 13 (the final LayerNorm, where SVHN solved to $-1.494$ and CIFAR-10 to $3.409$). The authors correctly attribute this to numerical ill-conditioning on low-parameter layers. While they use Ridge (L2) regularization to stabilize the system, they do not investigate other regularization schemes like Lasso (L1) or elastic net. L1 regularization could introduce sparsity, which has physical meaning (selecting only the most sensitive task experts for certain low-parameter layers), preventing excessive scaling factors and active cancellation blowups.
4. **Baseline Incompleteness in Physical Evaluation:**
   RegCalMerge is evaluated in simulations (Table 1) but missing from the physical validation (Table 2). To ensure absolute empirical completeness, the authors should include RegCalMerge's physical results or explicitly explain its omission.

---

## 4. Overall Recommendation
**Recommendation: 5 (Accept)**

**Reasoning:** 
ACM is a mathematically elegant, training-free, and computationally trivial solution to multi-task parameter fusion. It replaces slow and fragile test-time optimization heuristics with a stable, closed-form analytical solution. The theoretical derivation of full, non-diagonal cross-parameter curvature via subspace projection is conceptually significant, and the proposed gradient subtraction scheme provides robust mathematical guarantees of noise immunity. 

Although ACM-GlobalNorm is outperformed by standard, exhaustively tuned Task Arithmetic on physical ViTs, the authors' transparent and deep analysis of this "local-global optimization gap" (Section 4.5) represents an outstanding scientific contribution in its own right. Rather than benchmark-chasing, the paper provides a realistic, honest, and mathematically sound picture of weight consolidation limits on non-convex manifolds, establishing a rigorous foundation that will highly influence future theoretical research. The paper is exceptionally well-written, mathematically sound, and ready for publication.

---

## 5. Questions and Suggestions for the Authors
1. **Deriving Taylor Remainder Bounds:** Can you derive a formal mathematical bound on the Taylor remainder of the local quadratic surrogate as a function of the task vector norms $\|v_k\|_2$? This would mathematically formalize the local-global gap and show how the error scales with expert convergence.
2. **Iterative Block-Jacobi Coordinate Descent:** Since solving independently under coupled measurements introduces a projection error, have you considered an iterative block-coordinate descent scheme (like performing a few block Gauss-Seidel updates over the solved layer coefficients) to resolve the multi-layer Hessian coupling?
3. **L1 (Lasso) Regularization:** Have you experimented with L1/Lasso regularization instead of Ridge regularization? Given that low-parameter layers (like LayerNorm) are highly ill-conditioned, L1 regularization could zero out non-essential expert task vectors, acting as an automatic layer-wise expert selector and preventing excessive scaling.
4. **Missing Physical Baseline:** Why was RegCalMerge excluded from the physical validation table (Table 2)? Including its physical results would make the empirical comparison fully complete.
