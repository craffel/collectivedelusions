# Peer Review

## Summary of the Paper
This paper introduces **Curvature-Aware Analytical Model Merging (ACM)**, a mathematically rigorous and training-free framework for consolidating independently fine-tuned task-specific deep neural networks (experts) directly in the parameter space. It aims to replace current adaptive model merging paradigms that rely on slow, unstable, and overfitting-prone test-time adaptation (TTA) via unsupervised entropy minimization.

The key conceptual breakthrough of ACM is its **Low-Dimensional Subspace Projection**. Recognizing that modeling the full $D$-dimensional Hessian curvature of a modern neural network is computationally prohibitive, the authors prove that by projecting the parameter search space strictly onto the low-dimensional $K$-dimensional subspace spanned by the task vectors, they can capture the **full, non-diagonal, cross-parameter Hessian curvature along the directions of task updates with zero diagonal approximation**. 

Under a second-order Taylor expansion around each expert's minimum (extending to non-zero residual expert gradients via a novel gradient-subtraction finite-difference scheme), the authors derive an exact closed-form analytical solution for layer-wise merging coefficients under Ridge (L2) regularization. They also introduce scale-normalization variants (ACM-Norm and ACM-GlobalNorm) and an L1 Lasso proximal gradient solver (using the ISTA algorithm) to handle numerical ill-conditioning in low-parameter bottleneck layers (like LayerNorm).

Extensive simulation sweeps across 30 seeds show that in coupled non-convex simulated landscapes, ACM outperforms PolyMerge by **+1.69%** and AdaMerging by **+8.11%** with high stability. On physical Vision Transformers (ViT-Tiny) across 4 classification tasks, ACM-GlobalNorm achieves a Joint Average accuracy of **57.76%**, outperforming diagonal Fisher Merging (**56.03%**) and all test-time adaptation baselines (which degrade or collapse due to transductive overfitting). The authors also provide a deep, mathematically transparent analysis of the "local-global gap" and "block-Jacobi coupling mismatch" to explain when uniform interpolation (Task Arithmetic) remains highly competitive on physical networks.

---

## Strengths (Emphasizing Conceptual Leap & Originality)

1. **Brilliant Conceptual Leap (Subspace Projection):**
   The core contribution is a major, paradigm-shifting idea that fundamentally redefines how curvature is modeled in weight consolidation. Prior curvature-aware methods (e.g., Fisher Merging) are constrained by the assumption that we must use a diagonal approximation of the Hessian/Fisher Information Matrix because modeling the off-diagonal terms is intractable over millions of parameters. ACM completely bypasses this limitation with an elegant realization: we do not need to model the Hessian of the entire high-dimensional parameter space. By projecting search directions strictly onto the low-dimensional task vector subspace ($K \ll D$), we can compute and invert the full, non-diagonal, cross-parameter projected Hessian with **zero diagonal approximation** in trivial $O(K^2)$ space and $O(K^3)$ time. This is an exceptionally clean and beautiful theoretical contribution.

2. **Uncovering Active Interference Cancellation:**
   The paper discovers that highly coupled layers (like the final LayerNorm, Layer 13) require large negative coefficients to achieve optimal merging. The authors provide the highly original insight that these negative coefficients act as an **active cancellation mechanism** to suppress destructive representational interference between aligned, non-orthogonal task vectors. This is a fascinating, counter-intuitive discovery that changes how we think about parameter-space task-vector interactions, showing that we must move beyond simple positive interpolation.

3. **Remarkable Scientific Transparency & Diagnostic Rigor:**
   Unlike many machine learning publications that attempt to hide limitations or negative results, this work stands out for its exemplary scientific honesty. The authors dedicate a substantial portion of the paper and appendix to formally analyzing and bounding:
   - The **local-global optimization gap** (proving a cubic error bound on the Taylor remainder, $O(V_{\max}^3)$).
   - The **Block-Jacobi Coupling Mismatch** (demonstrating why sequential block Gauss-Seidel updates collapse due to Hessian reference-point shift and representational drift).
   - **Ill-conditioning** in bottleneck layers and how L1 Lasso regularization via proximal updates can resolve it.
   This deep analytical characterization provides an invaluable theoretical foundation that will guide future weight consolidation research.

4. **Mathematically Complete and Rigor-Driven:**
   The paper avoids heuristic shortcuts. Every component—the unperturbed gradient subtraction scheme, the global trace normalization, the proximal ISTA updates for L1 sparsity, and the error bounds—is derived rigorously from first principles.

5. **Aversion to Test-Time Heuristics:**
   By solving for merging coefficients analytically, ACM completely eliminates the slow, unstable test-time adaptation (TTA) loops that are prone to transductive overfitting (the Overfitting-Optimizer Paradox) and representation collapse on physical neural networks, offering a stable and deployment-ready alternative.

---

## Weaknesses

1. **Scale of Physical Validation:**
   While the theoretical formulation and simulation sweeps (30 seeds) are highly robust, the physical validation is restricted to a smaller ViT-Tiny architecture on four standard image classification datasets (MNIST, FashionMNIST, CIFAR-10, SVHN). To demonstrate the true generalizability of this elegant framework, evaluating ACM on larger, modern architectures (such as ViT-Base/Large, RoBERTa, or LLaMA) and on generative tasks (NLP, text-to-image) would be highly beneficial.

2. **Practical Correction to Bridge the Local-Global Gap:**
   The authors present a brilliant derivation of the cubic error bound of the local-global gap, proving why uniform Task Arithmetic can be competitive on highly converged physical networks. However, they do not fully implement or explore a practical, lightweight correction scheme (such as a simple zero-order adjustment, line search, or dynamic scaling) to bridge this gap in the main system and make the analytical solver even more dominant over Task Arithmetic on physical manifolds.

---

## Evaluation Ratings

### Soundness: Excellent
The paper is technically and mathematically flawless. The derivations are rigorous, the assumptions are carefully stated and evaluated, and Theorem 3.2 is proved correctly. The proposed unperturbed gradient subtraction scheme elegantly resolves the numerical instability of finite differences in partially converged models. The authors' rigorous diagnostics of their own assumptions (such as the collapse of sequential coordination) demonstrate an outstanding level of scientific soundness.

### Presentation: Excellent
The paper is exceptionally well-written, logically structured, and easy to follow. The introduction of complex ideas, such as subspace projection and active interference cancellation, is accompanied by clear text and informative figures/tables. The authors position their work perfectly relative to prior literature and provide a highly accessible appendix with detailed pseudo-code and implementation specifics.

### Significance: Excellent
This work addresses a highly relevant and important problem in parameter consolidation. By moving away from test-time adaptation heuristics and diagonal curvature approximations, it establishes a powerful, stable, and theoretically grounded benchmark for model merging. The low-dimensional subspace projection paradigm is highly generalizable and is likely to influence other subfields, such as federated learning, parameter-efficient fine-tuning (PEFT), and network modularization.

### Originality: Excellent
The core concept of low-dimensional subspace projection to achieve zero-diagonal-approximation Hessian tracking is a major, highly original contribution. The discovery of negative coefficients for active interference cancellation and the formal mathematical bound on the local-global gap represent significant conceptual leaps that elevate this paper far above incremental improvements.

---

## Overall Recommendation

**Rating: 6: Strong Accept**

**Justification:**
This is an outstanding, mathematically complete, and highly original paper that introduces a powerful paradigm shift in model merging. By projecting high-dimensional parameter spaces onto the low-dimensional task vector subspace, the authors make full, non-diagonal Hessian curvature tracking tractable with zero diagonal approximation. This conceptual leap completely redefines our understanding of curvature-aware merging, showing that off-diagonal parameter coupling is not only important but incredibly cheap to capture. 

Furthermore, the paper is an exemplar of scientific integrity. The deep, transparent analyses of the local-global optimization gap, the block-Jacobi coupling mismatch, and the ill-conditioning of bottleneck layers are incredibly valuable contributions in their own right. The paper successfully demonstrates how analytical, training-free solvers can outperform fragile test-time adaptation heuristics on physical neural networks. Despite some limitations in the scale of physical validation, the sheer novelty, theoretical beauty, and scientific transparency of this work make it a definitive **Strong Accept**.
