# 3. Soundness and Methodology Evaluation

## Clarity of Description
The methodology is described with exceptional clarity and mathematical rigor. The paper is outstandingly structured, walking the reader from the general multi-task merging setting to the Taylor expansion, the subspace projection, the closed-form derivations under Ridge/Lasso regularization, and the finite-difference estimation scheme. The appendix provides comprehensive proofs, complexity analyses, and pseudo-code that make the methodology highly transparent.

## Appropriateness of Methods
* **Formulating the Subspace Projection:** Restricting the parameter search space to the $K$-dimensional subspace of task vectors is a mathematically sound and clever way to make full-Hessian curvature modeling computationally tractable ($O(K^2)$ memory and $O(K^3)$ inversion complexity).
* **Gradient Subtraction:** The gradient subtraction scheme is highly appropriate and theoretically sound. The authors formally prove in Appendix B.1 that it cancels out unperturbed residual gradients, resolving the $1/\epsilon$ noise amplification bottleneck of standard finite-difference approximations.
* **The Local-Global Gap Mismatch (Inappropriateness of Quadratic Expansion):** A critical methodological flaw is the reliance on a **local** second-order Taylor expansion to solve a **global** model merging problem. Model merging involves large parameter steps ($W(\Lambda) - W_k$), moving far from the individual task expert's minima. On highly non-convex, converged physical neural manifolds, the local quadratic surrogate breaks down completely over these large distances. The authors prove in Appendix B.4 that this local-global approximation error scales **cubically** ($O(V_{\max}^3)$) with the task vector norm. This mathematical bound directly exposes why the local curvature-aware method is fundamentally mismatched with highly fine-tuned (converged) real-world models—where task vectors are large—and explains why it struggles to outperform a simple uniform interpolation baseline (Task Arithmetic).

## Potential Technical Flaws and Suspicions

### 1. The Ill-Conditioning and Large Coefficient Artifact (Layer 13)
The authors highlight that at Layer 13 (the final LayerNorm layer), Scale-Normalized ACM (ACM-Norm) solves the merging coefficients to extremely large positive and negative values (e.g., MNIST: $91.516$, SVHN: $-95.691$, FashionMNIST: $-40.076$). They present this as a "profound physical and mathematical insight" representing a "representation-space active cancellation mechanism" that "orthogonalizes the update pathways."

This interpretation is highly suspicious and represents a potential technical flaw:
* Layer 13 is the global LayerNorm layer, which has only 384 parameters in ViT-Tiny. 
* As the authors admit in Section 4.5, because of this low parameter count, the projected Hessian matrix $A^{13}$ has an extremely small trace and exhibits extreme ill-conditioning, with a condition number exceeding $10^4$.
* Solving a highly ill-conditioned system over a tiny, stochastic 32-sample calibration set naturally results in wild, massive coefficient blowups. 
* This is not a "beautiful active cancellation mechanism"—it is a standard numerical artifact of severe ill-conditioning and overfitting to calibration noise. When regularization (Ridge or Lasso) is applied to stabilize this layer, these wild coefficients are suppressed or zeroed out, but the final Joint Average accuracy of these regularized models (e.g., ACM-GlobalNorm at 57.76% or Lasso ACM-GlobalNorm at 57.52%) actually drops significantly below standard Task Arithmetic (60.72%). This strongly suggests that these huge coefficients were numerical noise rather than a meaningful representational feature.

### 2. The Complete Collapse of Multi-Layer Coordination (Gauss-Seidel)
The authors propose an iterative block Gauss-Seidel coordinate descent scheme (Equation 31) in Appendix B.5 to resolve the cross-layer coupling mismatch. However, in the physical experiments, this coordinated variant completely collapses, achieving a Joint Average accuracy of only **36.65%**. The authors dedicate significant space to diagnosing this collapse (representational drift cascade, Hessian reference-point collapse, out-of-distribution LayerNorm blowups). 

While their honesty is commendable, the complete failure of this coordination scheme exposes a major methodological weakness: ACM's layer-wise decoupled assumption (Assumption 3.1) is fundamentally incompatible with the deep, sequential nature of modern networks. The moment they attempt to resolve the cross-layer coupling sequentially, the representation space shifts, rendering the local Hessian approximations evaluated at the original experts completely invalid. This indicates that ACM is incapable of modeling sequential, cross-layer parameter interactions in deep networks.

## Reproducibility
The paper exhibits an exceptionally high standard of reproducibility. The authors document:
* Exact task-expert training hyperparameters (AdamW, epochs, learning rates, weight decay, dataset size).
* Detailed optimization steps, learning rates, and iterations for all test-time adaptation baselines.
* The exact calibration batch size ($M=32$), perturbation scale ($\epsilon=10^{-3}$), and the candidate hyperparameter sweep ranges.
* A complete procedural algorithm in pseudo-code (Algorithm 1, Appendix A).
* Detailed code-level descriptions of how they resolved PyTorch's autograd graph disconnection for TTA baselines.
