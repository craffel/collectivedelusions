# 1. Comprehensive Summary of the Paper

## Main Topic and Motivation
The paper addresses the problem of **multi-task model merging**, where independently fine-tuned task-specific neural networks (experts) derived from a common pretrained base model are combined directly in their parameter space without additional joint training. The central motivation is to overcome the limitations of:
1. **Task Arithmetic (Uniform Scaling):** Which assumes uniform parameter sensitivities across layers, leading to representation interference and performance degradation when task vectors collide.
2. **Adaptive Test-Time Adaptation (TTA) Methods (e.g., AdaMerging, PolyMerge):** Which optimize layer-wise merging coefficients at deployment time using unsupervised entropy minimization. These methods suffer from high computational latency, transductive overfitting, and sacrificial task bias.

## Core Approach: Curvature-Aware Analytical Model Merging (ACM)
To resolve these issues, the paper proposes **ACM**, a training-free framework that formulates model merging as a quadratic minimization problem using local second-order Taylor expansions of the task-specific losses around each expert's parameter minimum. 
* **Subspace Projection:** To avoid the intractable $O(D^2)$ space and $O(D^3)$ time complexity of modeling the full $D$-dimensional Hessian curvature of a modern neural network, ACM projects the parameter space of each layer $l$ onto the extremely low-dimensional $K$-dimensional subspace spanned by the task vectors $v_1^l, \dots, v_K^l$.
* **Full Non-Diagonal Curvature:** Within this low-dimensional subspace, ACM computes and inverts the full, non-diagonal, cross-parameter Hessian curvature with **zero diagonal approximation**. Unlike prior curvature-aware methods (such as Fisher Merging) which rely on a diagonal approximation of the Hessian to remain tractable, ACM models full cross-term parameter couplings.
* **Closed-Form Solution:** The layer-wise minimization problem yields an exact closed-form analytical solution under Ridge (L2) or Lasso (L1) regularization: 
  $$\Lambda^{l, *} = (A^l + \gamma I)^{-1} (b^l - d^l)$$
* **Gradient Subtraction Scheme:** To estimate the projected Hessian matrix $A^l$ and vectors $b^l, d^l$ efficiently, the authors propose a cheap finite-difference scheme over a tiny calibration batch (e.g., 32 samples). Crucially, they explicitly compute and subtract the unperturbed expert gradient $g_{0}$ to cancel out residual gradients and prevent the $1/\epsilon$ noise amplification inherent in standard finite-difference schemes.
* **Scale and Global Normalization:** To prevent a single high-magnitude task from dominating the merging process (sacrificial task bias), the authors introduce **ACM-Norm** (layer-wise trace-normalization) and **ACM-GlobalNorm** (global trace-normalization across all layers to preserve relative depth-wise sensitivities).

## Key Findings and Claims
1. **Simulation Robustness:** In simulated environments sweeping over 30 seeds, ACM outperforms PolyMerge by $+1.69\%$ and AdaMerging by $+8.11\%$ in coupled, dense, non-convex loss landscapes.
2. **Physical Validation on ViT-Tiny:** On a pretrained Vision Transformer backbone evaluated across four image classification tasks (MNIST, FashionMNIST, CIFAR-10, SVHN) in a low-data regime, the authors report:
   * **ACM-GlobalNorm** achieves **57.76%** Joint Average accuracy, outperforming diagonal Fisher Merging (56.03%) and polynomial test-time adaptation (38.96%).
   * **Vanilla ACM** achieves **60.89%** Joint Average accuracy.
   * **Lasso ACM** achieves **60.67%** and successfully prunes non-essential task vectors at highly ill-conditioned layers.
3. **Failure of TTA on Physical ViTs:** The authors find that standard test-time adaptation methods (AdaMerging, PolyMerge) experience severe generalization collapse (accuracies dropping to 55.42% and 38.96%) on physical models due to transductive overfitting on small unlabeled test streams.
4. **Active Cancellation Mechanism:** Layer-wise coefficient analysis reveals huge negative and positive scaling factors at Layer 13 (global LayerNorm), which the authors interpret as a representation-space interference cancellation mechanism.

## Explicitly Claimed Contributions (with Evidence)
* **Theoretical Rigor:** Establishing a formal quadratic optimization framework over task-specific local Hessian geometries. *Evidence: Detailed mathematical derivations in Section 3.2, 3.3, and Appendix B.*
* **Low-Dimensional Subspace Projection:** Proving that capturing full, non-diagonal Hessian interactions is tractable when restricted to the task-vector subspace. *Evidence: Complexity analysis showing $O(K^2)$ storage and $O(K^3)$ inversion per layer in Section 3.7.*
* **Training-Free Closed-Form Solution:** Deriving exact analytical solutions under Ridge/Lasso regularization that eliminate test-time adaptation, transductive overfitting, and sacrificial task bias. *Evidence: Formulations in Section 3.4, 3.5, and Appendix B.6.*
* **Empirical Characterization & Integrity:** Validating ACM via simulation sweeps and physical ViT-Tiny benchmarks, and selecting hyperparameters strictly via an unsupervised validation split heuristic to eliminate test-set leakage. *Evidence: Tables 1 & 2, and sensitivity studies in Appendix C.*
