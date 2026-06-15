# 1. Summary of the Paper

## Main Topic and Approach
The paper introduces **ChebyMerge** (Stable and Optimal Continuous Subspace Model Merging), a mathematical framework for multi-task model merging under unsupervised test-time adaptation (TTA). Dynamic model merging methods like AdaMerging optimize separate merging coefficients for each layer using incoming unlabeled data streams. However, the authors identify a critical vulnerability: **the Overfitting-Optimizer Paradox**, where unconstrained layer-wise optimization overfits to transductive local sampling noise, leading to representation collapse.

To resolve this, the authors propose projecting the high-dimensional layer-wise merging coefficients onto a low-dimensional continuous subspace spanned by orthogonal **Chebyshev polynomials of the first kind**. Instead of optimizing $K \times L$ independent parameters, they optimize a small set of spectral parameters $\boldsymbol{\alpha}$ representing the coefficients of the Chebyshev expansion.

## Key Findings and Claims
1. **Overfitting-Optimizer Paradox:** Unconstrained layer-wise TTA optimizers overfit to high-frequency local sampling noise (transductive noise) on small unlabeled streams, driving the surrogate entropy objective to zero but severely degrading generalization performance.
2. **Monomial Ill-Conditioning (PolyMerge):** While constraining the coefficient space using standard monomial power bases (like PolyMerge) prevents transductive overfitting, it introduces severe numerical ill-conditioning. The condition number of the monomial Gram matrix scales exponentially as $\mathcal{O}(4^d)$ (exceeding 10,400 for cubic degree). This extreme ill-conditioning creates anisotropic "stiff" loss valleys, destabilizing gradient descent.
3. **ChebyMerge Benefits:** 
   - **Perfect Numerical Conditioning:** Bounds the condition number of the Chebyshev Gram matrix to a tiny constant ($\approx 2.95$ for cubic degree), yielding up to a **3,527x improvement** over monomial bases.
   - **Minimax Optimality:** Chebyshev polynomials provide near-optimal uniform approximation under the supremum norm ($L_\infty$).
   - **Implicit Boundary Sensitivity Matching:** Chebyshev roots and extrema cluster near the boundary layers (early and late layers), matching the physical sensitivity profile of deep models.
4. **Conditioning-Generalization Paradox & Controllable Spectral Decay (CSD):** The authors show that PolyMerge's extreme ill-conditioning acts as an accidental, uncontrolled spectral damping filter. ChebyMerge decouples conditioning from regularization and introduces CSD to explicitly and controllably scale down learning rates of high-frequency coefficients, outperforming PolyMerge.

## Explicitly Claimed Contributions and Evidence
1. **Mathematical Formulation of ChebyMerge:** Complete formulation of the orthogonal continuous subspace projection.
2. **Theoretical Proofs of Conditioning:** Rigorous proof of monomial exponential ill-conditioning (linked to Hilbert matrices) vs. bounded Chebyshev conditioning.
3. **Physically Grounded Simulation:** Design of a coupled, non-convex Rastrigin-type simulation environment emulating layer sensitivity profiles, inter-layer couplings, and multi-scale transductive noise of deep Vision-Language networks.
4. **Empirical Results (Synthetic & Physical):** 
   - Under the coupled non-convex stress-test (Model II) over 30 seeds, unconstrained Adam collapses to 78.67% average accuracy, while ChebyMerge ($d=2$) maintains 85.25%, and ChebyMerge-CSD ($d=2$) achieves 85.48%.
   - Physical evaluation on pre-trained CLIP ViT-B/32 on MNIST/SVHN shows that ChebyMerge-CSD ($d=2$) preserves generalization accuracy at 75.50% compared to PolyMerge ($d=2$) which collapses to 70.50% due to monomial stiffness.
   - Robustness sweeps over learning rates show that ChebyMerge exhibits graceful degradation under high learning rates, unlike PolyMerge which suffers from catastrophic collapse.
