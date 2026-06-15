# 2. Novelty and Delta Analysis

## Analysis of Key Novel Aspects
The primary novelty of this paper is the parameterization of layer-wise merging coefficients using a continuous, low-degree polynomial of normalized layer depth:
$$\lambda_{k, l}(\boldsymbol{\alpha}) = \sum_{j=0}^d \alpha_{k, j} \cdot \left( \frac{l}{L-1} \right)^j$$
By representing the coefficients $\lambda_{k, l}$ as a function of the normalized depth $\bar{l} = \frac{l}{L-1}$, the method restricts the search space of test-time adaptation (TTA) from $L \times K$ parameters to $(d+1) \times K$ parameters.

This parameterization introduces several interesting ideas:
1. **Dimensionality Reduction for Derivative-Free Optimization:** High-dimensional search is notoriously difficult for derivative-free algorithms (like 1+1 ES) due to the curse of dimensionality. Projecting the parameter space from 56 down to 12 allows the zero-order search to function without collapsing, which is crucial since zero-order search requires zero activation caching and is thus highly SRAM-efficient.
2. **Smooth Parameterization Prior:** Constraining coefficients to a continuous, low-degree polynomial acts as a smooth regularization prior across layers. 
3. **Application to Model Merging under Quantization:** While model merging (e.g., AdaMerging) and post-training quantization (PTQ) have been explored separately, the combination of on-device TTA, continuous subspace mapping, and direct quantization-aware merging is a novel application area.

## Characterization of the 'Delta' from Prior Work
While the combination of these techniques is practical, the "delta" from prior work is relatively **incremental and heuristic**, especially when evaluated from a theoretical perspective:

1. **Similarity to Existing TTA and Model Merging Frameworks:** 
   The core test-time adaptation objective—unsupervised entropy minimization over unlabeled calibration streams—is directly borrowed from prior test-time adaptation works like **Tent** and **AdaMerging** (Yang et al., 2024). The optimization engine under quantization (Straight-Through Estimator and 1+1 ES) is also standard in quantization-aware optimization.
2. **Polynomial Constraints as Heuristic Regularization:**
   Representing layer-wise variables or hyperparameters as low-dimensional continuous functions of network depth (e.g., linear, polynomial, or exponential schedules) is a classic technique in deep learning (e.g., learning rate schedules, layer-wise weight decay scaling, or neural architecture search). The theoretical justification for choosing a polynomial function over other parametric curves (such as exponential decay or sinusoidal functions) is not rigorously established. The choice is a practical heuristic rather than a mathematically derived optimal constraint.
3. **Chebyshev Polynomial Scaling and Localized Splines:**
   The paper discusses orthogonal Chebyshev polynomials and piecewise quadratic splines in the appendix (Appendix B.7 and B.8). While mathematically elegant, these formulations are presented as a speculative "experimental blueprint" and "scaling pathway" rather than being implemented, tested, or theoretically analyzed in the main paper. Thus, they represent a conceptual extension rather than a verified contribution.
4. **Standard Evolutionary Strategies:**
   The algorithmic suggestions to bridge the 4-bit zero-order search gap (e.g., heavy-tailed Cauchy mutations, adaptive-population CMA-ES) are standard concepts in the evolutionary computation literature. Presenting them as "proposed advanced strategies" exaggerates their novelty, as their theoretical properties in non-smooth landscapes have been extensively studied in optimization theory.

In summary, the paper offers a clever, highly practical engineering parameterization that yields strong empirical and systems-level benefits on edge devices, but the fundamental mathematical novelty is limited, representing a straightforward application of polynomial fitting to an existing optimization space.
