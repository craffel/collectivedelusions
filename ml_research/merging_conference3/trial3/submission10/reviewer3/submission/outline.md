# Paper Outline: ChebyMerge (Stable and Optimal Continuous Subspace Model Merging)

## 1. Abstract
- **Context:** Multi-task model merging via test-time adaptation (AdaMerging) improves task specialization but suffers from the "Overfitting-Optimizer Paradox."
- **Problem:** Unconstrained layer-wise coefficients overfit to local, high-frequency transductive noise on the test stream, causing representation collapse and poor generalization.
- **Proposed Solution:** ChebyMerge. Project the layer-wise coefficient space onto a low-dimensional orthogonal subspace spanned by Chebyshev polynomials of the first kind ($T_j(x)$).
- **Mathematical Advantages:** Minimax optimal uniform approximation, edge concentration matching deep network sensitivity profiles, and perfect numerical conditioning.
- **Key Results:** 3,527x improvement in Gram matrix condition number over monomial basis (PolyMerge) for cubic parameterization. Robust performance across 30 seeds, preventing representation collapse and achieving state-of-the-art generalization on MNIST, FashionMNIST, CIFAR-10, and SVHN.

## 2. Introduction
- **Model Merging Landscape:** Task Arithmetic and the promise of combining task-specific vectors.
- **The Shift to Test-Time Adaptation:** On-the-fly coefficient optimization via entropy minimization on unlabeled streams.
- **The Overfitting-Optimizer Paradox:** Unconstrained optimization has too many degrees of freedom, memorizing transductive noise and destroying model integrity.
- **The ChebyMerge Paradigm:** Framing coefficient parameterization as a continuous, spectral approximation problem. Chebyshev polynomials offer continuous, minimax-optimal interpolation, high resolution at the boundaries (matching early/late layer sensitivities), and exceptional numerical conditioning.
- **Summary of Contributions:**
  1. Rigorous formulation of ChebyMerge.
  2. Mathematical proof showing exponential ill-conditioning of monomials vs. bounded conditioning of Chebyshev polynomials.
  3. Extensive empirical validation over 30 independent random seeds, showcasing superior conditioning and resilience to representation collapse.

## 3. Related Work
- **Multi-Task Model Merging:** Task Arithmetic, Fisher merging, RegCalMerge.
- **Test-Time Adaptation (TTA):** Entropy minimization (TENT), AdaMerging.
- **Subspace and Polynomial Parameterization:** PolyMerge. Contrast our orthogonal Chebyshev formulation with PolyMerge's ill-conditioned monomial basis.

## 4. Methodology
- **Problem Formulation:** Task vectors, layer-wise coefficients, merged weights $\Theta_{\text{merged}, l}(\boldsymbol{\alpha})$.
- **Linear Domain Mapping:** Mapping layer indices $l \in \{0, \dots, L-1\}$ to the compact interval $[-1, 1]$.
- **Chebyshev Recurrence Relation:** Computing $T_j(x)$ and constructing the Chebyshev design matrix $\mathbf{C} \in \mathbb{R}^{L \times (d+1)}$.
- **Entropy Minimization Objective:** Mathematical definition of unsupervised test-time loss $\mathcal{L}_{\text{TTA}}$.
- **Theoretical Analysis of Numerical Conditioning:**
  - Defining the Gram matrix $\mathbf{X}^T\mathbf{X}$ and the condition number.
  - Theorem: Monomial Gram matrix condition number $\kappa(\mathbf{V}^T\mathbf{V}) = \mathcal{O}(4^d)$.
  - Theorem: Chebyshev Gram matrix condition number $\kappa(\mathbf{C}^T\mathbf{C}) \le B \ll \kappa(\mathbf{V}^T\mathbf{V})$.
  - Proof and mathematical exposition.

## 5. Experimental Results
- **Experimental Design:**
  - Model I: Convex Quadratic Distance under high-frequency alternating noise.
  - Model II: Coupled Non-Convex Rastrigin Stress-Test under multi-scale transductive noise (alternating, white Gaussian, Brownian drift) with layer sensitivity scaling.
- **Condition Number Comparison:** Table showing condition numbers for degrees 1, 2, and 3.
- **Quantitative Accuracies:**
  - Table of Model I results (average accuracies across 30 seeds).
  - Table of Model II results (average accuracies across 30 seeds, highlighting unconstrained Adam's collapse to 55.30% on SVHN and ChebyMerge's robust 85.25% average).
- **Qualitative Dynamics:**
  - Figure 1: Optimization loss trajectories under Model II.
  - Figure 2: Reconstructed merging coefficient profiles vs. true optimal sensitivity targets.

## 6. Conclusion and Future Work
- Recap of the mathematical rigor and empirical success of ChebyMerge.
- Propose future directions: Piecewise cubic B-spline manifolds for ultra-deep LLMs ($L \ge 32$) and localized spectral projections.
