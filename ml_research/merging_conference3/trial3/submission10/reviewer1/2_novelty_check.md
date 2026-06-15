# 2. Novelty Check

## Key Novel Aspects
1. **Model Merging as a Spectral Approximation Problem:** Framing layer-wise model-merging coefficient parameterization as a continuous spectral approximation problem, and introducing Chebyshev polynomials of the first kind to map the discrete layer domain $[-1, 1]$.
2. **Identification of Key Paradoxes:** 
   - **Overfitting-Optimizer Paradox:** Exposing how unconstrained layer-wise optimization under Test-Time Adaptation (TTA) overfits to local transductive sampling noise, destroying generalization and leading to representation collapse.
   - **Conditioning-Generalization Paradox:** Highlighting the subtle tension where PolyMerge's severe, exponential monomial ill-conditioning actually acts as an accidental, implicit spectral damping filter, protecting it from overfitting.
3. **Controllable Spectral Decay (CSD):** Proposing a principled mechanism to separate numerical conditioning from regularization. Instead of relying on accidental numerical errors from ill-conditioning, CSD maintains perfect conditioning and applies controllable low-pass filtering.

## The 'Delta' from Prior Work
- **From AdaMerging (Lu et al., 2023):** AdaMerging introduced layer-wise dynamic coefficient optimization at test time. The delta is that AdaMerging uses unconstrained optimization which suffers from the Overfitting-Optimizer Paradox (representation collapse). ChebyMerge restricts the parameters to a low-dimensional subspace.
- **From PolyMerge (2024):** PolyMerge introduced continuous subspace parameterization using low-degree polynomials to filter out noise. However, PolyMerge uses a standard monomial power basis ($1, \bar{l}, \bar{l}^2, \dots$), leading to exponential ill-conditioning. The delta is the use of the orthogonal Chebyshev basis, which reduces the Gram matrix condition number for a cubic degree by 3,527x (from 10,406 to 2.95), resolving the numerical stiffness.

## Characterization of Novelty
From a conceptual and paradigm perspective:
1. **Incremental Algorithm, Significant Theoretical Insights:** The core algorithmic idea—projecting layer-wise coefficients onto a polynomial subspace—is directly inherited from PolyMerge (2024). From a methodological perspective, changing a monomial basis to a Chebyshev basis is a standard technique in numerical analysis and polynomial interpolation. Thus, the algorithmic "delta" is somewhat incremental.
2. **Original Conceptual Contributions:** The true conceptual leaps in the paper lie in the theoretical formulation of the **Conditioning-Generalization Paradox** and the development of **Controllable Spectral Decay (CSD)**. The insight that monomial ill-conditioning acts as a hidden, accidental regularizer is a highly profound and original contribution that deepens our understanding of optimization dynamics in deep neural networks.
3. **Principled Decoupling:** Proposing to decouple numerical stability from regularization (via CSD) is an ambitious and original principle that moves beyond heuristic-driven deep learning engineering. It provides a new perspective on how to design stable, regularized continuous parameterizations.
