# 4. Experimental Setup and Claims Verification

This check is performed from an **empiricist's** perspective, focusing on the strength of baselines, the realism of datasets, the soundness of statistical evaluation, and whether the claims are genuinely supported by the empirical data.

## Baselines Evaluation
The paper evaluates a comprehensive and highly rigorous set of baselines:
1. **Task Arithmetic (Static Uniform)**: A crucial, non-adaptive baseline. Many model-merging papers fail to compare their adaptive methods against a simple static uniform scaling. Including this baseline allows the paper to honestly evaluate whether test-time adaptation actually provides a net benefit.
2. **Unconstrained AdaMerging (Layer-wise)**: Represents the state-of-the-art in unconstrained test-time model-merging. This is the main baseline used to demonstrate the Overfitting-Optimizer Paradox.
3. **TV and L2 Regularized AdaMerging**: Verifies whether simple spatial smoothing (Total Variation) or weight decay (L2) can resolve unconstrained overfitting without projecting onto a continuous subspace.
4. **PolyMerge (Monomial Subspace)**: The state-of-the-art continuous subspace model-merging method. This is the direct competitor of ChebyMerge, representing identical continuous curves but parameterized with standard monomials.

This is an exceptionally complete and fair baseline set. No obvious baselines are missing.

## Datasets and Evaluation Environments
- **Model I (Convex Landscape)**: Evaluates optimization and convergence in a clean, localized environment with high-frequency noise.
- **Model II (Coupled Non-Convex Stress-Test)**: Designed to emulate modern deep neural networks. It incorporates non-diagonal covariance coupling ($\boldsymbol{\Sigma}$), layer-wise sensitivity scaling (foveated boundaries), a highly non-convex Rastrigin landscape (introducing multiple local minima), and multi-scale transductive noise (alternating, white, and Brownian drift). 
- **Real-World Physical CLIP ViT-B/32 Merging**: Validates the methods using actual weights and visual features of pre-trained models. The task vectors are computed from real checkpoints, and classification is performed on standard MNIST and SVHN datasets. 

## Statistical Soundness
- **Multiple Seeds**: The simulated experiments (Model I and Model II) are evaluated across **30 independent random seeds** (seeds 42 to 71, inclusive), reporting both the mean and standard deviation. This provides strong statistical confidence.
- **Significance of Results**: The standard deviations for continuous subspace methods in Model II are around $0.7\%$ to $1.3\%$. The $+0.81\%$ improvement of ChebyMerge ($d=2$) over Task Arithmetic is statistically visible, and the $+6.58\%$ improvement over unconstrained Adam is highly significant.

## Verification of Central Claims

### Claim 1: Unconstrained TTA model merging overfits to transductive noise (Overfitting-Optimizer Paradox)
- **Supported? Yes.** 
  - In Model II, unconstrained Adam's accuracy collapses to $78.67\%$ (far below Task Arithmetic's $84.44\%$), with SVHN dropping catastrophically to $55.30\% \pm 17.80\%$.
  - In the physical CLIP experiment, unconstrained AdaMerging drops from Task Arithmetic's $81.50\%$ down to $78.00\%$. The final prediction entropy is successfully minimized (4.6046 vs. 4.6047), confirming that the optimizer overfits to local transductive batch noise, driving down the surrogate loss while destroying generalization.

### Claim 2: Monomial subspaces suffer from severe numerical ill-conditioning
- **Supported? Yes.** 
  - The Gram matrix condition numbers computed for $L=12$ are:
    - Linear ($d=1$): Monomial $16.40$ vs. Chebyshev $2.54$ ($6.5\times$ improvement)
    - Quadratic ($d=2$): Monomial $389.31$ vs. Chebyshev $2.75$ ($141.8\times$ improvement)
    - Cubic ($d=3$): Monomial $10,406.63$ vs. Chebyshev $2.95$ ($3,527.4\times$ improvement)
  - This perfectly validates the theoretical analysis of exponential ill-conditioning.

### Claim 3: Monomial ill-conditioning acts as an accidental implicit regularizer (Conditioning-Generalization Paradox)
- **Supported? Yes.**
  - Under Model II with $d=3$, PolyMerge achieves $85.31\% \pm 1.33\%$, while standard unregularized ChebyMerge achieves $84.63\% \pm 1.72\%$. 
  - Standard ChebyMerge has "too much freedom" because its landscape is perfectly isotropic, allowing the optimizer to update higher-order coefficients and fit a small amount of transductive noise. PolyMerge's extreme ill-conditioning (condition number $> 10,400$) effectively freezes updates to the higher-degree components, acting as an implicit spectral filter (early stopping).

### Claim 4: Controllable Spectral Decay (CSD) decouples conditioning from regularization and achieves state-of-the-art results
- **Supported? Yes.**
  - Under Model II, **ChebyMerge-CSD ($d=2$) achieves $85.48\% \pm 1.13\%$**, outperforming both standard ChebyMerge ($85.25\%$) and PolyMerge ($85.39\%$).
  - In the physical CLIP ViT-B/32 experiment, **ChebyMerge-CSD ($d=2$) achieves $75.50\%$ accuracy**, outperforming standard ChebyMerge ($74.00\%$) and PolyMerge ($70.50\%$) by substantial margins (+5.00% over PolyMerge).
  - This empirically demonstrates that separating numerical conditioning from regularization is superior. Rather than relying on accidental ill-conditioning, explicitly decaying higher-order learning rates provides highly stable, predictable, and performant spectral filtering.

### Claim 5: ChebyMerge provides superior optimization stability and learning rate robustness
- **Supported? Yes.**
  - In the learning rate sweep (Table 6), as the learning rate is increased to $\eta = 2 \cdot 10^{-2}$, PolyMerge's accuracy collapses catastrophically from $81.00\%$ to $66.00\%$ (a $-15.00\%$ drop) because larger steps on its stiff, highly anisotropic landscape trigger divergence.
  - In contrast, ChebyMerge maintains $71.00\%$ and ChebyMerge-CSD maintains $70.00\%$, demonstrating graceful degradation and proving that a well-conditioned landscape acts as a critical safety buffer.
