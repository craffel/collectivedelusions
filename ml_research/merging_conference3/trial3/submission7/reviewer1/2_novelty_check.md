# 2. Novelty and Literature Check

## Key Novel Aspects of the Work
1. **Unified Structural Hierarchy (GranMerge Taxonomic Continuum):** Unlike prior work that operates at a single fixed granularity, this paper introduces a unified taxonomic continuum of five nested parameter resolution levels (Global, Layer-wise, Block-wise, Component-wise, and Tensor-wise) for multi-task adaptive model merging.
2. **First Systematic Study of the Generalization-Granularity Trade-off:** The paper explicitly identifies and analyzes the conflict between optimization capacity (granularity) and generalization (transductive overfitting) on small calibration streams ($N=256$) at test-time.
3. **First-order vs. Zero-order Overfitting Dynamics:** The paper provides a deep, rigorous comparative analysis of optimization trajectories under first-order (gradient descent via Adam) and zero-order (derivative-free via 1+1 ES) methods in high-dimensional merging spaces.
4. **Deconstruction of ES Robustness (Sluggishness Hypothesis):** The paper provides a highly candid and intellectually honest explanation of why zero-order ES appears to perform better at high granularities: it's not because ES is a superior optimizer, but because it is too sluggish to optimize away from the high-quality initialization in high-dimensional spaces (the "Curse of Dimensionality").
5. **Honest Evaluation of Test-Time Adaptation Failure:** The paper is novel in its willingness to highlight that *none* of the test-time adaptive configurations beat the simple static uniform baseline, and to diagnose *why* (surrogate prediction entropy misalignment).

## The 'Delta' from Prior Work
- **From Task Arithmetic (Ilharco et al., 2022):** Task Arithmetic uses a single manual or global scale per task (Level 1). GranMerge expands this to five layers of granularity and automates it via test-time optimization.
- **From AdaMerging (Yang et al., 2023):** AdaMerging introduced layer-wise (Level 2) adaptive coefficient optimization. GranMerge extends this up to Level 5 (Tensor-wise) and down to Level 3/4, providing the missing intermediate and high-resolution picture.
- **From SplineMerge / PolyMerge (Jung et al., 2025):** These methods use low-degree splines/polynomials to restrict layer-wise coefficients. GranMerge provides the empirical justification for why such restrictions are necessary by showing what happens when updates are unconstrained at finer granularities.
- **From RegCalMerge (Jin et al., 2026):** RegCalMerge focuses on class bias and spatial regularization. GranMerge focuses specifically on parameter structural resolution boundaries and optimizer-driven overfitting trajectories.

## Characterization of Novelty
The novelty of this paper is **significant, empirical, and diagnostic** rather than purely algorithmic:
- It does not introduce a major new algorithm; instead, it uses existing building blocks (Task Arithmetic, AdaMerging's entropy minimization, 1+1 ES, basic L2 penalties).
- Its high value lies in **demystifying and mapping the structural limits of adaptive model merging**. For practitioners, this "honest analysis of negative results" is a major contribution, as it stops them from implementing complex, high-latency test-time adaptation schemes when a simple static average is superior and has zero overhead.
- The conceptual framework and taxonomy are elegant, and the deconstruction of optimizer behaviors is highly rigorous.
