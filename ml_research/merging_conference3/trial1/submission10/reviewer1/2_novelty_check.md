# 2. Novelty Check

## Characterization of Novelty
The paper presents an **exploratory, concept-driven** novelty. It represents one of the first attempts to move beyond the flat, linear Euclidean paradigm of model merging into a continuous, learned, non-linear parameter-space coordinate warping framework. 

Rather than relying on linear convex combinations or rigid manifold projections, it introduces a learned weight-space coordinate warping via differentiable normalizing flows (specifically, RealNVP affine coupling layers). 

## Detailed "Delta" from Prior Work
The "delta" from existing literature can be characterized across three main dimensions:

1. **Delta from Linear Merging (Task Arithmetic, Ties-Merging, AdaMerging):**
   - *Prior Work:* Linear merging averages parameters directly in the flat, rigid Euclidean weight-space, which often forces the merged model across high-loss barriers.
   - *FoldMerge:* It warps the weight coordinate system itself so that linear combinations in the latent "Origami Space" map back to non-linear paths in the original weight-space, avoiding high-loss barriers.

2. **Delta from Geometric/Manifold Alignment (Git Re-Basin, ZipIt!, OrthoMerge):**
   - *Prior Work:* Git Re-Basin and ZipIt! utilize discrete permutation matrices to align models into shared basins. OrthoMerge uses rigid, predefined Lie algebra projections.
   - *FoldMerge:* It replaces discrete permutations and rigid, static manifold projections with **continuous, learned, and data-driven coordinate transformations** optimized dynamically on downstream test-time data streams.

3. **Delta from Test-Time Adaptive Merging (SyMerge, AdaMerging):**
   - *Prior Work:* SyMerge is a state-of-the-art test-time adaptation protocol, but its adaptive capability is fundamentally restricted to linear scaling of the classifier heads and parameter scaling.
   - *FoldMerge:* It adopts SyMerge's unsupervised self-labeling protocol but replaces linear parameter scaling with a highly expressive 4-layer normalizing flow network, yielding significantly greater representational capacity.

## Theoretical Critique of the Novelty (Theorist's Perspective)
While the paper is conceptually elegant, its theoretical novelty is somewhat limited when scrutinized through a rigorous mathematical lens:

- **Heuristic Application of Normalizing Flows:** The primary tool used for the coordinate warp is RealNVP, a standard architecture developed for density estimation in generative modeling. The paper uses it purely as a black-box invertible neural network (INN) for function mapping, without any novel architectural modification.
- **Diffeomorphism by Construction, Not Proof:** The claim of using a "diffeomorphism" is true by construction (RealNVP layers are bijective and differentiable by design), but there is no formal proof or mathematical guarantee that this specific class of diffeomorphisms can successfully align arbitrary neural loss basins or preserve the functional properties of neural representations.
- **Lack of Geometric Proofs:** The paper lacks any formal theorems or proofs. For example, there is no theoretical proof showing that Origami Space actually exists as a stable manifold, nor is there a proof that the non-linear path mapped by $g_\phi^{-1}$ has lower loss than a straight-line path under standard assumptions.
- **Dressed-up Heuristics:** The mathematical terminology ("diffeomorphism", "volume-preserving", "differential topology", "manifold") serves as high-level motivation, but the actual implementation relies on standard feed-forward MLPs within RealNVP coupling layers operating on flattened row vectors. Thus, the actual novelty is a highly speculative, empirical heuristic disguised in geometric language rather than a rigorous theoretical breakthrough.
- **Negligible Practical Utility over Baselines:** Empirically, the complex non-linear coordinate warping performs virtually identically ($+0.02\%$) to a simple, highly optimized linear baseline (SyMerge). In the frozen classifier head ablation, the performance is also identical. This suggests that the expensive 2.6M parameter normalizing flow network does not provide any measurable theoretical or practical delta over basic linear scaling, casting doubt on the true value of the proposed non-linear formulation.
