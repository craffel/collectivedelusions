# 2. Novelty Check

## Characterization of Novelty
The novelty of this work is **highly significant**. Rather than presenting another empirical heuristic or trial-and-error routing scheme, the paper provides a complete paradigm shift by mapping the network's depth dimension to a 1D lattice in statistical mechanics. It reframes the layer-wise model ensembling problem as a global path-optimization problem over a Markov Random Field (MRF) and solves it exactly using classical probabilistic graphical model (PGM) theory. 

This theoretical framing is mathematically elegant and represents a deep departure from existing heuristic controllers.

## Key Novel Aspects and the "Delta" from Prior Work

### 1. Decoupling of Spatial and Temporal Domains
- **Prior Work:** Existing models resolve the layer-wise routing jitter paradox by applying a temporal low-pass filter (e.g., biochemistry kinetics in ChemMerge, ODEs in PAC-Kinetics, or EMAs in Momentum-Merge) across sequential samples. This couples the spatial smoothness of a single forward pass with the temporal history of the stream, introducing severe serving lag and hysteresis during sudden task switches.
- **QPathMerge Delta:** QPathMerge is the first serving controller to decouple spatial layer smoothing from temporal tracking. By executing symmetric belief propagation entirely within the depth lattice of a single forward pass, it achieves near-oracle layer-wise trajectory smoothness while maintaining absolute statelessness across sequence samples.

### 2. The Predict-then-Smooth Two-Pass Pipeline
- **Prior Work:** Standard stateless routers (SABLE, SPS-ZCA) compute ensembling weights at each layer independently, leading to spatial routing oscillations.
- **QPathMerge Delta:** QPathMerge introduces a bidirectional message-passing architecture over network depth. In the exact model, a rapid trial pass extracts local node potentials, which are then globally optimized via a bidirectional Forward-Backward sum-product sweep before the final low-jitter forward pass is executed.

### 3. Recursive On-The-Fly Speculative Backward Pass (QPathMerge-Single)
- **Prior Work:** Global PGM message-passing typically requires multiple passes over the graph, which would double parameter reads and FLOPs during serving.
- **QPathMerge Delta:** To enable single-pass execution, the paper proposes a novel speculative backward pass. By assuming constant future potentials (or projecting linear/rolling trends), the backward messages are computed recursively on-the-fly over a Truncated Backward Horizon of length $H$.

### 4. Rigorous Theoretical Foundations and Analysis
- **Prior Work:** Most routing papers are heavily empirical, with minimal theoretical justification.
- **QPathMerge Delta:** The paper provides a dense, multi-faceted theoretical analysis that validates its design:
  - **Dobrushin Contraction Theorem:** Formally proving that the scale-normalized transition leakage operator acts as a contraction mapping on the probability simplex, guaranteeing exponential convergence of truncated backward messages and justifying the small truncated horizon ($H=4$).
  - **Perron-Frobenius Power Iteration:** Revealing that the speculative constant future potential assumption mathematically reduces the backward pass to a power iteration that converges to a dominant eigenvector.
  - **Symmetric Cancellation of Drift:** Formally proving that at the absolute identity coupling limit ($M \to 0$), the exponential sharpening of the forward and backward passes perfectly cancel out, yielding exactly zero trajectory jitter across all layers.
  - **Tree-Structured Extension:** Mathematically proving that the message-passing framework generalizes to tree-structured network architectures.
