# 1. Summary of the Paper

## Topic and Context
The paper addresses the **accuracy-stability dilemma** (or **routing jitter paradox**) in Mixture-of-Experts (MoE) and dynamic adapter-merging systems when serving sequential, heterogeneous query streams on edge devices. 
- **Stateless routers** (e.g., SABLE, SPS-ZCA) compute ensembling weights for each layer independently. While they adapt instantly to sequential task switches, noisy activations cause the routing weights to oscillate violently across adjacent layers (high-frequency spatial jitter), triggering representation drift and cascading representation collapse.
- **Stateful routers** (e.g., ChemMerge, PAC-Kinetics) use temporal low-pass filters (continuous-time chemical kinetics or differential equations) across sequence samples to smooth spatial jitter. However, carrying an internal historical state across sequential samples introduces temporal inertia (representational hysteresis), causing severe accuracy drops immediately following rapid task transitions.

## Proposed Approach: QPathMerge
The authors propose **Markovian Path-Integral Ensembling (QPathMerge)**, a training-free serving controller designed to decouple spatial smoothing (across network layers) from temporal sample tracking (across the query stream). 
- **Physical Metaphor:** The sequence of network layers is modeled as a discrete 1D lattice, and the routing trajectory is represented as a discrete Euclidean path integral over network depth.
- **Probabilistic Graphical Model (PGM):** This formulation is mapped to a 1D chain-structured Markov Random Field (MRF) or Potts-like model. The path cost contains matching potentials (clamped cosine similarities between activations and expert centroids) and transition barriers (penalizing expert switches between adjacent layers).
- **Exact Sum-Product Belief Propagation:** Rather than continuous-time differential equations, the exact Forward-Backward sum-product algorithm (Belief Propagation) is executed to calculate the exact marginal probability of expert selection at each layer in $O(L K^2)$ time.
- **Key Deployment Candidate (QPathMerge-Single):** To bypass the computational overhead of a trial forward pass, the authors introduce a single-pass, on-the-fly variant. It propagates forward messages using actual representations and speculatively assumes constant future potentials to recursively compute backward messages over a *Truncated Backward Horizon* ($H \ll L$, default $H=4$).
- **Extrapolation Relaxations:** To break the power-iteration degeneracy of constant future potentials, two variants are introduced: `QPathMerge-LinearExtrap` (linear slope projection of potentials) and `QPathMerge-RollingExtrap` (rolling average of past potentials).

## Key Claims and Findings
1. **Spatio-Temporal Decoupling:** By performing message passing entirely within the depth lattice of a single forward pass, QPathMerge is completely stateless across sequence samples (eliminating temporal serving lag and hysteresis) while acting as a mathematically optimal spatial low-pass filter across network depth.
2. **Jitter Reduction:** In a 14-layer Analytical Coordinate Sandbox, QPathMerge slashes spatial layer-wise routing jitter by over $3\times$ compared to SABLE and ChemMerge, while achieving a leading 98.50% serving accuracy under heterogeneous streams.
3. **Linear Complexity via Truncation:** Utilizing Dobrushin's contraction theorem, the authors show that backward messages converge exponentially fast. A truncated horizon of $H = 4$ is shown to be sufficient to sustain near-oracle spatial smoothness while restoring linear $O(L H K^2)$ complexity.
4. **Physical Validation:** The framework is physically validated on a pre-trained ResNet-18 model using a natural ImageNet-1K stream of 40 classes (10 classes per task, 200 query samples), where QPathMerge reduces Layer Jitter by over $3.2\times$ while preserving classification accuracy.
