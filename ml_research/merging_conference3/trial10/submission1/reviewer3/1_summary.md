# 1. Summary of the Paper

## Main Topic
The paper addresses the **accuracy-stability dilemma** in dynamic Mixture-of-Experts (MoE) and adapter-merging systems serving sequential, heterogeneous query streams on edge devices. In this setting, serving controllers face a trade-off:
- **Stateless routers** (e.g., SABLE, SPS-ZCA) adapt instantly to task switches but suffer from **high-frequency spatial (layer-to-layer) oscillations** of ensembling weights, leading to representation drift and collapse.
- **Stateful routers** (e.g., ChemMerge, PAC-Kinetics) smooth out spatial jitter by applying a temporal low-pass filter across sequential samples, but introduce severe **inertial lag (hysteresis)** during task transitions, collapsing downstream accuracy.

## Proposed Approach: QPathMerge
The authors propose **Markovian Path-Integral Ensembling (QPathMerge)**, a training-free serving controller that decouples spatial smoothing across network depth from temporal tracking across sequential samples. 
- **Lattice Formulation:** The sequence of $L$ network layers is modeled as a discrete 1D lattice, and the routing trajectory is represented as a discrete Euclidean path integral over network depth.
- **Probabilistic Graphical Model:** The formulation is mapped to a 1D chain-structured Markov Random Field (MRF), where node potentials capture local expert-representation matching, and edge potentials enforce a transition barrier ($\gamma$) to penalize expert switches between adjacent layers.
- **Exact Solver:** The exact globally optimized marginal probabilities of expert selection are calculated in $O(L K^2)$ time using the scale-normalized Forward-Backward sum-product algorithm (Belief Propagation).
- **Single-Pass Variant (QPathMerge-Single):** To bypass the trial forward pass of the exact two-pass bidirectional solver, the authors introduce a recursive, on-the-fly single-pass variant. It speculatively assumes constant future potentials and runs a backward recurrence over a Truncated Backward Horizon ($H=4$), which scales linearly as $O(L H K^2)$.
- **Extrapolation Relaxations:** To break the power-iteration degeneracy of constant speculative potentials, the authors propose Linear Extrapolation (`LinearExtrap`) and Rolling Extrapolation (`RollingExtrap`) of potentials.

## Key Findings and Claims
1. **Resolution of the Trade-off:** QPathMerge achieves near-oracle spatial trajectory smoothness (slashing layer-wise routing jitter by over $3\times$ to $5\times$ on synthetic and physical manifolds) while maintaining absolute statelessness across samples (eliminating temporal serving lag and hysteresis).
2. **Superiority over Basic Filtering:** Basic post-hoc signal processing (causal EMA or symmetric Gaussian smoothing across depth) is shown to be insufficient, reducing jitter by only 1% to 20% compared to QPathMerge's $3.65\times$ reduction.
3. **Truncation Effectiveness:** A tiny truncated backward horizon $H=4$ yields results that are statistically indistinguishable from the full-depth bidirectional recurrence, verifying the contraction mapping property of the transition leakage operator.
4. **Physical Validation:** The framework is physically validated on a ResNet-18 model using natural ImageNet-1K streams, demonstrating its generalizability to high-dimensional representation manifolds and its negligible serving-time overhead.

## Explicit Claimed Contributions
1. **Mathematical Framework:** The first framework to formulate deep network ensembling as a path integral over network depth, solved exactly via sum-product message passing on a 1D MRF.
2. **The QPathMerge-Single Controller:** A highly efficient, single-pass on-the-fly deployment candidate utilizing a Truncated Backward Horizon.
3. **Theoretical Analysis:** Formal proofs of symmetric cancellation of forward-backward drift at $M \to 0$, Dobrushin contraction mapping for the truncated horizon, and power-iteration degeneracy analysis under the Perron-Frobenius theorem.
4. **Empirical Evaluation:** Extensive evaluations inside a high-fidelity 14-layer Coordinate Sandbox and a physical ResNet-18 ImageNet-1K setup, demonstrating a Pareto-optimal frontier balancing accuracy and stability.
