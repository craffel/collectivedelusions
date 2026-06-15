# 1. Summary of the Paper

## Overview
This paper introduces **Lyapunov-Stable Active Representation Coupling (L-ARC)**, a training-free closed-loop control system designed to stabilize and optimize dynamic model ensembling for parameter-efficient adapters (e.g., LoRAs) served on resource-constrained edge devices. 

In dynamic ensembling, stateful continuous-depth methods track routing weights as concentrations governed by ordinary differential equations (ODEs). While this smooths trajectories, it often relies on heuristic constant feedback step sizes ($\eta$) to warp intermediate representations toward static early-stage centroids. Under mixed workloads, this causes a "representational backward-shift"—pulling highly refined, late-layer activations backward toward noisy early-stage coordinates and degrading performance.

To resolve these limitations, L-ARC introduces:
1. **A Lyapunov Feedback Controller**: Derived by modeling representation similarity error as a system-level candidate Lyapunov function, providing formal control-theoretic guarantees for continuous-depth feature propagation.
2. **An Analytical Dissipation Guard**: A closed-loop controller that regulates the feedback rate ($\eta$) on-the-fly, ensuring that representation warping is strictly error-decreasing (dissipative) and gating it off when updates are non-dissipative.
3. **Entropy-Triggered Gating (ET-L-ARC)**: A control-theoretic optimization that dynamically evaluates the Dissipation Guard only under routing uncertainty, reducing the relative routing latency overhead to just $99.85\%$ (adding only $0.06$ ms per sample) under clean workloads.
4. **Entropy-Gated Concentration Gating (ECG-Reset)**: An online state-space shield that monitors routing entropy to dynamically freeze continuous ODE kinetics during routing dropouts or failures (Setting C), preventing memory corruption.
5. **Representation-Agreement State Correction (RASC)**: A dual-loop control mechanism that leverages representation-space coordinate tracking to dynamically correct and override biased feedforward routing confidence under systematic router bias (Setting D).
6. **Extremes of Scalability and Deep Serving**:
   - **Mid-Network Recalibration (MNR)**: Solves representation-space drift in very deep networks (e.g., 32-to-72 layers) by partitioning the network into coordinate anchor zones with intermediate centroids.
   - **Hierarchical or Top-$p$ Expert Subsetting (H-RASC)**: Collapses similarity search complexity from $O(K \cdot D)$ to $O(p \cdot D)$ per layer under dense adapter serving scales, eliminating the scalability bottleneck.

## Key Contributions
- **Continuous-Time Stateful Serving (ECG-Kinetics)**: Proposes ECG-Reset to shield continuous-depth ensembling from propagating memory faults under transient failures.
- **Control-Theoretic Feedback Formulation**: Establishes a formal Lyapunov stability framework analyzing representation error across layer depth, presenting active representation feedback as a theoretically elegant boundary case.
- **Analytical Dissipation Guard**: Derives a closed-loop controller that adaptively modulates feedback step sizes to ensure representation warping is strictly dissipative.
- **Representation-Agreement State Correction (RASC)**: Resolves state-locking failures under persistent, confident router corruption by using representation-space coordinate tracking to override biased routing outputs.
- **Rigorous Evaluation in the Coordinate Sandbox**: Benchmarks L-ARC under Static Centroids (Setting A), Transient Failures (Setting C), and Confident Router Bias (Setting D) in a 14-layer Coordinate Sandbox, reporting statistical significance (paired t-tests), latency profiling, and gating rates.
- **Real-World Feasibility Verification**: Conducts a small-scale pilot study on LLaMA-3-8B with 3 parallel LoRAs on a single GPU, validating high-dimensional manifold orthogonality and Dissipation Guard stability on realistic hidden states.
