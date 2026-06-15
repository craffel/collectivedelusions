# 2. Novelty and Contextualization Check

## Relationship to Prior Work

### 1. Stateless Dynamic Ensembling (SPS-ZCA, SABLE)
Stateless approaches such as **SPS-ZCA** and **SABLE** compute ensembling weights independently at each layer using local activation similarities to pre-computed centroids. While they are highly responsive, they suffer from high-frequency layer-wise weight fluctuations ("routing jitter") and lack spatial continuity across depth. 
*   **L-ARC's Departure**: L-ARC builds upon the stateful, continuous-depth ODE tracking paradigm to act as a natural low-pass filter that smooths routing trajectories. Unlike stateless methods, it enforces depth-wise memory and spatial consistency.

### 2. Open-Loop Stateful Ensembling (ChemMerge)
**ChemMerge** introduced discretized reaction-decay ODEs to model expert concentrations across layers, which resolved routing jitter. However, it relies on a heuristic, constant feedback step size ($\eta$) to warp activations toward early-stage centroids. Under mixed workloads, this open-loop constant feedback causes "representational backward-shift", pulling highly refined late-layer representations backward toward noisy early-stage coordinates and monotonically degrading performance.
*   **L-ARC's Departure**: L-ARC replaces this open-loop heuristic with a closed-loop controller. By modeling representation error as a candidate Lyapunov function, it analytically derives a local **Dissipation Guard** that adaptively scales or gates representation warping on-the-fly, guaranteeing strictly dissipative (error-decreasing) updates.

## Novel Technical Components of L-ARC

1.  **Closed-Loop Lyapunov Controller & Dissipation Guard**: This is a major conceptual leap. Instead of treating representation warping as an empirical heuristic, L-ARC treats it as a non-linear control problem. By deriving the dissipation coefficient $A^{(l)}$ on-the-fly, the controller mathematically guarantees that feedback warping is strictly dissipative.
2.  **Entropy-Triggered Lyapunov Gating (ET-L-ARC)**: Rather than executing computationally heavy matrix/vector multiplications at every layer, ET-L-ARC leverages routing uncertainty (Shannon entropy) to dynamically gate the Dissipation Guard. Under highly confident or complete failure regimes, it bypasses the guard entirely, reducing absolute overhead to just $0.06$ ms per sample.
3.  **Entropy-Gated Concentration Gating (ECG-Reset)**: An online state-space shield that freezes ODE kinetics under high entropy. While prior works allow transient routing failures to corrupt the stateful memory across subsequent layers, ECG-Reset acts as a sample-and-hold circuit, preserving ensembling weights.
4.  **Representation-Agreement State Correction (RASC)**: Resolves the "state-locking failure" under systematic router bias (where a confident but incorrect router corrupts stateful kinetics). RASC's dual-loop feedback compares feedforward selections with physical representation similarity, dynamically overriding corrupted feedforward inputs.
5.  **Scaling and Adaptation to Deep/Dense Pipelines (MNR & H-RASC)**:
    - **Mid-Network Recalibration (MNR)** represents a novel method to counter representation-space coordinate drift across massive depths by establishing localized anchor zones.
    - **Hierarchical or Top-$p$ Expert Subsetting (H-RASC)** resolves the $O(K)$ computational complexity of coordinate similarity checks, permitting the closed-loop system to scale to hundreds of concurrent adapters.

## Synthesis on Originality
The originality of L-ARC is **excellent**. It elegantly bridges classical control theory (Lyapunov stability, closed-loop feedback, state-shielding, dual-loop correction) with deep neural network representations. The mathematical proofs are rigorous and represent a highly novel, principled departure from the usual ad-hoc deep learning heuristics.
