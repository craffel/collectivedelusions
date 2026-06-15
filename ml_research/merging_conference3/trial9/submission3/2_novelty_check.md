# 2. Novelty and Originality Check

This section evaluates the originality, uniqueness, and intellectual novelty of the proposed **Contraction-Regularized Router (CR-Router)** in comparison to existing literature in sequential deep model serving and model ensembling.

## Core Original Contributions

1. **Dynamical Systems Lens on Sequential Ensembling:**
   - Framing sequential layer-by-layer dynamic ensembling as a discrete-time dynamical system on a Banach space is a highly original and elegant conceptual shift. While previous work (like ChemMerge) acknowledged trajectory oscillations, it approached them via empirical approximations inspired by chemical kinetics. This paper is the first to formalize feedforward representation-routing propagation as an iterative operator and analyze its properties using Banach's Fixed-Point Theorem.

2. **Derivation of Joint representation-routing Lipschitz Bounds:**
   - The derivation of Lipschitz bounds for the joint mapping (Theorems 3.1 & 3.2) represents an original mathematical contribution. By modeling the nonlinear feedback loop where the routing gating coefficients directly depend on the representation, and the representation update is blended via these coefficients, the authors capture the coupled dynamics rather than analyzing routing or representation drift in isolation.

3. **Co-Designed Spectral-Temperature Objective:**
   - The resulting objective function is a cohesive, theoretically motivated regularization framework. The authors demonstrate that standard L2 weight decay is mathematically insufficient for stability because temperature parameters can collapse to zero, driving the Softmax to behave as a discontinuous step function with infinite Lipschitz constant. Penalizing both the Frobenius norm (as a differentiable upper bound on the spectral norm) and the inverse routing temperature ($1/\tau_l^2$) is a novel and rigorous co-design.

4. **Decoupled Test-Time Annealing:**
   - Decoupling training-time contraction stability from inference-time sharpness via **Adaptive Test-Time Temperature Annealing** ($\gamma_{\text{scale}} \le 1.0$) is a highly creative and practical solution to the "expert dilution" trade-off. This post-hoc modification allows smooth optimization during calibration while ensuring highly focused, sharp gating selections during test-time.

5. **Centroid-Based Routing Warm-Starting:**
   - Proposing to initialize the routing weights $W_{\text{route}}^{(l)}$ using task centroids extracted from the extremely scarce calibration split provides a powerful geometric prior. This guides optimization directly into stable, task-aligned attraction basins, mitigating seed sensitivity under representational overlap.

## Comparison with Related Literature

- **SABLE \cite{sable2025samplewise}:**
  - SABLE is a non-parametric approach that uses stateless nearest-centroid projections. While highly accurate, SABLE incurs significant memory and distance-computation overhead. CR-Router is a parametric learned alternative. By introducing a concrete serving latency benchmark (Table 9), the authors prove that CR-Router achieves a **1.51x throughput speedup** and **33.7% latency reduction** over SABLE, validating the practical motivation of parametric learned routers.
  
- **ChemMerge \cite{chemmerge2025kinetics}:**
  - ChemMerge utilizes empirical chemical kinetics equations to smooth routing trajectories, but lacks formal stability guarantees, Lipschitz bounds, or parameter constraints. CR-Router replaces these heuristics with a mathematically rigorous foundation, proving that stability can be enforced directly through joint regularization.

- **Standard L2 Regularization (L2-Fixed Router Baseline):**
  - The paper designs a strong competitor, the L2-Fixed Router, which applies standard weight decay but keeps the temperature fixed. Under depth-heterogeneous settings (Experiments 1 & 2), the L2-Fixed Router's fixed temperature heuristic collapses, whereas CR-Router's joint spectral-temperature contraction regularization achieves superior performance, outperforming L2-Fixed by **+14.37%** in Experiment 1 and **+8.25%** in Experiment 2.

## Critical Remarks on Originality
- **Adoption of Sandbox and SEP:** The Subspace Energy Projection (SEP) and the Sandbox coordinate environment are adopted from prior work (\cite{sable2025samplewise}, \cite{subspaceenergy2024sep}). However, this is standard practice in machine learning theory papers, and the primary originality lies in the mathematical formulation of contraction mapping on ensembling trajectories and the resulting joint regularizer. This represents a substantial and legitimate scientific contribution.
