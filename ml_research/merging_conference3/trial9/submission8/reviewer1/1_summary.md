# 1. Summary of the Paper

## Main Topic and Approach
The paper introduces **GraviMerge**, a dynamic, test-time model ensembling framework designed for resource-constrained multi-task edge serving of parameter-efficient expert adapters (like LoRA). At its core, the paper aims to resolve the "accuracy-stability dilemma," where dynamic ensembling methods experience large layer-to-layer routing weight jitter, disrupting the coherence of activation propagation. 

To address this, GraviMerge models deep representation routing as a continuous, second-order physical dynamical system inspired by classical orbital mechanics and Newtonian gravitation. In this framework:
1. **Representational Manifold:** The representation space is modeled on a unit hypersphere ($\mathbb{S}^{D-1}$).
2. **Spacecraft Probe (Coordinates):** A stateful virtual spacecraft coordinate probe tracks the routing state at layer step $l$.
3. **Stellar Attractors (Centroids):** Pre-trained expert centroids act as high-mass stationary celestial bodies exerting gravitational pull on the spacecraft.
4. **Arrhenius Mass Activation (AMA):** Early-layer activations are used to compute dynamic expert masses at test-time via an exponential Arrhenius/Gibbs factor.
5. **Geodesic Trajectory Integration (GTI):** The spacecraft's motion is governed by Newtonian second-order mechanics, including momentum, acceleration, and viscous medium drag. The dynamics are projected onto local tangent spaces, integrated via the spherical Exponential Map, and the velocity vector is parallel transported across layer steps to preserve geometric invariants.
6. **Gravitational Influence Blending (GIB):** The final ensembling weights $\alpha_k^{(l)}$ are calculated dynamically from the relative magnitudes of the softened gravitational force vectors pulling the spacecraft towards each expert.

The paper argues that by introducing physical mass, velocity, acceleration, and viscous drag, the routing trajectory naturally filters out high-frequency noise and stabilizes representation propagation.

## Key Findings
- On a **Projected Digit Representation Space (RDS) Proxy** benchmark, GraviMerge achieves a joint serving accuracy of **88.69%**, while reducing layer-to-layer ensembling weight jitter (measured as Mean Absolute Deviation of ensembling coefficients) to **0.00190**.
- This is a reported **6.01$\times$** reduction in jitter compared to ChemMerge (a first-order chemical ODE-based ensembling method), a **5.47$\times$** reduction compared to first-order Exponential Moving Average (EMA) smoothing, and a **2.40$\times$** reduction compared to SABLE (a stateless ensembling method).
- The authors claim that GraviMerge avoids "Heterogeneity Collapse" (accuracy drop on mixed-domain batches) and "Vectorization Collapse" (accuracy drop under batch size $B=1$) across homogeneous, heterogeneous, and vectorized serving workloads.
- Scalability analyses in the appendix suggest that the method scales to deeper Transformer dimensions (using a simulated GPT-2 Base dimension, $D=768$) and remains robust under simulated representational noise.

## Explicitly Claimed Contributions
1. **Physics-Informed Routing Paradigm:** A dynamic ensembling framework that maps deep activation trajectories to multi-body gravitational physics on spherical manifolds, adding interpretable second-order physical inertia.
2. **Three Novel Mechanisms:** 
   - *Arrhenius Mass Activation (AMA)* for early, dynamic mass assignment.
   - *Geodesic Trajectory Integration (GTI)* incorporating force, velocity projection, exact geodesic steps, and parallel transport.
   - *Gravitational Influence Blending (GIB)* translating force magnitudes into continuous, differentiable ensembling weights.
3. **Rigorous Evaluation on Stability:** Comprehensive testing over 10 seeds showing significant jitter reduction while maintaining or slightly improving accuracy compared to state-of-the-art baselines.
4. **Resilience to Edge-Serving Workloads:** Proof of performance across homogeneous and heterogeneous batching schemes (at $B=256$ and $B=1$).
