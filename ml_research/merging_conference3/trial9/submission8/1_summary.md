# 1. Summary of GraviMerge

## Paper Overview and Context
The paper presents **GraviMerge**, a physics-informed, stateful test-time model ensembling framework designed to resolve the fundamental "accuracy-stability dilemma" in dynamic parameter-efficient adapter (specifically Low-Rank Adaptation, or LoRA) ensembling. 

In dynamic model merging systems (such as SABLE), ensembling weights are computed at each network layer based on the similarity between intermediate network activations and pre-extracted task centroids. However, because representations naturally fluctuate across sequential network layers, stateless routing leads to severe **layer-to-layer ensembling weight jitter**. This weight jitter means that the model's blended parameters oscillate rapidly through weight space across sequential depth, disrupting the smooth propagation of activations and causing representational incoherence. Prior state-dependent systems (such as ChemMerge) attempted to smooth updates using first-order non-equilibrium chemical reaction kinetics, but these linear, first-order systems lack physical inertia and momentum. Under competitive reaction rates (e.g., low temperature regimes), they are highly prone to volatile concentration oscillations and overreactions, amplifying routing weight jitter rather than resolving it.

To resolve this, GraviMerge models deep activation trajectories as a multi-body physical system governed by second-order Newtonian gravitational dynamics on a spherical manifold (the unit hypersphere $\mathbb{S}^{D-1}$):
1. **Auxiliary State Space:** The intermediate activation routing is decoupled from the main network propagation by mapping routing coordinates to a virtual, stateful "spacecraft probe" traveling across the unit hypersphere.
2. **Task Attractors:** Pre-trained expert centroids are mapped to stationary celestial bodies (stars or massive planets) with dynamic masses.
3. **Physical Integration:** The spacecraft probe's trajectory is updated under the influence of softened gravitational forces, velocity, and viscous drag, which are integrated via rigorous spherical geodesic steps (using the exponential map) and parallel transport of the velocity state vector.
4. **Ensembling Weights:** The ensembling weights for blending the expert adapters are derived directly from the relative gravitational forces exerted on the probe.

---

## Core Contributions
1. **Physics-Informed Routing Paradigm:** A novel, second-order Newtonian multi-body gravitational framework mapping deep learning activation trajectories to orbital dynamics. This provides an elegant, physically intuitive, and interpretable smoothing mechanism.
2. **Key Geometric & Physical Mechanisms:**
   - **Arrhenius Mass Activation (AMA):** Dynamically allocates task masses at test-time using an exponential alignment factor.
   - **Geodesic Trajectory Integration (GTI):** Projecting forces, velocity, and viscous drag onto the local tangent space of the sphere, updating positions via the exact spherical exponential map and parallel transporting velocity vectors across steps.
   - **Gravitational Influence Blending (GIB):** Translating localized physical gravitational attraction forces into continuous ensembling weights.
3. **Closed-Loop Feedback and Temporal Carryover:**
   - **Coupled GraviMerge:** Incorporates a closed-loop correction force pulling the spacecraft probe toward live intermediate activations.
   - **Temporal State Carryover:** Carries velocity states across sequentially arriving queries to provide inter-query momentum.
4. **Rigorous Empirical Evaluation:**
   - Evaluated on a **Real-World Digit Representation Space (RDS)** benchmark built from the scikit-learn digits dataset projected into a 192-dimensional latent space.
   - Demonstrates highest joint serving accuracy (**88.69%**) while reducing layer-to-layer ensembling weight jitter by **6.01$\times$** compared to ChemMerge, **5.47$\times$** compared to EMA, and **2.40$\times$** compared to SABLE.
   - Proves complete resilience to "Heterogeneity Collapse" and "Vectorization Collapse" (maintaining optimal performance at both $B=256$ and $B=1$).
   - Features comprehensive scaling and robustness studies in the appendix, including deep Transformer verification ($D=768$), representational noise studies, out-of-distribution (OOD) safeguard via **Sentinel Attractor Dynamics**, and adaptive viscous drag scheduling.
