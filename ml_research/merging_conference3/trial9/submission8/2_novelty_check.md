# 2. Novelty and Positioning Check

## Positioning Relative to Prior Work

### 1. Versus Stateless Dynamic Merging (e.g., SABLE)
- **SABLE** computes ensembling weights at every layer independently using the cosine similarity between the intermediate activation vector and task-specific centroids. Since the representations of deep neural networks naturally fluctuate from layer to layer, SABLE is highly susceptible to severe layer-to-layer ensembling weight jitter (rapid oscillations).
- **GraviMerge** replaces SABLE’s stateless mapping with a stateful trajectory tracking controller. By mapping the activations to a virtual coordinate probe (spacecraft) and task experts to celestial bodies, GraviMerge uses physical inertia to naturally smooth out layer-wise weight variations.

### 2. Versus State-Dependent First-Order Kinetic Merging (e.g., ChemMerge)
- **ChemMerge** modeled the ensembling weights using first-order non-equilibrium chemical reaction kinetics.
- The authors argue that first-order systems lack physical inertia and momentum. Under competitive reaction rates, first-order chemical ODE systems lag significantly behind representational changes or undergo volatile concentration oscillations.
- **GraviMerge** introduces a **second-order** physical dynamical system. The introduction of virtual mass, velocity, acceleration, and viscous medium drag provides the exact second-order inertia needed to break the lag-accuracy bottleneck.

### 3. Versus Traditional Physics-Informed Neural Networks / Neural ODEs
- Traditional Neural ODEs formulation models the depth of residual connections as first-order gradient flows or continuous concentration dynamics.
- **GraviMerge** differs fundamentally because its second-order physical dynamics are engineered specifically to govern the **weight-space blending coefficients** across layers. It decouples the routing state (virtual spacecraft) from the main neural activations to preserve pre-trained scale expectations.

---

## Technical and Conceptual Novelty Assessment

The paper exhibits an exceptionally high degree of technical and conceptual novelty. It is not merely a shallow physics metaphor; the authors have built a mathematically rigorous and structurally complete framework:

1. **Second-Order Inertial Formulation:** By introducing a virtual mass ($m = 1.0$), velocity, and frictional drag, the routing controller possesses physical momentum, acting as an active low-pass filter (decaying at $-40$ dB/decade) to suppress high-frequency noise.
2. **Manifold-Consistent Spherical Differential Geometry:** Recognizing that representations are bounded to a unit hypersphere, the authors implement:
   - Tangent plane projections for acceleration and velocity.
   - The exact spherical **Exponential Map (Geodesic Step)** to update coordinates on $\mathbb{S}^{D-1}$.
   - Exact closed-form **Parallel Transport** of velocity vectors from one tangent space to another to avoid numerical energy drift.
3. **Closed-Loop Feedback (Coupled GraviMerge):** Resolves the critique that the spacecraft is completely "decoupled" from the actual representation path by adding a virtual gravitational correction force pointing toward the live activation vector.
4. **Temporal State Carryover:** Resolves the statelessness of query-by-query inference in continuous streaming servers by propagating the terminal velocity of one query to initialize the velocity of the next.
5. **Sentinel Attractor Dynamics (SAD):** Integrates an out-of-distribution confidence-gating function that collapses task masses under OOD inputs, naturally trapping the coordinate spacecraft at the sphere's geometric barycenter (producing a safe, uniform blend).
6. **Adaptive Viscous Drag Scheduling:** Adjusts the drag coefficient based on proximity to expert attractors to speed up transitions and suppress local overshoots.
7. **Adaptive Gravitational Scheduling (AGS):** Formulates self-calibrating gravitational constants to restrict orbital energy and guarantee stable trajectories under any scale.

These extensive mathematical and system-level additions demonstrate that GraviMerge is a highly original, exceptionally comprehensive, and deeply engineered contribution to the field of dynamic model ensembling.
