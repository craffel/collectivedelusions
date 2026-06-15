# 1. Summary of the Paper

## Main Topic and Objective
The paper addresses the challenge of **test-time ensembling of parameter-efficient expert adapters (such as LoRA)** under dynamic, non-stationary workloads for resource-constrained, multi-task edge serving. 
The core objective is to resolve the **stability-accuracy bottleneck** in dynamic model merging. While stateless dynamic ensembling methods (like SABLE) compute routing similarities independently at each layer, they suffer from extreme layer-to-layer ensembling weight jitter due to fluctuating intermediate representations. Conversely, existing state-dependent first-order methods (like EMA or ChemMerge) introduce severe phase lag in the closed feedback loop, leading to overshoots, oscillations, and a significant drop in serving accuracy.

---

## Proposed Approach: GraviMerge
The authors present **GraviMerge**, a mathematically rigorous, physics-informed model merging framework inspired by classical mechanics and orbital dynamics. Instead of using static distances or linear chemical decay, GraviMerge models deep representation routing as a second-order multi-body physical system on a curved manifold.

Key architectural and mathematical components of the GraviMerge framework include:
1. **Coordinate Space and Spacecraft Probe:** The representation space is mapped to the unit hypersphere $\mathbb{S}^{D-1}$ to align with cosine routing. The intermediate activation trajectory is modeled as a virtual stateful "spacecraft coordinate probe" ($\mathbf{h}_{\text{sc}}^{(l)} \in \mathbb{S}^{D-1}$) traversing a latent gravitational field.
2. **Task Centroids as Stellar Attractors:** The pre-trained expert centroids ($\boldsymbol{\mu}_k$) are modeled as stationary celestial bodies of varying gravitational mass ($M_k$) exerting softened gravitational forces.
3. **Arrhenius Mass Activation (AMA):** Test-time zero-shot alignment is used to compute dynamic task attractor masses ($M_k \in (0, 1]$) based on the cosine similarity between the initial spacecraft state (Layer 3) and the centroids, stabilized via maximum similarity subtraction.
4. **Geodesic Trajectory Integration (GTI):** Net gravitational forces, velocity, and viscous drag are integrated. To maintain strict manifold adherence on $\mathbb{S}^{D-1}$, forces, acceleration, and velocity are projected onto the local tangent space, and the coordinates are updated via the exact spherical **Exponential Map** (geodesic step). The velocity vector is carried forward via closed-form **Parallel Transport** to prevent numerical drift.
5. **Gravitational Influence Blending (GIB):** Ensembling weights ($\alpha_k^{(l)}$) are defined as continuous, differentiable functions proportional to the relative magnitude of the softened gravitational forces, ensuring smooth layer-to-layer weight trajectories.
6. **Coupled GraviMerge (Closed-Loop Variant):** Incorporates an active feedback force ($\mathbf{F}_{\text{feedback}}^{(l)}$) pulling the spacecraft probe toward the actual propagating neural activations, enabling closed-loop tracking of representational shift.
7. **Temporal State Carryover:** Velocity states ($\mathbf{v}^{(L)}$) are carried over between sequential edge queries in a continuous stream, introducing inter-query physical momentum to adapt to task boundary shifts smoothly.

---

## Key Findings and Empirical Results
The authors evaluate GraviMerge on a Projected Digit Representation Space (RDS) Proxy benchmark (using projected scikit-learn digits to model semantic clustering) across $10$ independent seeds and three serving scenarios (Homogeneous, Heterogeneous, and Real-Time $B=1$):
* **High Joint Serving Accuracy:** GraviMerge achieves **88.69%** serving accuracy, outperforming stateless SABLE ($87.65\%$) and SPS-ZCA ($88.51\%$), confirming that it preserves and enhances dynamic ensembling accuracy.
* **Dramatic Jitter Reduction:** GraviMerge slashes layer-to-layer ensembling weight jitter to **0.00190 MAD**, representing a **6.01$\times$** reduction compared to ChemMerge ($0.01141$ MAD), a **5.47$\times$** reduction compared to first-order EMA, and a **2.40$\times$** reduction compared to SABLE ($0.00456$ MAD).
* **Superiority Over Weight-Space Second-Order Momentum (WMomentum):** WMomentum degrades accuracy ($87.09\%$) and increases jitter to $0.02763$ MAD ($14.54\times$ worse than GraviMerge) due to probability-simplex clipping discontinuities ("chatter").
* **Comparison with Kalman Filter Tracking:** A mathematically optimal first-order Kalman Filter fails to stabilize routing jitter ($0.00447$ MAD), proving the unique advantage of GraviMerge's second-order velocity/inertia states.
* **Noise and Dimensional Robustness:** Extends flawlessly to $D=768$ (GPT-2 scale) and $D=4096$ (Llama-3 scale) dimensions. Under layer-specific representational drift, it achieves up to a **$1.06 \times 10^6\times$ jitter reduction** compared to SABLE, while remaining robust to environmental noise and out-of-distribution streams via Sentinel Attractor Dynamics (SAD).

---

## Explicitly Claimed Contributions and Supporting Evidence
The paper explicitly claims four main contributions, which are supported by detailed evidence in the manuscript:
1. **A Physics-Informed Routing Paradigm:** Introduces GraviMerge as a robust and interpretable physical formulation of representational stability. *Evidence: Conceptually detailed in Section 3.1 and Figure 1.*
2. **Novel Physical and Geometric Mechanisms (AMA, GTI, GIB):** Formulates Arrhenius Mass Activation, Geodesic Trajectory Integration (tangent projections, exponential maps, parallel transport), and Gravitational Influence Blending. *Evidence: Defined mathematically in Equations 1–10 in Section 3.*
3. **Resolution of the Accuracy-Stability Dilemma:** Achieves top ensembling accuracy while minimizing layer-wise weight jitter. *Evidence: Exhaustive evaluation over 10 seeds, Table 1, and Figure 1b.*
4. **Resilience to Dynamic Workloads:** Demonstrates flat accuracy profiles and high throughput suitability across diverse serving sizes ($B=256$ and $B=1$) and sequential streams. *Evidence: Quantified in Table 1 (Homogeneous vs. Heterogeneous vs. Real-Time) and Table 3.*
