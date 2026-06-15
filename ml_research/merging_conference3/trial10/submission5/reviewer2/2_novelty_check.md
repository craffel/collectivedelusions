# 2. Novelty Check

## Assessment of Key Novel Aspects
The paper introduces several highly novel concepts to the domain of test-time model ensembling and Mixture-of-Experts (MoE) routing:
1. **Hyperspherical State Space ($\mathbb{S}^{K-1}$):** Operating entirely on a curved, non-Euclidean manifold for online ensembling states. Previous methods (e.g., Momentum-Merge, ChemMerge) rely on flat Euclidean spaces ($\mathbb{R}^K$), treating coordinates as independent variables and projecting back to the probability simplex post-hoc.
2. **Born Simplex Projection (Fisher-Rao Geodesic Flow):** Using the coordinate-wise square-root homeomorphism ($\alpha_k = s_k^2$)—an idea borrowed from classical Information Geometry (historical Hellinger/Bhattacharyya mapping) and sharing a beautiful analogy with Born's rule in quantum mechanics—to project spherical coordinates back to the probability simplex in a Softmax-free and scale-preserving manner. Bypassing the Softmax layer prevents representational compression at simplex boundaries.
3. **Rodrigues-like Spherical Linear Interpolation (Slerp):** Deriving a closed-form, computationally efficient spherical rotation operator to interpolate along the shortest great-circle path of the hypersphere. This completely bypasses the costly virtual-time numerical ODE integration steps required by continuous-time biochemical kinetic models (ChemMerge) or matrix exponentials.
4. **Torque-Driven Step-Size Adaptation:** Designing a self-regulating control loop where the rotational step size $\theta_t^{(l)}$ is dynamically scaled in proportion to the "representational torque" $\phi_t^{(l)} = \arccos(c_t^{(l)})$ (the angular distance between the current state and bottom-up target vector). This acts as a first-order non-linear dynamical system with non-linear damping, eliminating representational lag under sudden transitions while maintaining pristine stability when the stream is stationary.

## The 'Delta' from Prior Work
The table below highlights the key differences between UGR and existing test-time ensembling paradigms:

| Architectural Property | Stateless (SABLE) | Stateful Flat (Momentum-Merge) | Stateful Continuous (ChemMerge) | **Unitary Geodesic Routing (UGR)** |
| :--- | :--- | :--- | :--- | :--- |
| **State Space** | None (Stateless) | Flat Euclidean ($\mathbb{R}^K$) | Flat Euclidean ($\mathbb{R}^K$) | **Curved Hypersphere ($\mathbb{S}^{K-1}$)** |
| **Simplex Projection** | Softmax | Softmax | Softmax or Clipping | **Born Mapping (Softmax-Free)** |
| **Temporal Updates** | None (Instantaneous) | Flat Linear (EMA) | Continuous-Time ODEs | **Curved Geodesic Slerp** |
| **Computational Method** | Direct | Direct | Iterative ODE Solvers (Euler/Heun) | **Closed-Form Algebraic** |
| **Agility Adaptation** | High (No smoothing) | Constant inertia ($\beta$) | Linear concentration potentials | **Non-linear Torque Feedback** |
| **Temporal Coupling** | None | Intra-Query only | Intra-Query only | **Spatial-Temporal 2D coupling** |

## Characterization of Novelty
The novelty of this work is **significant**. Rather than introducing minor, incremental improvements or hyperparameter tuning to existing flat-space architectures, the authors reject the unconstrained flat-space paradigm altogether. They provide a cohesive, mathematically rigorous, and physically inspired geometric alternative. 

The paper beautifully bridges three separate fields:
* **Information Geometry** (the square-root map representing exact Fisher-Rao geodesic flows on the probability simplex),
* **Physical Mechanics** (the first-order torque control loop acting as a non-linearly damped dynamical system), and
* **Parameter-Efficient Deep Learning Serving** (on-the-fly, training-free token-level or query-level LoRA ensembling).

The algebraic derivations for Slerp updates, proofs of positive orthant persistence (Appendix A), and the derivation of backpropagation gradients (Appendix A) are complete, elegant, and highly original, moving far beyond standard heuristics.
