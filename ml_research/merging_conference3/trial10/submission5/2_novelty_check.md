# Novelty and Originality Check

## 1. Core Mathematical and Geometric Novelty
The core novelty of the paper lies in replacing unconstrained flat-space updates and post-hoc Softmax projection with native curved-manifold flow on the unit hypersphere $\mathbb{S}^{K-1}$, mapped to the probability simplex via the square-root mapping (Born mapping) from Information Geometry.
- **Why this is novel:** While Information Geometry and the square-root (Hellinger/Bhattacharyya) mapping are classical mathematical concepts, applying them to the field of stateful test-time model ensembling is highly original. 
- **The Born Mapping Connection:** The paper connects Born's rule of quantum mechanics ($\alpha_k = s_k^2$) with the Fisher-Rao metric on the simplex. Geodesic rotations on the hypersphere map natively to exact Fisher-Rao geodesic flows on the probability simplex, completely bypassing Softmax projection for the state representation.

## 2. Algorithmic and Mechanistic Novelty
- **Closed-Form Rodrigues-like Slerp:** The paper derives a closed-form Rodrigues-like formulation to perform spherical linear interpolation (Slerp) on the hypersphere. This bypasses expensive matrix exponentials (typically required for Lie group updates) and virtual-time numerical ODE integration (required by *ChemMerge*), achieving linear scalability $\mathcal{O}(K)$ with the size of the expert pool.
- **Torque-Driven Adaptive Agility:** This physics-inspired mechanism uses the angular distance (representational torque) $\phi = \arccos(c)$ to scale the angular velocity. It acts as a first-order non-linear dynamical system with non-linear damping. Unlike second-order momentum methods, it is mathematically guaranteed to avoid overshoot and oscillation, representing a significant control-theoretic advancement for stability-plasticity trade-offs.
- **Spatial-Temporal Geodesic Coupling:** While carrying over history across query boundaries has been used in some sequential processing, doing so within a curved geodesic framework and smoothly coupling spatial (layer-to-layer) and temporal (query-to-query) trajectories is a highly elegant and novel design.

## 3. Critical Critique of Conceptual Clashes
While the mathematical formulation is elegant, there is a conceptual clash regarding the "Softmax-Free" claim:
- **Softmax in Target Construction:** Standard UGR relies on a localized Softmax over cosine similarities (Equation 10) to construct the bottom-up target vector $\mathbf{e}_t^{(l)}$, which is then projected onto the sphere.
- **Softmax-Free vs. Softmax-Dependent:** Although the *state representation*, *geodesic updates*, and *simplex projections* are Softmax-free (preventing scale compression during propagation), the target vector itself still relies on Softmax in the standard setup. The authors propose a fully Softmax-free target ablation (Equation 11 using ReLU and $L_1$-normalization), but this variant incurs a noticeable drop in accuracy on both benchmarks (72.73% on synthetic sandbox, 87.40% on real-world text classification). This reveals that standard UGR's peak performance is still partially dependent on a Softmax operation, which slightly dilutes the "completely Softmax-free" narrative.

## 4. Positioning and Comparison with Prior Literature
The paper provides a comprehensive related work section, clearly positioning UGR relative to:
- **Parameter-Efficient Fine-Tuning and Model Merging:** Distinguishes UGR's *online, test-time* ensembling from *offline, static* merging techniques like Model Soups, Ties-Merging, or AdaMerging.
- **Test-Time Adaptation (TTA) and Dynamic Ensembling:** Contrasts UGR's stateful, temporally coherent routing with stateless approaches (SABLE, Tent) which suffer from severe routing oscillations.
- **Stateful Serving and Temporal Routing:** Directly addresses and solves the unconstrained-to-constrained mismatch and representational hysteresis of stateful flat-space models (*Momentum-Merge*, *ChemMerge*).

## Novelty Conclusion
The novelty of this paper is highly significant. It is not an incremental combination of existing methods; rather, it introduces a fundamentally different geometric paradigm for stateful test-time adaptive serving. However, the "completely Softmax-free" claim is slightly overstated given the performance dependence on the target Softmax.
