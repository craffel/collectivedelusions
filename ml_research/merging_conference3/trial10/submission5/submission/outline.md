# Unitary Geodesic Routing (UGR): Detailed Paper Outline

## Section 0: Abstract (00_abstract.tex)
- **Context:** Test-time model ensembling across non-stationary heterogeneous task streams is critical for adaptive serving.
- **Problem:** Current stateful approaches (like ChemMerge, Momentum-Merge) perform continuous or exponential updates in flat Euclidean spaces, relying on post-hoc Softmax normalization. This compromises geometric scale, causes representation lag under sudden switches, and is mathematically unconstrained.
- **Proposed Solution:** Unitary Geodesic Routing (UGR), a quantum-inspired, curved state-space routing paradigm on the hypersphere $\mathbb{S}^{K-1}$.
  - Ensembling weights are coordinate-wise squared magnitudes of the state (Born's rule), satisfying simplex constraints natively.
  - State updates are closed-form geodesic rotations (spherical interpolations) along the shortest great-circle path.
  - Step-size is modulated by Torque-Driven Adaptive Agility, exploding on task switches to eliminate lag and vanishing on stable streams to eliminate jitter.
  - Spatial-Temporal Geodesic Coupling propagates state across sequential queries.
- **Results:** Evaluated on the Analytical Coordinate Sandbox (ICS). Outperforms ChemMerge by **+5.43%** classification accuracy, reduces routing jitter ($L \ge 5$) by **2.10x** to near-zero (0.001951), and runs completely training-free without ODE solvers.

## Section 1: Introduction (01_intro.tex)
- **Background:** Test-time adaptation (TTA) and multi-task learning. Parameter-efficient fine-tuning (PEFT) has enabled task-specific expert adapters. Test-time ensembling fuses these experts dynamically.
- **The Core Challenge:** Real-world task streams are highly non-stationary, requiring a stateful router to handle sequential dependencies.
- **Critical Audit of Existing Stateful Routers:**
  - Standard Euclidean EMA (e.g., Momentum-Merge) and continuous biochemical systems (e.g., ChemMerge) perform linear blending in unconstrained spaces.
  - Normalizing these updates with post-hoc Softmax maps creates "representational lag" (hysteresis) and "scale-mismatch" because the Euclidean distance does not align with the probability manifold.
- **The Visionary Paradigm Shift:** Reject flat spaces. Model the ensembling state on the curved $(K-1)$-sphere $\mathbb{S}^{K-1}$.
  - Under Born's rule, projection onto the simplex is exact, Softmax-free, and norm-preserving.
  - Geodesic flow ensures updates follow the natural geometry of the probability manifold.
  - Torque-driven agility mimics torque-induced physical acceleration to override inertia.
- **Key Contributions:**
  1. Spherical State Representation & Born's Rule Simplex Mapping.
  2. Closed-Form Rodrigues-like Geodesic Rotation (Spherical EMA) bypassing matrix exponential.
  3. Torque-Driven Adaptive Agility to resolve the stability-plasticity trade-off.
  4. Spatial-Temporal Geodesic Coupling for seamless sample-to-sample recurrence.
  5. State-of-the-art results on ICS: +5.43% accuracy and 2.10x lower jitter ($L \ge 5$) than ChemMerge.

## Section 2: Related Work (02_related_work.tex)
- **Parameter-Efficient Fine-Tuning & Model Merging:** Fusing PEFT adapters (LoRA, adapters) for multi-task learning (Model Soups, Ties-Merging, AdaMerging). Focus is mostly static/offline.
- **Test-Time Adaptation & Dynamic Ensembling:** Adapting models on-the-fly (Tent, SABLE).
- **Stateful Sequential Serves:** Prior arts Momentum-Merge (minimalist Euclidean EMA), ChemMerge (metaphorical biochemical kinetics), PAC-Kinetics.
- **Geometric & Quantum-Inspired Machine Learning:** Spherical manifolds, Slerp, quantum probability representations in neural networks. Highlight that UGR is the first to combine these for stateful test-time ensembling.

## Section 3: Methodology (03_method.tex)
- **Problem Setting:** Input stream $X_t$, frozen backbone, $K$ expert adapters, dynamic weights $\boldsymbol{\alpha}_t^{(l)}$ at layer $l$.
- **Spherical State Representation & Born's Rule:**
  - Define state $\mathbf{s}_t^{(l)} \in \mathbb{S}^{K-1}$.
  - Map to simplex: $\alpha_{k, t}^{(l)} = (s_{k, t}^{(l)})^2$. Prove why this is exact and norm-preserving.
- **Bottom-Up Target Vector Construction:**
  - Compute similarities to task centroids, apply Softmax with temperature $\tau$ to get $\mathbf{e}_t^{(l)}$.
  - Map to unit sphere: $\mathbf{w}_t^{(l)} = \mathbf{e}_t^{(l)} / (\|\mathbf{e}_t^{(l)}\|_2 + \epsilon)$.
- **Geodesic Rotation Operator (Spherical EMA):**
  - Compute alignment $c = \mathbf{s}^T \mathbf{w}$.
  - Extract orthogonal target component $\mathbf{u}$.
  - Compute angle $\phi = \arccos(c)$, scale by inertia coefficient: $\theta = \eta \phi$.
  - Geodesic rotation: $\mathbf{s}_t^{(l)} = \cos(\theta) \mathbf{s}_t^{(l-1)} + \sin(\theta) \mathbf{u}$. Prove norm preservation.
- **Torque-Driven Adaptive Agility:**
  - Explain how the step-size scales with torque (angular distance $\phi$), enabling fast switches and quiet stability.
- **Spatial-Temporal Geodesic Coupling:**
  - Describe the boundary recurrence: $\mathbf{s}_t^{(L_{\text{frozen}})} = \mathbf{s}_{t-1}^{(L)}$.

## Section 4: Experimental Evaluation (04_experiments.tex)
- **Experimental Setup:**
  - The 14-layer, 192-dimensional Analytical Coordinate Sandbox (ICS).
  - Stream settings: heterogeneous tasks with transitions.
- **Baselines:** Uniform, SABLE (stateless), ChemMerge (SOTA biochemical), Momentum-Merge (SOTA Euclidean EMA), PAC-Kinetics.
- **Main Quantitative Results Table:** Joint Mean Accuracy (%) and Mean Routing Jitter across 10 random seeds.
- **In-Depth Scientific Analysis:**
  - *The Magic of Non-Euclidean Geodesic Flow:* Why flat space updates fail (representational distortion, Softmax warping) and why UGR succeeds.
  - *Jitter Suppression & Agility:* How spatial-temporal coupling and torque-driven agility solve the stability-plasticity dilemma.
- **Hyperparameter Sensitivity Analysis & Grid Sweep:**
  - 2D grid sweep of $\eta$ and $\tau$ across 3 seeds. Discuss the sharp gating regime and low-pass filtering behavior.

## Section 5: Conclusion & Future Directions (05_conclusion.tex)
- **Summary of Work:** Successful formulation of Unitary Geodesic Routing (UGR).
- **Impact:** Establishes a new geometric paradigm for stateful serving, completely eliminating the need for unconstrained Euclidean state updates.
- **Future Directions:** Scaling to LLM ensembling, training-time geometric routers, and formalizing a general Lie group formulation of model ensembling.
