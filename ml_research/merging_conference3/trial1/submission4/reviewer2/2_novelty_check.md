# Novelty Check

## Key Novel Aspects
1. **Physical Metaphor for Parameter Merging:** Viewing the trajectory of model parameters during post-hoc test-time adaptation as a continuous physical fluid flow (advection-diffusion fluid phases coalescing) is an original conceptual perspective. Applying physical conservation laws to the weights themselves rather than activations or data distributions is a fresh framing.
2. **Fisher-Information-based Viscosity:** Formulating a functional, coordinate-free, and permutation-invariant viscosity operator based on the empirical Fisher Information Matrix to prevent representational tearing during parameter adaptation.
3. **Hybrid Continuous-Discrete Dynamical System:** Designing a coupling where the shared image encoder evolves along a discretized continuous physical flow while the task-specific classification heads adapt via discrete-time momentum-based optimization (Adam).

## The 'Delta' from Prior Work
1. **From Static Merging (Task Arithmetic, Ties-Merging, DARE, RegMean) to Dynamic Trajectories:** Standard methods perform static algebraic combinations in parameter space. The delta is that FluidMerge models a continuous-time trajectory that allows for post-hoc refinement on unlabeled test data.
2. **From Parameter-Efficient/Single-Layer TTA (AdaMerging, SyMerge) to Full-Encoder Adaptation:** Prior dynamic methods restrict updates to merging coefficients or a single classification/projection layer. FluidMerge adapts the entire 86M parameter encoder, constrained by a function-sensitive viscosity metric.
3. **De-escalating Metaphorical Novelty:** The authors demonstrate that once discretized and executed, the core techniques are mathematically equivalent to:
   - **Expert-Weighted Boundary Conditions $\equiv$ Task Arithmetic (TA) initialization** (Ilharco et al., 2023).
   - **Fisher-Information Viscosity under Euler Integration $\equiv$ Gradient descent under an Elastic Weight Consolidation (EWC) penalty** (Kirkpatrick et al., 2017) anchoring parameters to their initial TA state.
   - **Advection force $\equiv$ Teacher-Student soft-label distillation** (Jung et al., 2025).

Thus, while the physical analog is a novel packaging, the actual algorithmic "delta" is the combination and simultaneous application of these three existing techniques (Task Arithmetic initialization, full-encoder teacher-student TTA, and EWC regularization) for multi-task model merging.

## Characterization of Novelty
- **Conceptual Novelty: Moderate-to-Significant.** Framing weight trajectories as fluid flows is highly creative, and demonstrating how a spatial Laplacian baseline fails due to permutation invariance (the "permutation invariance paradox") provides strong conceptual depth.
- **Practical/Algorithmic Novelty: Incremental.** Since the resulting updates are mathematically isomorphic to running standard gradient descent with an EWC penalty starting from a Task Arithmetic initialization, the actual operational novelty is incremental. The method is a repackaging of known deep learning techniques (EWC, Task Arithmetic, and pseudo-labeling) into a single pipeline. From an engineering and deployment perspective, the contribution lies in showing that this specific combination is highly effective at refining merged models, rather than introducing completely new optimization primitives.
