# 2. Novelty Check

This section assesses the novel aspects of **PID-Merge**, characterizes the 'delta' from prior literature, and evaluates its significance.

## Characterization of Novelty
The novelty of this work is **significant and highly creative**. It bridges two historically distinct fields: classical process control theory and dynamic test-time model ensembling (PEFT serving). While PID control has previously been used to adapt hyper-parameters in deep learning (e.g., learning rate scheduling or GAN stabilization), this is the first work to apply closed-loop PID control to dynamic, sample-wise layer-stateful weight ensembling in multi-tenant serving workloads.

## The 'Delta' from Prior Work
1. **From Stateless Routers (e.g., SABLE, SPS-ZCA):** 
   - *Stateless:* Calculate ensembling weights at every layer independently, which ignores the depth-wise dependency of representations. This results in severe layer-to-layer weight oscillations (the "routing jitter paradox") due to activation noise.
   - *PID-Merge Delta:* Introduces a depth-wise stateful low-pass filter that smooths out layer-to-layer oscillations, ensuring stable layer-wise convergence within 2--3 layers.
2. **From Prior Stateful Routers (e.g., ChemMerge, Momentum-Merge):**
   - *Prior Stateful:* Use continuous-time ODE kinetics (ChemMerge) or open-loop Exponential Moving Average (Momentum-Merge) to accumulate historical routing weights. Because they are open-loop and do not feed back active ensembling weights, they suffer from severe "inertial drag" (phase delay), lagging during rapid task transitions. Furthermore, ChemMerge carries state across separate user queries, violating multi-tenant security/privacy boundaries and risking cross-user leakage, while requiring costly ODE solver integrations.
   - *PID-Merge Delta:* Treats ensembling as a discrete-time closed-loop control system that monitors tracking errors at each layer. It resets controller states per-query to enforce complete tenant isolation. Crucially, it leverages the **Derivative (D) term** (error acceleration tracking) to anticipate task boundaries, completely suppressing phase lag and enabling near-instant adaptation (within 2--3 layers) without carrying state across steps.
3. **From Standard PID Applications:**
   - Adapts traditional continuous-time PID to depth-wise discrete-time updates on a probability simplex.
   - Formulates specialized deep-learning control constructs: logit mean-centering for float overflow protection, $K$-scaled conditional integration (anti-windup clamping) to prevent saturation-induced transition delays in deep topologies, and Softplus transformations to guarantee positive gains.

## Level of Novelty
The proposed framework is a substantial step forward. Rather than introducing marginal or heuristic adjustments to chemical kinetic metaphors, it introduces a rigorous, control-theoretic closed-loop feedback design that is mathematically grounded, systems-aware, and computationally lightweight.
