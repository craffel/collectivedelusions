# Intermediate Review Step 2: Novelty Check

## Characterization of Novelty
The novelty of this paper is characterized as **significant and highly creative**. It bridges two seemingly disparate disciplines—systems biochemistry (non-equilibrium chemical kinetics) and deep learning architecture ensembling—to solve a long-standing physical instability in dynamic model serving. 

While individual components like cosine-similarity-based nearest-centroid routing (SPS-ZCA) and activation blending (SABLE) are known, the conceptualization of deep neural layers as sequential steps in a continuous chemical reactor cascade is entirely novel. The introduction of stateful, continuous-depth concentration tracking (NEKR) is a substantial departure from the dominant paradigm of stateless, layer-decoupled routing.

---

## The "Delta" from Prior Work

### 1. Delta from Static Model Merging (Task Arithmetic, TIES-Merging, DARE)
- **Prior Work:** Static merging averages fine-tuned parameter weights into a single monolithic model.
- **The ChemMerge Delta:** Static merging is lossy and collapses under heterogeneous, mixed-task streaming workloads due to parameter interference (*Heterogeneity Collapse*). ChemMerge performs dynamic, activation-space blending sample-wise, preserving the original specialized performance of each expert and achieving complete immunity to heterogeneity collapse.

### 2. Delta from Standard Mixture-of-Experts (MoE) and Parametric Routing
- **Prior Work:** Classic MoEs (e.g., Switch Transformers, Vision MoEs) route samples via parameterized gating networks that must be trained jointly with the experts from scratch. Under sample-by-sample serving ($B=1$), parametric routers suffer from *Vectorization Collapse* (producing erratic routing due to a lack of batch statistics).
- **The ChemMerge Delta:** ChemMerge is completely training-free and can be applied post-hoc to independently trained PEFT adapters. It utilizes pre-computed early-layer centroids to route without any parameter optimization, maintaining flat and optimal ensembling performance even at $B=1$ without any vectorization collapse.

### 3. Delta from Systems-Level Scheduling (Micro-Batch Homogenization)
- **Prior Work:** MBH buffers and groups heterogeneous streaming samples into homogeneous batches to prevent collapse.
- **The ChemMerge Delta:** MBH restores stability but incurs an $O(K)$ sequential backbone latency penalty, which violates real-time edge Serving requirements. ChemMerge processes samples on-the-fly in a single parallel pass with a constant, stateless $O(1)$ latency.

### 4. Delta from Existing Post-Hoc Dynamic Ensembling (SABLE, SPS-ZCA)
- **Prior Work:** SABLE and SPS-ZCA perform training-free dynamic ensembling but are fundamentally *stateless*. They evaluate cosine similarities either once globally or independently at each layer block. Due to local representation noise and non-orthogonal manifolds, this leads to sharp, high-frequency ensembling coefficient oscillations (*routing weight jitter*) across depth, causing representational drift and activation spikes.
- **The ChemMerge Delta:** ChemMerge rejects stateless execution. It models ensembling weights as physical concentrations $C_k^{(l)}$ that evolve continuously across depth via a first-order chemical kinetics ODE. This introduces physical temporal memory (inertia), acting as a state-dependent adaptive low-pass filter that suppresses routing jitter by up to 9.9$\times$ compared to SPS-ZCA and over 2.15$\times$ compared to SABLE (under equivalent sensitivity).

### 5. Mathematical Delta: Exact Exponential Integration
- **Prior Work:** Numerical solvers for physical ODEs in neural architectures (like Neural ODEs) rely on explicit Runge-Kutta or Euler schemes, which require heuristic projection clipping or small step-size constraints to prevent numerical divergence.
- **The ChemMerge Delta:** The authors derive and implement an exact analytical Exponential Integrator (Eq. 9). This scheme ensures that updated concentrations are a strict convex combination of previous concentrations and steady-state equilibrium, guaranteeing that concentrations remain bounded in $[0, 1]$ for any virtual step size $\Delta t > 0$ without heuristic clipping or numerical instability.
