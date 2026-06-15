# Evaluation Task 1: Summary of the Paper

## Main Topic and Motivation
This paper addresses the critical problem of **dynamic model merging and ensembling** on edge devices under streaming, heterogeneous, and noisy workloads. In real-world edge deployment (e.g., robotics, IoT), input streams shift sample-by-sample across different tasks. 

Existing approaches fall into two categories:
1. **Static weight-space merging** (e.g., Task Arithmetic, TIES-Merging, DARE), which merges expert weights once prior to inference. This is computationally efficient ($O(1)$ latency) but suffers from **Heterogeneity Collapse** when processing mixed-task streams because conflicting parameters neutralize one another.
2. **Dynamic activation-space ensembling** (e.g., SABLE, SPS-ZCA), which routes samples dynamically during the forward pass. These methods are fundamentally bottlenecked by their **stateless, layer-wise decoupled execution**, treating each layer as an independent routing block. Under noisy or rapidly switching input streams, this statelessness leads to high-frequency ensembling coefficient oscillations (**routing weight jitter**), representation saturation, and activation spikes across depth.
3. **Systems-level scheduling wrappers** (e.g., Micro-Batch Homogenization - MBH), which group and queue samples by task identity. While this restores representation stability, it introduces a prohibitive $O(K)$ sequential latency penalty on edge hardware (requiring $K$ full backbone passes), violating real-time serving constraints.

To resolve this fundamental accuracy-latency-stability trade-off, the paper proposes **ChemMerge**, a training-free, continuous-time ensembling paradigm inspired by non-equilibrium chemical reaction kinetics.

---

## Technical Approach
ChemMerge models the forward representation flow through a deep network's depth as a multi-component chemical reactor:
- **Task experts** (specifically LoRA adapters) act as reactive chemical species.
- **Task-specific early-layer centroids** act as catalytic enzymes that lower reaction energy barriers.
- **Hidden activations** act as the reacting solution.
- Instead of calculating stateless similarities at each layer, ChemMerge tracks a continuous, sample-wise expert concentration state vector $C_b^{(l)} \in [0, 1]^K$ that evolves across the depth of the network.

The core methodology consists of three components:
1. **Catalytic Zero-Shot Alignment (C-ZCA):** Consists of a one-time calibration phase where task centroids $\mu_k^{(l)}$ are extracted. Two modes are defined: *Global Early-Layer Anchoring* (Single-Centroid Mode) which anchors rate calculations to early shared features to minimize parameter storage ($O(K \cdot D)$), and *Layer-Specific Centroid Routing* (Multi-Centroid Mode) which queries layer-wise centroids of size $O(L \cdot K \cdot D)$ to handle non-linear representation shifts in pre-trained models.
2. **Non-Equilibrium Kinetic Routing (NEKR):** Computes a temperature-scaled Arrhenius equation based on cosine similarity to model forward reaction rates $k_k^{(l)}$, then evolves the expert concentration state variables $C_k^{(l)}$ across depth via a system of coupled first-order ODEs:
   $$\frac{d C_{k}}{d t} = k_{k}^{(l)} (1 - C_{k}) - k_{\text{decay}} C_{k}$$
   The authors present two discretization schemes for this ODE across layers:
   - *Explicit Euler with Boundary Projection:* An explicit step with non-linear clipping to $[0, 1]$.
   - *Exact Analytical Exponential Integrator:* A convex combination scheme that guarantees stability inside $[0, 1]$ without heuristic clipping:
     $$C_{k}^{(l)} = C_{k}^{(l-1)} e^{-(k_{k}^{(l)} + k_{\text{decay}})\Delta t} + \frac{k_{k}^{(l)}}{k_{k}^{(l)} + k_{\text{decay}}} \left(1 - e^{-(k_{k}^{(l)} + k_{\text{decay}})\Delta t}\right)$$
   An *Active Representation Coupling* mechanism ($\eta > 0$) is also introduced to warp representations toward active task centroids layer-by-layer, though the default is set to $\eta=0$ for heterogeneous streams.
3. **Catalytic Activation Blending (CAB):** Blends base and expert activations in a single parallel forward pass using ensembling weights $\alpha_k^{(l)}$ derived from active concentrations proportional to the Law of Mass Action:
   $$\alpha_{k, b}^{(l)} = \frac{C_{k, b}^{(l)}}{\sum_{j=1}^K C_{j, b}^{(l)}}$$

---

## Key Findings
1. **Performance Ceiling Recovery:** Inside a 14-layer high-fidelity *Analytical Coordinate Sandbox (ICS)* (simulating MNIST, Fashion-MNIST, CIFAR-10, and SVHN streams across 10 random seeds), ChemMerge achieves a Joint Mean accuracy of **78.11%** (homogeneous) and **78.06%** (heterogeneous), recovering **98.81%** of the Expert Oracle ceiling (79.00%).
2. **Robustness to Collapse:** ChemMerge exhibits absolute robustness to Heterogeneity Collapse ($B=256$) and Vectorization Collapse ($B=1$), maintaining flat performance. Unregularized parametric routers (Linear, QWS-Merge) collapse under heterogeneous serving, and SPS-ZCA collapses under manifold entanglement.
3. **O(1) Edge Serving Latency:** ChemMerge maintains a constant $O(1)$ single-pass inference latency. Unlike the PFSR+MBH wrapper (which requires $4\times$ sequential latency), ChemMerge matches its accuracy with $1\times$ computational overhead.
4. **Dramatic Jitter Reduction:** In a routing-only simulation on a pre-trained Vision Transformer ($\text{ViT-B/16}$) over synthetic shape streams, ChemMerge reduces layer-to-layer ensembling routing jitter by **9.9$\times$** compared to SPS-ZCA and by over **2.15$\times$** compared to SABLE (at equivalent routing sensitivities).
5. **State-Dependent Smoothing Superiority:** Compared to static Exponential Moving Average (EMA) smoothing of routing weights, ChemMerge's state-dependent adaptive kinetics avoid the "lag-accuracy trade-off." Static EMA drops accuracy by $-4.2\%$ to match ChemMerge's low routing jitter, whereas ChemMerge maintains both high accuracy ($93.20\%$) and low jitter ($0.0156$).

---

## Explicitly Claimed Contributions and Supporting Evidence
- **Contribution 1: Biochemical Paradigm for Deep Architectures.** 
  *Evidence:* Mathematical formulation mapping experts to chemical species, centroids to catalysts, and depth to reaction time.
- **Contribution 2: Non-Equilibrium Kinetic Routing (NEKR).** 
  *Evidence:* System of coupled linear first-order ODEs and derivation of both explicit Euler and exact Exponential Integrator schemes.
- **Contribution 3: Evaluation in a High-Fidelity Simulation Sandbox (ICS).** 
  *Evidence:* Evaluation of joint mean accuracy, heterogeneity robustness, and layer trajectories across 10 independent random seeds.
- **Contribution 4: Routing-Only Validation on Pre-trained ViT-B/16.** 
  *Evidence:* PyTorch hook activations extracted from 12 encoder layers of a ViT-B/16 model on geometric shape streams, reporting routing accuracy and layer-to-layer ensembling weight variance (routing jitter).
- **Contribution 5: Robustness to Structural Collapses and Entanglement.** 
  *Evidence:* Systematic sweeps of batch sizes ($B=1$ to $250$), task manifold entanglement ($\rho \in [0.0, 0.5]$), and expert scaling ($K \in \{4, 8, 12, 16\}$).
