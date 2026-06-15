# Intermediate Review Phase 1: Summary of the Paper

## 1. Core Motivation and Problem Statement
The paper, titled **"ChemMerge: Non-Equilibrium Chemical Reaction Kinetics for Dynamic Model Merging"**, addresses a critical systems-level bottleneck in deep neural network serving: the trade-off between ensembling accuracy and serving stability under noisy, highly heterogeneous streaming workloads on edge hardware. 

Standard multi-task serving with modular parameter adapters (such as task-specific LoRA adapters fine-tuned on a frozen shared backbone) is highly appealing for edge devices due to its low storage footprint. However, existing ensembling and model merging paradigms suffer from three major limitations:
- **Heterogeneity Collapse in Static Merging:** Static weight-space merging (e.g., Task Arithmetic or TIES-Merging) averages task-specific parameters into a single monolithic model. When subjected to mixed-task streams, static merging catastrophically fails due to parameter interference and mutual neutralization.
- **Routing Weight Jitter in Stateless Ensembling:** Stateless activation-space ensembling (e.g., SABLE or SPS-ZCA) dynamically blends expert pathways sample-by-sample. However, they treat layers as independent, decoupled blocks. Under noisy, rapidly shifting streams, this lack of continuous state tracking leads to sharp, discontinuous ensembling coefficient oscillations ("routing weight jitter") across adjacent layers and downstream representational drift.
- **Latency Overheads in Stateful Schedulers:** Systems-level scheduling wrappers (such as Micro-Batch Homogenization - MBH) attempt to restore representational stability by buffering and grouping samples, but they require $O(K)$ sequential backbone passes, introducing severe latency overhead on real-time edge devices.

ChemMerge aims to resolve this fundamental accuracy-stability trade-off. By modeling the flow of representations through the depth of the network as a multi-component chemical reactor governed by non-equilibrium reaction kinetics, the authors introduce continuous concentration-based state variables $C_{k,b}^{(l)} \in [0, 1]^K$ that evolve smoothly layer-by-layer across sequential blocks, establishing a physical "representation inertia" (memory) that filters out high-frequency routing noise.

---

## 2. Proposed Methodology and Technical Components
The ChemMerge framework introduces three main training-free, activation-space components:

### A. Catalytic Zero-Shot Alignment (C-ZCA)
- Pre-computes task-specific centroids $\mu_k^{(3)}$ at the output of the frozen shared early layers (Layer 3) during a one-time, low-resource offline calibration phase (typically using 64 samples per task).
- Applies **Unit-Norm Calibration (UNC)** to derive the calibrated catalytic coordinate (cosine similarity) $u_{k, b} = \text{cos\_sim}(h_b^{(3)}, \mu_k^{(3)})$ to represent task manifolds.

### B. Non-Equilibrium Kinetic Routing (NEKR)
- Tracks a continuous, sample-wise expert concentration state vector $C_{k,b}^{(l)} \in [0, 1]^K$ across the depth of the network ($l \in [4, L]$), initialized uniformly as $C_{k,b}^{(3)} = 1/K$.
- Computes forward reaction rates $k_{k,b}^{(l)}$ using a temperature-scaled Arrhenius rate equation with a Catalytic Competition partition function:
  $$k_{k,b}^{(l)} = \frac{\exp\left( S(h_b^{(l-1)}, \mu_k^{(3)}) / \tau \right)}{\sum_{j=1}^K \exp\left( S(h_b^{(l-1)}, \mu_j^{(3)}) / \tau \right)}$$
  where $\tau$ is the routing reaction temperature.
- Models active concentration updates using a first-order kinetic rate equation representing a reversible chemical reaction:
  $$\frac{d C_{k}}{dt} = k_{k}^{(l)} (1 - C_{k}) - k_{\text{decay}} C_{k}$$
  where $k_{\text{decay}} \in [0, 1]$ is the back-reaction/decay rate that prevents representation saturation.
- Discretizes the ODE across layers using two alternative numerical solvers:
  - **Explicit Euler with Boundary Projection:** 
    $$C_k^{(l)} = \left[ C_k^{(l-1)} + \Delta t \left( k_k^{(l)} (1 - C_k^{(l-1)}) - k_{\text{decay}} C_k^{(l-1)} \right) \right]_0^1$$
    where $[z]_0^1$ clips concentrations to $[0, 1]$.
  - **Exact Analytical Exponential Integrator:** 
    $$C_k^{(l)} = C_k^{(l-1)} e^{-(k_k^{(l)} + k_{\text{decay}})\Delta t} + \frac{k_k^{(l)}}{k_k^{(l)} + k_{\text{decay}}} \left( 1 - e^{-(k_k^{(l)} + k_{\text{decay}})\Delta t} \right)$$
    This convex combination mathematically guarantees bounded concentrations within $[0, 1]$ under any virtual step size $\Delta t > 0$, bypassing heuristic clipping.
- Defines two operating modes: **Single-Centroid Mode** (stable spaces and memory savings) and **Multi-Centroid Mode** (for deep pre-trained models with severe representation drift).
- Integrates **Active Representation Coupling ($\eta \ge 0$)** to warp representations layer-by-layer toward the active centroids.

### C. Catalytic Activation Blending (CAB)
- Normalizes active concentrations using the Law of Mass Action:
  $$\alpha_{k, b}^{(l)} = \frac{C_{k, b}^{(l)}}{\sum_{j=1}^K C_{j, b}^{(l)}}$$
- Blends shared base model activations and parallel expert updates dynamically in a single parallel forward pass:
  $$h_b^{(l)} = h_{\text{base}, b}^{(l)} + \sum_{k=1}^K \alpha_{k, b}^{(l)} h_{\text{expert}, k, b}^{(l)}$$

---

## 3. Scope of Claims and Empirical Evaluations
The authors evaluate ChemMerge inside two distinct empirical environments:
- **Analytical Coordinate Sandbox (ICS) (Primary Accuracy Sweep):** A 14-layer, 192-dimensional synthetic simulation sandbox. All primary task results (MNIST, Fashion-MNIST, CIFAR-10, SVHN) are entirely simulated using synthetic orthogonal coordinates and hand-calibrated logit noise scales, rather than training or running actual neural networks on raw image pixels.
- **Vision Transformer (ViT-B/16) (Routing-Only Validation):** A validation of routing trajectories on frozen activations from a pre-trained ImageNet `ViT-B/16` model. This is a routing-only simulation on offline, frozen activations over PIL-generated synthetic geometric shapes (Circles, Squares, Triangles, Crosses). No task-specific expert adapters are trained or loaded, and no parallel ensembled forward execution passes (CAB) are executed in this section.

The authors claim that ChemMerge:
1. Recovers **98.81%** of the theoretical Expert Ceiling in the sandbox, outperforming stateless nearest-centroid routing (SPS-ZCA) by **+8.22%** absolute accuracy.
2. Is completely immune to Heterogeneity Collapse ($B=256$) and Vectorization Collapse ($B=1$) under constant $O(1)$ latency.
3. Successfully scales to high expert densities ($K=16$).
4. Achieves high final-layer routing accuracy ($93.20\%$) on pre-trained `ViT-B/16` while reducing layer-to-layer ensembling weight routing jitter by **9.9$\times$** compared to SPS-ZCA and over **2.15$\times$** compared to SABLE (at equivalent sensitivities).
5. Resolves the accuracy-stability trade-off where static digital smoothing filters (EMA) fail.
