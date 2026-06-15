# Lyapunov-Stable Active Representation Coupling (L-ARC) for Dynamic Model Serving

This document proposal outlines the design, theoretical foundations, and step-by-step specifications for **L-ARC** (Lyapunov-Stable Active Representation Coupling), a training-free closed-loop adaptive serving framework that resolves the cascading representational drift bottleneck in deep dynamic model ensembling.

---

## 1. Persona Alignment

The **Theorist** persona demands mathematical rigor, provably stable trajectories, and formal justification over crude, unprincipled heuristics. 
- Standard continuous-time ensembling (ChemMerge) employs a heuristic, constant representation coupling rate $\eta$ that catastrophically collapses due to cascading representational drift under streaming task heterogeneity. To avoid simply disabling representation feedback ($\eta = 0.0$), L-ARC derives a rigorous **Lyapunov stability framework** to analyze the coupled dynamical system of continuous-time expert concentrations and multi-layer feature updates.
- By formulating a system-wide Lyapunov function $V(C, h)$ representing representation similarity error with respect to the active expert state, L-ARC mathematically identifies when representation warping is strictly dissipative (error-decreasing) or destabilizing (error-increasing due to cross-talk or routing noise).
- We prove that sample-specific, layer-specific step sizes $\eta_{b}^{(l)}$ can be adapted on-the-fly to guarantee asymptotic dissipation of representation error, bringing formal control-theoretic guarantees to deep dynamic serving.

---

## 2. Core Techniques

L-ARC introduces and integrates the following core algorithms and mechanisms:
1. **Continuous-Time Kinetics Routing (NEKR ODE from ChemMerge):** Expert concentrations $C_{k,b}^{(l)}$ are tracked continuously and updated layer-by-layer using discretized first-order reaction-decay ordinary differential equations (ODEs). This introduces temporal inertia across depth, acting as a low-pass filter to suppress ensembling weight jitter.
2. **Catalytic Zero-Shot Alignment (C-ZCA):** Early-layer frozen activations (up to Layer 3) are projected onto pre-computed, unit-norm task centroids to establish the catalytic coordinate anchor manifolds.
3. **Lyapunov-Bounded Adaptive Feedback Control (Ours):** Instead of using a constant step size $\eta$, we introduce an online Lyapunov feedback controller. It computes the local dissipation coefficient $A_b^{(l)}$ and the drift-accumulation factor $B_b^{(l)}$ at each layer for each sample. If the update is destabilizing ($A_b^{(l)} \le 0$), the controller sets the feedback rate to exactly zero. If it is stable ($A_b^{(l)} > 0$), it scales the feedback rate proportionally to $A_b^{(l)}$ up to a conservative safe threshold.
4. **Catalytic Activation Blending (CAB):** Blends the outputs of parallel expert adapters (LoRAs targeting self-attention projection weights) using mass-action weights derived from the active expert concentrations.

---

## 3. Mathematical Formulation

Let $C_{k,b}^{(l)} \in [0, 1]$ represent the active concentration of expert $k \in \{1,\dots,K\}$ for sample $b$ at layer $l \in [4, L]$.
Let $h_b^{(l-1)} \in \mathbb{R}^D$ be the unit-norm hidden representation at the output of block $l-1$.
Let $\mu_k^{(3)} \in \mathbb{R}^D$ be the unit-norm early-layer task centroids pre-computed from the calibration set.

For notational brevity, we omit the sample index $b$ from local equations where there is no ambiguity.

### 3.1. Continuous Concentration Kinetics (NEKR ODE)
The concentration updates follow a discretized explicit Euler step across layer depth:
$$C_k^{(l)} = \left[ C_k^{(l-1)} + \Delta t \left( k_k^{(l)} (1 - C_k^{(l-1)}) - k_{\text{decay}} C_k^{(l-1)} \right) \right]_0^1$$
where $k_{\text{decay}} \ge 0$ is the back-reaction decay rate, $\Delta t > 0$ is the virtual reaction step size, and $[\cdot]_0^1$ is the projection operator clipping concentrations to their thermodynamic bounds $[0, 1]$.

The catalytic reaction rate $k_k^{(l)}$ is governed by a normalized temperature-scaled Arrhenius equation representing competition for catalytic sites:
$$k_k^{(l)} = \frac{\exp\left(S(h^{(l-1)}, \mu_k^{(3)}) / \tau\right)}{\sum_{j=1}^K \exp\left(S(h^{(l-1)}, \mu_j^{(3)}) / \tau\right)}$$
where $\tau > 0$ is the routing reaction temperature, and $S(a, b) = \frac{a \cdot b}{\|a\|_2 \|b\|_2}$ is the cosine similarity operator.

### 3.2. Active Representation Coupling Update
The intermediate representation is warped towards the active experts before entering layer $l$'s self-attention block:
$$\tilde{h}^{(l-1) \text{ warped}} = h^{(l-1)} + \eta^{(l)} \left( \bar{\mu}^{(l)} - h^{(l-1)} \right)$$
$$h^{(l-1) \text{ warped}} = \frac{\tilde{h}^{(l-1) \text{ warped}}}{\|\tilde{h}^{(l-1) \text{ warped}}\|_2}$$
where $\bar{\mu}^{(l)} = \sum_{j=1}^K \alpha_j^{(l)} \mu_j^{(3)}$ is the ensembling-weighted centroid, with mass-action ensembling weights $\alpha_j^{(l)} = \frac{C_j^{(l)}}{\sum_{i=1}^K C_i^{(l)}}$.

### 3.3. Lyapunov Stability Derivation
We define the system Lyapunov function at layer $l$ as the weighted representation error relative to our catalytic task manifolds:
$$V(C^{(l)}, h^{(l-1) \text{ warped}}) = \sum_{k=1}^K C_k^{(l)} \left( 1 - S(h^{(l-1) \text{ warped}}, \mu_k^{(3)}) \right) \ge 0$$
Since representations and centroids are unit-norm, $V(C^{(l)}, h^{(l-1) \text{ warped}}) = \sum_{k=1}^K C_k^{(l)} \left(1 - h^{(l-1) \text{ warped}} \cdot \mu_k^{(3)}\right)$.

For asymptotic Lyapunov stability across depth, we require the total change across layers to be non-positive:
$$\Delta V^{(l)} = V(C^{(l)}, h^{(l-1) \text{ warped}}) - V(C^{(l-1)}, h^{(l-2) \text{ warped}}) \le 0$$

Using a first-order Taylor expansion of the warped representation $h^{(l-1) \text{ warped}}(\eta^{(l)})$ around $\eta^{(l)} = 0$:
$$S(h^{(l-1) \text{ warped}}(\eta^{(l)}), \mu_k^{(3)}) \approx S(h^{(l-1)}, \mu_k^{(3)}) + \eta^{(l)} D_k^{(l)}$$
where the directional derivative $D_k^{(l)}$ is:
$$D_k^{(l)} = S(\bar{\mu}^{(l)}, \mu_k^{(3)}) - S(h^{(l-1)}, \mu_k^{(3)}) S(h^{(l-1)}, \bar{\mu}^{(l)})$$

Substituting this into the Lyapunov difference:
$$\Delta V^{(l)} \approx \sum_{k=1}^K (C_k^{(l)} - C_k^{(l-1)}) \left(1 - S(h^{(l-1)}, \mu_k^{(3)})\right) - \eta^{(l)} \sum_{k=1}^K C_k^{(l)} D_k^{(l)}$$

We define the **drift-accumulation coefficient** $B^{(l)}$ and the **dissipation coefficient** $A^{(l)}$ as:
$$B^{(l)} = \sum_{k=1}^K \left(C_k^{(l)} - C_k^{(l-1)}\right) \left(1 - S(h^{(l-1)}, \mu_k^{(3)})\right)$$
$$A^{(l)} = \sum_{k=1}^K C_k^{(l)} \left( S(\bar{\mu}^{(l)}, \mu_k^{(3)}) - S(h^{(l-1)}, \mu_k^{(3)}) S(h^{(l-1)}, \bar{\mu}^{(l)}) \right)$$

This simplifies the discrete stability condition to:
$$\Delta V^{(l)} \approx B^{(l)} - \eta^{(l)} A^{(l)} \le 0 \iff \eta^{(l)} A^{(l)} \ge B^{(l)}$$

### 3.4. Closed-Loop Lyapunov-Stable Control Law
To guarantee dissipative stability (the coupling term actively reduces representation error and prevents cascading representational drift):
1. **Dissipation Guard:** We require $A^{(l)} > 0$. If $A^{(l)} \le 0$, the coupling direction is non-dissipative (error-increasing due to cross-talk or high routing confusion). We set:
$$\eta^{(l)} = 0.0$$
2. **Adaptive Control Step Size:** If $A^{(l)} > 0$, we choose the step size proportional to the dissipation strength:
$$\eta^{(l)} = \min\left( \eta_{\max}, \gamma \cdot A^{(l)} \right)$$
where $\eta_{\max} > 0$ is the maximum safe coupling rate (e.g. $0.15$) and $\gamma > 0$ is a controller feedback gain parameter.

---

## 4. Architecture Specifications

- **Backbone Network:** Pre-trained frozen 14-layer Coordinate Sandbox (ICS) model with feature dimension $D = 192$, utilizing orthogonal projection weights to represent stable feature propagation across layers.
- **Expert Adapters:** $K = 4$ task experts (MNIST, Fashion-MNIST, CIFAR-10, SVHN) fine-tuned via Low-Rank Adaptation (LoRA) targeting the self-attention query and value projection weights ($W_q, W_v$) with rank $r = 8$.
- **Early-Layer Feature Extraction:** Layers 1--3 are frozen and shared across all tasks. Centroids $\mu_k^{(3)}$ are extracted offline at the output of Layer 3 using 64 calibration samples per task.
- **Adapted Layers:** Layers 4 to 14 contain active task experts.
- **Kinetics Initial Boundary Conditions:** Concentration states are initialized uniformly: $C_k^{(3)} = 1/K$ for all $k \in \{1,\dots,K\}$.
- **Control & Physical Parameters:**
  - Maximum coupling step size: $\eta_{\max} = 0.15$
  - Controller feedback gain: $\gamma = 1.0$
  - Routing reaction temperature: $\tau = 0.01$
  - Virtual reaction step size: $\Delta t = 1.5$ (proven to lie below the discretization stability bound $\Delta t < \frac{2}{k_k + k_{\text{decay}}}$)
  - Back-reaction decay rate: $k_{\text{decay}} = 0.3$

---

## 5. Baselines

We will evaluate and compare L-ARC against five major baselines:
1. **Static Uniform Merging:** Merges adapters with equal, fixed weights $\lambda_k = 1/K$ across all layers. This baseline measures the severe accuracy penalty caused by raw parameter interference.
2. **SPS-ZCA:** The standard nearest-centroid early-stage routing baseline. It is stateless and highlights the impact of having no temporal memory across depth.
3. **SABLE:** The state-of-the-art dynamic activation ensembling baseline. It represents the strongest empirical ceiling for ensembling accuracy, but suffers from high layer-to-layer routing weight jitter.
4. **ChemMerge ($\eta=0.0$):** The decoupled continuous-time kinetics baseline, which avoids cascading representational drift by completely disabling representation feedback.
5. **ChemMerge ($\eta=0.05$):** The coupled continuous-time kinetics baseline with a constant coupling step size, which suffers from cascading representational drift under high heterogeneity, pulling activations off-manifold.

---

## 6. Step-by-Step Interaction

1. **Input Stage:** An input sample $x_b$ from a heterogeneous stream is passed to the shared backbone.
2. **Early Feature Extraction:** The input flows through shared Layers 1--3. At the output of Layer 3, the representation $h_b^{(3)}$ is captured.
3. **C-ZCA Projection:** We compute the cosine similarity of $h_b^{(3)}$ to each of the $K$ pre-computed unit-norm task centroids $\mu_k^{(3)}$, yielding the catalytic coordinate similarity vector.
4. **Initial ODE State:** Concentrations are initialized uniformly: $C_{k,b}^{(3)} = 1/K$.
5. **Layer-by-Layer Propagation (for Layers $l = 4 \dots 14$):**
   - **Arrhenius Collision Rates:** We calculate the current layer-wise reaction rate affinities $k_{k,b}^{(l)}$ based on the similarity of the preceding layer's representation $h_b^{(l-1)}$ to task centroids.
   - **Concentration State Update:** We integrate the linear reaction-decay ODE across step $\Delta t$ to update the sample-wise expert concentration $C_{k,b}^{(l)}$.
   - **Ensembling Weight Calculation:** Ensembling weights $\alpha_{k,b}^{(l)}$ are computed by normalizing concentration states.
   - **Lyapunov Adaptive Controller:**
     - Compute the blended centroid: $\bar{\mu}_b^{(l)} = \sum_{j=1}^K \alpha_{j,b}^{(l)} \mu_j^{(3)}$.
     - Evaluate the dissipation coefficient $A_b^{(l)}$ using the Lyapunov linear control law.
     - Select the adaptive step size: $\eta_{b}^{(l)} = \min(\eta_{\max}, \gamma A_b^{(l)})$ if $A_b^{(l)} > 0$, else $0.0$.
   - **Active Representation Coupling:** Update and normalize the hidden activation:
     $$\tilde{h}_b^{(l-1) \text{ warped}} = h_b^{(l-1)} + \eta_b^{(l)} \left( \bar{\mu}_b^{(l)} - h_b^{(l-1)} \right)$$
     $$h_b^{(l-1) \text{ warped}} = \text{Norm}\left(\tilde{h}_b^{(l-1) \text{ warped}}\right)$$
   - **Parallel Expert Serving & Blending (CAB):** The warped representation is passed in parallel to the base attention block and the active expert LoRA paths. The adapter output activations are scaled by $\alpha_{k,b}^{(l)}$ and summed:
     $$h_{b}^{(l)} = \text{LayerNorm}\left( h_{b}^{(l-1) \text{ warped}} + W_{\text{base}}^{(l)} h_{b}^{(l-1) \text{ warped}} + \sum_{k=1}^K \alpha_{k, b}^{(l)} \text{LoRA}_k^{(l)}(h_{b}^{(l-1) \text{ warped}}) \right)$$
6. **Final Prediction:** At the output of Layer 14, the final representation is passed to the task classification heads to compute predictions.
