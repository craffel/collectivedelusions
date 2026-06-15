# Idea Proposal: Lotka-Volterra Competitive Serving (LVCS)

## 1. Persona Alignment
Lotka-Volterra Competitive Serving (LVCS) completely embodies **The Visionary** persona. Instead of making incremental tweaks to existing linear, state-space serving architectures (like ChemMerge or PAC-Kinetics), LVCS completely rethinks the dynamic ensembling paradigm by drawing inspiration from **mathematical biology and ecology**. 

We reject the standard assumption that dynamic routers must be feedforward open-loop networks or simple linear-decay recurrences. Instead, we propose that the internal states of task-specific adapters behave like **species populations competing for limited resources in a localized ecosystem**. By modeling expert interaction through non-linear multi-species competition equations with carrying capacities, we construct an entirely new, self-regulating serving architecture from scratch. This radical perspective heavily emphasizes high novelty, poetic scientific connection, and transformative future potential over traditional linear, "safe" designs.

---

## 2. Core Techniques
LVCS introduces several highly novel, technically grounded mechanisms:
1.  **Discrete-Time Lotka-Volterra Ricker Recurrence:** We model the layer-by-layer ensembling state of the experts using the Ricker formulation of multi-species competition. The exponential nature of the Ricker model mathematically guarantees population positivity ($x_k > 0$) across all layers, completely bypassing the ad-hoc "clamping hacks" (e.g. hard clamping to $[0,1]$) required by continuous ODE solvers in ChemMerge.
2.  **Carrying Capacity and Self-Limitation Constraints:** Diagonal self-limitation coefficients act as ecological carrying capacities, stabilizing the ensembling trajectory and preventing population explosions under high activation noise.
3.  **Adaptive Niche Plasticity (Disturbance-Gated Competition):** To eliminate the representational lag (phase delay) that plagues stateful systems under rapid task switches, we introduce a temporal disturbance gate. This gate dynamically scales down the inter-species competition coefficients $c_{kj}$ ($k \neq j$) to zero when consecutive queries are orthogonal, representing a sudden "ecological disturbance" that allows the colonizer species (the new task expert) to establish dominance instantly without historical drag.
4.  **Unit-Norm PCA Coordinate Projection:** Following PAC-Kinetics, we extract bounded task coordinates from early-layer activations to serve as the local resource availability ($R_k$) driving species growth.

---

## 3. Mathematical Formulation
Let $f_\theta$ be a 14-layer pre-trained backbone, and let $\mathcal{E} = \{E_1, \dots, E_K\}$ be $K$ task-specific LoRA adapters with rank $r=8$.

### 3.1. Resource Extraction (Coordinate Projection)
At the designated routing layer $l_{\text{route}} = 3$, we extract the pooled intermediate activation vector $z_t \in \mathbb{R}^D$ and normalize it to the unit sphere:
$$\tilde{z}_t = \frac{z_t}{\|z_t\|_2 + \epsilon}$$
For each task $k$, we project $\tilde{z}_t$ onto its pre-calculated top-$d$ orthonormal PCA principal components matrix $V_{k, d} \in \mathbb{R}^{D \times d}$ to get the task resource coordinate:
$$R_{k, t} = e_{k, t} = \|V_{k, d}^T \tilde{z}_t\|_2 \in [0, 1]$$
This guarantees a strict, dimension-free coordinate bound $\|\mathbf{R}_t\|_\infty \le 1$.

### 3.2. Species Growth Rates
The growth rate $r_{k, t}$ of expert species $k$ is a linear function of its local resource coordinate:
$$r_{k, t} = w_k^{\text{grow}} R_{k, t} + b_k^{\text{grow}}$$
where $w_k^{\text{grow}} > 0$ represents the learned positive growth scaling factor (enforced via $w_k^{\text{grow}} = e^{s_k}$), and $b_k^{\text{grow}} \in \mathbb{R}$ is the learned growth bias.

### 3.3. Discrete Ecological Recurrence across Layers
Let $x_{k, t}^{(l)}$ be the virtual population state of expert $k$ at layer $l \in \{l_{\text{route}}+1, \dots, L\}$. The population density evolves across depth according to:
$$x_{k, t}^{(l)} = x_{k, t}^{(l-1)} \exp\left( r_{k, t} - \sum_{j=1}^K c_{kj, t} x_{j, t}^{(l-1)} \right)$$
where $c_{kj, t}$ are the elements of the dynamic competition matrix $C_t \in \mathbb{R}_+^{K \times K}$.
We initialize the starting ecosystem at layer $l_{\text{route}}$ with a uniform, balanced population:
$$x_{k, t}^{(l_{\text{route}})} = \frac{1}{K} \quad \forall k \in \{1, \dots, K\}$$

### 3.4. Parametric Constraints & Dynamic Competition
To preserve biological and mathematical validity during gradient-based optimization:
*   **Intra-Species Self-Limitation (Diagonal carrying capacity):**
    $$c_{kk, t} = c_{kk} = e^{u_k} + 0.1 \ge 0.1$$
    which represents the task's carrying capacity, guaranteeing a stable population ceiling.
*   **Inter-Species Niche Competition (Off-diagonal overlap):**
    $$c_{kj} = \sigma(v_{kj}) \in [0, 1] \quad (\text{for } k \neq j)$$
    which models the representational overlap/interference between tasks $k$ and $j$.
*   **Adaptive Niche Plasticity:**
    We measure stream homogeneity via consecutive coordinate cosine similarity:
    $$Sim_t = \frac{\mathbf{e}_t^T \mathbf{e}_{t-1}}{\|\mathbf{e}_t\|_2 \|\mathbf{e}_{t-1}\|_2 + \epsilon} \in [0, 1]$$
    The active inter-species competition is dynamically scaled:
    $$c_{kj, t} = c_{kj} \cdot Sim_t \quad (\text{for } k \neq j)$$

### 3.5. Simplex Mapping & Activation Blending
The active ensembling weights $\alpha_{k, t}^{(l)} \in \Delta^{K-1}$ are the normalized populations:
$$\alpha_{k, t}^{(l)} = \frac{x_{k, t}^{(l)}}{\sum_{j=1}^K x_{j, t}^{(l)}}$$
For all layers $l > l_{\text{route}}$, we perform single-pass activation blending in the forward pass:
$$h_t^{(l)} = h_t^{(l-1)} W_{\text{base}}^{(l)} + \sum_{k=1}^K \alpha_{k, t}^{(l)} \left( h_t^{(l-1)} A_k^{(l)} B_k^{(l)} \right)$$

---

## 4. Architecture Specifications
*   **Backbone dimensions:** $D = 192$, representing a 14-layer deep backbone network ($L = 14$).
*   **Expert settings:** $K = 4$ task-specific adapters (representing MNIST, Fashion-MNIST, CIFAR-10, and SVHN experts) with LoRA rank $r=8$.
*   **Routing layer:** $l_{\text{route}} = 3$. We extract activations at Layer 3, and apply the ecological recurrence across layers $l \in [4, 14]$.
*   **PCA dimensions:** We project activations onto the top $d=10$ principal components for each task.
*   **Trainable parameters:** $\Theta = \{\mathbf{s}, \mathbf{b}^{\text{grow}}, \mathbf{u}, V\} \in \mathbb{R}^{3K + K(K-1)}$, where:
    *   $\mathbf{s} \in \mathbb{R}^K$ (determines positive growth rates $w_k^{\text{grow}} = e^{s_k}$)
    *   $\mathbf{b}^{\text{grow}} \in \mathbb{R}^K$ (growth biases)
    *   $\mathbf{u} \in \mathbb{R}^K$ (determines diagonal carrying capacities $c_{kk} = e^{u_k} + 0.1$)
    *   $V \in \mathbb{R}^{K \times (K-1)}$ (determines off-diagonal competition $c_{kj} = \sigma(v_{kj})$)
*   **Prior Initialization $\Theta_0$ (Stable Grounded Ecosystem):**
    We center our Gaussian prior at stable default parameters:
    *   $s_{0, k} = \ln(1.0) = 0$ (unit growth scaling)
    *   $b_{0, k}^{\text{grow}} = 0$ (zero growth bias)
    *   $u_{0, k} = \ln(0.9) \approx -0.105$ (yielding $c_{kk} = 1.0$, a standard ecological carrying capacity)
    *   $v_{0, kj} = \sigma^{-1}(0.1) \approx -2.197$ (yielding $c_{kj} = 0.1$, representing a healthy ecosystem with minor niche overlap and high task separation).

---

## 5. Baselines
We will evaluate LVCS against five critical, representative baselines:
1.  **Uniform Merging (Static):** Static parameter addition. Sets the non-adaptive floor.
2.  **SABLE (Stateless Centroid Routing):** Samples ensembling weights sample-by-sample without stateful layers. Highly accurate but suffers from severe routing jitter.
3.  **ChemMerge (Continuous Stateful):** Continuous-time biochemical reaction kinetics governed by ODEs. Highly complex, prone to virtual-time numerical instability, and hyperparameter-sensitive.
4.  **Momentum-Merge (Constant Stateful):** Simple constant EMA weight smoothing across depth. Stable and fast, but completely non-adaptive and lacks learning capacity.
5.  **PAC-Kinetics (Linear Recurrent Stateful):** Learned stateful linear recurrence optimized via a PAC-Bayesian bound. Shows the upper bound of linear stateful routing.

---

## 6. Step-by-Step Interaction
1.  **Input Influx:** A query sample $\mathbf{x}_t$ is fed into the pre-trained backbone.
2.  **Early Processing:** Activations propagate through layers 1, 2, and 3.
3.  **Resource Extraction:** At Layer 3, we extract the activation vector $z_t \in \mathbb{R}^D$, normalize it to the unit sphere ($\tilde{z}_t$), and project it onto each of the $K$ orthonormal PCA coordinate matrices to compute the resource coordinates $R_{k, t} = e_{k, t}$.
4.  **Growth Activation:** The growth rate $r_{k, t}$ is calculated for each species using $w_k^{\text{grow}}$ and $b_k^{\text{grow}}$.
5.  **Disturbance Sensing:** We compute the cosine similarity $Sim_t$ between the current coordinates $\mathbf{e}_t$ and the previous coordinates $\mathbf{e}_{t-1}$.
6.  **Ecological Competition:** We construct the dynamic competition matrix $C_t$: the diagonal carrying capacities remain constant, while the inter-species competition terms are dynamically scaled down by $Sim_t$.
7.  **Population Recurrence:** Starting from the uniform population state $\mathbf{x}_t^{(3)} = \mathbf{1}/K$, we run the discrete Lotka-Volterra Ricker recurrence layer-by-layer for $l \in [4, 14]$. At each layer, species populations grow or decay based on their growth rate and the suppressive pressure exerted by competing species.
8.  **Simplex Normalization:** At each layer $l$, the positive population densities $\mathbf{x}_t^{(l)}$ are normalized to yield the ensembling weights $\alpha_{k, t}^{(l)}$.
9.  **Activation Blending:** We perform low-rank single-pass activation blending of the LoRA experts at each layer $l \in [4, 14]$ using $\boldsymbol{\alpha}_t^{(l)}$, propagating the blended activations to the final prediction head.
