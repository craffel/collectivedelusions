# Idea Proposal: HyperMerge

## 1. Persona Alignment
In direct alignment with **The Visionary** persona, **HyperMerge** rejects incremental Euclidean optimizations and hyperparameter tweaks. It represents a fundamental paradigm shift in modular deep learning by questioning and overturning the core geometric assumption of model merging: *that representations and task boundaries are best modeled in flat Euclidean space.*

Instead of viewing representation spaces as flat vector planes ($\mathbb{R}^D$), HyperMerge treats them as intrinsically hierarchical, multi-scale manifolds. By shifting the entire model ensembling and routing substrate from Euclidean geometry to **Hyperbolic geometry** (specifically the Poincaré Ball model $\mathbb{D}_c^D$), HyperMerge leverages negative curvature to represent nested task-relationships and power-law distribution behaviors with zero topological distortion. Shifting to hyperbolic space is a bold, high-risk, and mathematically elegant conceptual leap. By employing non-linear Möbius algebraic primitives rather than flat linear interpolations, HyperMerge represents a completely fresh perspective that resolves representation crowding and cross-talk fundamentally at the geometric level.

---

## 2. Core Techniques
HyperMerge introduces the following novel mechanisms and algebraic structures to model ensembling:

1. **Poincaré Ball Model of Hyperbolic Space ($\mathbb{D}_c^D$):** A Riemannian manifold with constant negative curvature $-c$, which serves as our geometric workspace.
2. **Hyperbolic Exponential and Logarithmic Mappings ($\exp_{\mathbf{0}}^c$, $\log_{\mathbf{0}}^c$):** Differentiable, closed-form mapping functions to project standard intermediate neural activations into and out of the Poincaré Ball at the origin.
3. **Hyperbolic Centroid Alignment (HCA):** Uses the Klein-coordinate transformation and the Einstein midpoint formulation to compute robust, non-parametric task-specific barycenters (Fréchet means) in hyperbolic space from a tiny calibration split.
4. **Möbius Activation Blending (MAB):** Replaces standard linear addition and scaling of expert activations with non-linear **Möbius addition ($\oplus_c$)** and **Möbius scalar multiplication ($\otimes_c$)**. This naturally isolates conflicting task trajectories towards the exponential volume boundaries of the hyperbolic space.
5. **Hyperbolic Cosine Distance ($d_{\mathbb{D}}^c$):** A robust metric for measuring distance-based routing similarity, naturally resolving representation crowding.
6. **Hyperbolic OOD Rejection (HOR):** Employs hyperbolic boundary distance thresholds to identify and reject out-of-distribution (OOD) queries with high precision.

*Foundational References:*
*   **Ganea et al., (2018):** "Hyperbolic Neural Networks" for formulating Möbius deep learning operations.
*   **Chami et al., (2019):** "Hyperbolic Graph Convolutional Networks" for low-distortion hierarchical embeddings.
*   **SPS-ZCA / SABLE (Trial 7 Submissions):** For the baseline single-pass dynamic ensembling frameworks.

---

## 3. Mathematical Formulation

### 3.1 Geometric Workspace
Let $\mathbb{D}_c^D = \{\mathbf{x} \in \mathbb{R}^D : \|\mathbf{x}\|_2 < 1 / \sqrt{c}\}$ be the Poincaré Ball model of hyperbolic space with curvature $c > 0$ and metric tensor $g_{\mathbf{x}} = \lambda_{\mathbf{x}}^2 g_{\text{Eucl}}$, where the conformal factor is $\lambda_{\mathbf{x}} = \frac{2}{1 - c\|\mathbf{x}\|_2^2}$.

### 3.2 Mapping Functions
To project Euclidean activations $h \in \mathbb{R}^D$ into the Poincaré Ball, we define the exponential map at the origin $\mathbf{0}$:
$$\mathbf{z} = \exp_{\mathbf{0}}^c(h) = \tanh(\sqrt{c} \|h\|_2) \frac{h}{\sqrt{c} \|h\|_2}$$

To project hyperbolic states $\mathbf{z} \in \mathbb{D}_c^D$ back to flat Euclidean space, we define the logarithmic map at the origin $\mathbf{0}$:
$$h = \log_{\mathbf{0}}^c(\mathbf{z}) = \text{artanh}(\sqrt{c} \|\mathbf{z}\|_2) \frac{\mathbf{z}}{\sqrt{c} \|\mathbf{z}\|_2}$$

### 3.3 Möbius Algebraic Primitives
For any points $\mathbf{x}, \mathbf{y} \in \mathbb{D}_c^D$, the Möbius addition is formulated as:
$$\mathbf{x} \oplus_c \mathbf{y} = \frac{(1 + 2c\langle\mathbf{x}, \mathbf{y}\rangle + c\|\mathbf{y}\|_2^2)\mathbf{x} + (1 - c\|\mathbf{x}\|_2^2)\mathbf{y}}{1 + 2c\langle\mathbf{x}, \mathbf{y}\rangle + c^2\|\mathbf{x}\|_2^2\|\mathbf{y}\|_2^2}$$

For any scalar $r \in \mathbb{R}$ and point $\mathbf{x} \in \mathbb{D}_c^D \setminus \{\mathbf{0}\}$, the Möbius scalar multiplication is formulated as:
$$r \otimes_c \mathbf{x} = \frac{1}{\sqrt{c}} \tanh\left( r \cdot \text{artanh}(\sqrt{c} \|\mathbf{x}\|_2) \right) \frac{\mathbf{x}}{\|\mathbf{x}\|_2}$$
with $r \otimes_c \mathbf{0} = \mathbf{0}$.

### 3.4 Hyperbolic Distance
The geodesic distance between any two points $\mathbf{x}, \mathbf{y} \in \mathbb{D}_c^D$ is given by:
$$d_{\mathbb{D}}^c(\mathbf{x}, \mathbf{y}) = \frac{2}{\sqrt{c}} \text{artanh}\left( \sqrt{c} \|\!-\mathbf{x} \oplus_c \mathbf{y}\|_2 \right)$$

### 3.5 Hyperbolic Centroid Alignment (HCA)
For task expert $k$, let $\mathcal{C}_k$ be the tiny calibration split of embedding representations. We project each $z_s \in \mathcal{C}_k$ to the Poincaré Ball: $\mathbf{z}_s = \exp_{\mathbf{0}}^c(z_s)$.
To compute the mathematically optimal hyperbolic centroid (the Fréchet mean), we transform the points to Klein coordinates:
$$\mathbf{z}_s^{Klein} = \frac{2 \mathbf{z}_s}{1 + c \|\mathbf{z}_s\|_2^2}$$

The Einstein midpoint in Klein space is a weighted average scaled by the Lorentz factor $\gamma_s$:
$$\boldsymbol{\mu}_k^{Klein} = \frac{\sum_{s \in \mathcal{C}_k} \gamma_s \mathbf{z}_s^{Klein}}{\sum_{s \in \mathcal{C}_k} \gamma_s}, \quad \text{where } \gamma_s = \frac{1}{\sqrt{1 - c \|\mathbf{z}_s^{Klein}\|_2^2}}$$

We project the centroid back to the Poincaré Ball coordinates to obtain the final task centroid reference $\boldsymbol{\mu}_k$:
$$\boldsymbol{\mu}_k = \frac{\boldsymbol{\mu}_k^{Klein}}{1 + \sqrt{1 - c \|\boldsymbol{\mu}_k^{Klein}\|_2^2}}$$

### 3.6 Hyperbolic Routing
For an online sample $b$ with Layer 0 embedding representation $z_b^{\text{embed}}$, we project it to hyperbolic space: $\mathbf{z}_b = \exp_{\mathbf{0}}^c(z_b^{\text{embed}})$.
The hyperbolic distance to task centroid $k$ is calculated as $D_{k, b} = d_{\mathbb{D}}^c(\mathbf{z}_b, \boldsymbol{\mu}_k)$.
We compute distance-based routing similarity coordinates $u_{k, b} = -D_{k, b}$.
The dynamic ensembling coefficients $\alpha_{k, b}$ are derived using temperature-scaled Softmax:
$$\alpha_{k, b} = \frac{\exp(u_{k, b} / \tau)}{\sum_{j=1}^K \exp(u_{j, b} / \tau)}$$

### 3.7 Möbius Activation Blending (MAB)
At layer $l$, the base model output is $h_{base, b}^{(l)} = h_b^{(l-1)} W_{\text{base}}^{(l)} \in \mathbb{R}^D$, and each task expert's low-rank adapter update is $E_{k, b}^{(l)} = h_b^{(l-1)} A_k^{(l)} B_k^{(l)} \in \mathbb{R}^D$.
We project each expert update to the Poincaré Ball:
$$\mathbf{v}_{k, b} = \exp_{\mathbf{0}}^c\left( E_{k, b}^{(l)} \right)$$

We scale each hyperbolic update by its routing weight using Möbius scalar multiplication:
$$\tilde{\mathbf{v}}_{k, b} = \alpha_{k, b} \otimes_c \mathbf{v}_{k, b}$$

We perform non-linear ensembling of the scaled updates using iterative Möbius addition:
$$\mathbf{v}_{\text{merged}, b} = \tilde{\mathbf{v}}_{1, b} \oplus_c \tilde{\mathbf{v}}_{2, b} \oplus_c \dots \oplus_c \tilde{\mathbf{v}}_{K, b}$$

We map the merged update back to flat Euclidean space using the logarithmic map:
$$E_{\text{merged}, b}^{(l)} = \log_{\mathbf{0}}^c\left( \mathbf{v}_{\text{merged}, b} \right)$$

The final activation state propagated to layer $l$ is:
$$h_b^{(l)} = h_{base, b}^{(l)} + E_{\text{merged}, b}^{(l)}$$

---

## 4. Architecture Specifications

The system parameters are designed to plug directly into the established Analytical Coordinate Sandbox and scale to physical deep neural networks:

*   **Geometric Parameters:**
    *   **Curvature Parameter ($c$):** Curvature is set to $c = 0.1$ (optimal for embedding power-law trees).
    *   **Dimension ($D$):** $D = 192$ (matching the coordinate sandbox dimensionality) or $D = 768$ (matching Vision Transformer ViT-B/16).
*   **Backbone & Experts Configuration:**
    *   **Shared Base Model ($W_{base}$):** Frozen 14-layer pre-trained network or ViT-B/16 backbone.
    *   **Expert Adapters ($E_k$):** $K = 4$ independent task experts (MNIST, F-MNIST, CIFAR-10, SVHN) decomposed into LoRA adapters of rank $r = 8$.
*   **Routing & OOD Specifications:**
    *   **Routing Layer:** Executed after the first Layer (Layer 0) of the network to resolve the Routing Paradox.
    *   **Routing Temperature ($\tau$):** Temperature is set to $\tau = 0.05$.
    *   **OOD Rejection Threshold ($\gamma_{\text{OOD}}$):** Rejection threshold set to $\gamma_{\text{OOD}} = 1.5$ in hyperbolic distance space.

---

## 5. Baselines
HyperMerge will be rigorously evaluated against the complete hierarchy of prior merging schemes:

1.  **Uniform Merging (Static):** The default parameter-space linear average of task vectors (Task Arithmetic). Serves as the static baseline.
2.  **PFSR + MBH (Dynamic, Euclidean):** Closed-form linear task-space projection combined with Micro-Batch Homogenization. This is the state-of-the-art systems-heavy baseline.
3.  **SABLE (Dynamic, Euclidean):** Sample-wise Euclidean activation-space blending using low-rank expert passes. This represents the state-of-the-art minimalist network-level Euclidean baseline.
4.  **SPS-ZCA (Dynamic, Euclidean):** Zero-Shot Centroid Alignment with Single-Pass Activation Blending. This is the top-performing baseline from Trial 7, achieving Joint Mean $79.80\%$.

---

## 6. Step-by-Step Interaction

An input batch flows through the HyperMerge system according to the following timeline:

```
                  Input Batch X = {x_1, ..., x_B}
                               |
                               v
                     Layer 0 Patch Embedding
                               |
                               v
             Euclidean Embeddings z_b_embed \in R^D
                               |
                      [Hyperbolic Mapping]
                               v
             Hyperbolic Embeddings z_b \in D_c^D
                               |
             [Hyperbolic Geodesic Distance d_D^c]
           against pre-computed HCA centroids \mu_k
                               |
                               v
                  Routing Coordinates u_{k, b}
                               |
                        [Softmax \tau]
                               v
                 Dynamic Routing Coefficients \alpha_{k, b}
                               |
              -----------------------------------
             |                                   |
             v                                   v
    [For Layers l = 1..L]              [Hyperbolic OOD Check]
     Run Base Pass:                     If min_k d_D^c > \gamma_OOD
      h_base_b = h_b_prev * W_base       -> Reject query as OOD task
     Run Parallel Adapter Passes:
      E_k_b = h_b_prev * A_k * B_k
             |
             v
   [Möbius Activation Blending]
    1. Project updates: v_k_b = exp_0(E_k_b)
    2. Scale updates: \tilde{v}_k_b = \alpha_k_b \otimes v_k_b
    3. Aggregate: v_merged = \tilde{v}_1 \oplus \tilde{v}_2 ...
    4. Euclidean project: E_merged = log_0(v_merged)
             |
             v
     Update State: h_b = h_base_b + E_merged
             |
             v
       To Next Layer
```

This ensures complete mathematical integrity, single-pass constant $O(1)$ backbone scaling, and zero-distortion ensembling.
