# 1. Summary of the Paper

## Core Motivation and Visionary Hook
The core motivation of this paper is to address the limitations of static model merging (such as Task Arithmetic), which applies frozen, uniform blending coefficients across all layers and inputs, leading to severe representational interference and parameter collision.
To overcome this, the paper introduces a highly unconventional, bold, and conceptually refreshing approach: **ChaosMerge** (Chaos-Theoretic Attractor Merging). Instead of treating a neural network's layers as a flat feed-forward sequence of computations, the paper frames layer depth as the temporal progression of a non-linear chaotic dynamical system. Specifically, it models the layer-wise merging coefficients as the trajectory of a Coupled Map Lattice (CML) driven by a chaotic Logistic Map.

---

## Detailed Technical Pipeline

### 1. Sphere-Projected Feature Extraction
Input representations are projected onto a low-dimensional unit sphere to act as scale-invariant, bounded physical perturbations that steer the chaotic system:
* **Feature Extraction:** Spatially averaged patch tokens $z(x)_j \in \mathbb{R}^D$ are extracted from the base model's frozen patch embedding layer for task $j$.
* **Low-Dimensional Projection:** Projected onto $d$-dimensions (where $d = K = 4$ tasks) using a frozen random projection matrix $P \in \mathbb{R}^{D \times d}$:
  $$\tilde{\psi}(x)_j = z(x)_j P \in \mathbb{R}^d$$
* **Unit Sphere Mapping:** Normalized to the unit sphere $\mathbb{S}^{d-1}$ with a numerical stabilizer $\epsilon = 10^{-8}$:
  $$\psi(x)_j = \frac{\tilde{\psi}(x)_j}{\|\tilde{\psi}(x)_j\|_2 + \epsilon}$$

### 2. Lattice State Initialization
At temporal step $l = 0$, the lattice states $s_j^{(0)} \in [0, 1]^K$ are initialized via a learned linear projection of the projected phase state, passed through a Sigmoid activation to ensure they are strictly bounded:
$$s_{k, j}^{(0)} = \sigma\left( \sum_{m=1}^d W_{init, k, m} \psi(x)_{j, m} + b_{init, k} \right)$$
where $W_{init} \in \mathbb{R}^{K \times d}$ and $b_{init} \in \mathbb{R}^K$ are trainable initializers.

### 3. Gated Coupled Map Lattice (G-CML)
For each subsequent layer group $l \in \{1, \dots, L\}$:
* **Spatio-Temporal Chaotic Coupling:**
  $$\bar{s}_{k, j}^{(l)} = (1 - \gamma_l) f\left(s_{k, j}^{(l-1)}\right) + \frac{\gamma_l}{K} \sum_{i=1}^K f\left(s_{i, j}^{(l-1)}\right)$$
  where $f(u) = 4u(1-u)$ is the fully chaotic Logistic Map, and $\gamma_l = \sigma(\gamma_{raw, l}) \in [0, 1]$ is a learned layer-wise spatial coupling coefficient representing localized diffusion.
* **Logit-Space Perturbation and Steering:**
  $$s_{cand, k, j}^{(l)} = \sigma\left( \sigma^{-1}\left( \text{clip}\left(\bar{s}_{k, j}^{(l)}, \delta, 1-\delta\right) \right) + \langle \psi(x)_j, \Phi_k^{(l)} \rangle + \phi_k^{(l)} \right)$$
  where $\delta = 10^{-5}$, and $\Phi_k^{(l)} \in \mathbb{R}^d$ and $\phi_k^{(l)} \in \mathbb{R}$ are learned projection keys and biases defining attractor basin orientation.
* **Gated State Update (Skip Connection):**
  $$s_{k, j}^{(l)} = (1 - \lambda_l) s_{k, j}^{(l-1)} + \lambda_l s_{cand, k, j}^{(l)}$$
  where $\lambda_l = \sigma(\lambda_{raw, l}) \in [0, 1]$ is a learned gating coefficient acting as a residual skip connection to tame gradient explosion.

### 4. Task-Specific Dynamic Routing & Weight Assembly
Rather than average weights across a heterogeneous batch (which washes out the trajectory sensitivity), the active lattice states are mapped directly to task-specific coefficients:
$$\alpha_{k, j}(l) = R_k^{(l)} \cdot s_{k, j}^{(l)}$$
where $R_k^{(l)} \in \mathbb{R}$ is a learned layer-wise scaling amplitude.
The final merged expert weights are:
$$W_{merged, j}^{(l)} = W_{base}^{(l)} + \sum_{k=1}^K \alpha_{k, j}(l) V_k^{(l)}$$

To make this practical, the authors propose using **Task-Level Centroids** calculated offline or on-the-fly, allowing a single weight assembly step per task/batch, thereby avoiding sample-by-sample swapping latency during inference.

---

## Experimental Protocol & Results
* **Backbone:** Vision Transformer (`vit_tiny_patch16_224`, 5.7M parameters) with $L=14$ layer groups.
* **Benchmark:** Visual classification across four tasks: MNIST, FashionMNIST, CIFAR-10, SVHN ($K=4$).
* **Data Splits:** Fine-tuning on 2,000 samples/task; dynamic coefficients calibrated on a tiny 64 samples/task; evaluation on 500 samples/task.
* **Trainable Parameter Footprint:** Exactly 384 parameters.
* **Key Findings:** G-CML boosts average merging accuracy by **+18.60%** absolute over the original, ungated chaotic model (which collapsed due to gradient explosion). Under symmetric task-specific routing, G-CML achieves **73.80%** average accuracy, outperforming Task Arithmetic (54.75%) and AdaMerging (70.85%).
