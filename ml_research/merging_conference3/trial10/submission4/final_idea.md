# Idea Proposal: QA-Merge (Quantization-Robust Centroid Routing for Low-Precision Edge Serving)

## 1. Persona Alignment
As **The Pragmatist**, my research values are rooted in real-world feasibility, deployment constraints, and production-level efficiency. In modern deep learning serving, deploying model weights and activations in full 32-bit floating-point precision is computationally prohibitive, especially on edge hardware. Industry standards mandate model quantization (such as INT8 or INT4) to reduce memory bandwidth and latency. 

However, existing dynamic model ensembling methods (like SABLE, ChemMerge, and PAC-Kinetics) are designed under the assumption of high-precision floating-point coordinate representations. Under extreme quantization (e.g., INT8 activations and INT4 ensembling weights), rounding noise collapses representation manifolds, leading to overlapping centroids, highly unstable routing gating, and severe downstream classification degradation. 

**QA-Merge** directly targets this critical real-world bottleneck. Rather than proposing another complex physical or theoretical metaphor, QA-Merge introduces practical, mathematically robust mechanisms—Quantized Centroid Calibration (QCC), Straight-Through Estimator (STE) parametric gating, and Error-Feedback Trajectory Stabilization (EF-Smooth)—to enable highly stable and accurate activation-space model merging directly within quantized, low-precision pipelines. This is the epitome of pragmatic research: making dynamic ensembling actually deployable on cheap edge hardware.

---

## 2. Core Techniques
QA-Merge introduces three core, mutually reinforcing techniques to survive low-precision constraints:

1. **Quantized Centroid Calibration (QCC):** Standard SABLE uses float32 centroids. Under quantization, these centroids shift and overlap. QCC computes and calibrates the task-specific centroids directly in the target quantized integer representation space, applying scale-and-bias optimization to maximize centroid separation under discrete rounding.
2. **Straight-Through Estimator (STE) Gating Optimization:** To optimize gating parameters (like temperatures $\tau$ or gating weights $W_g$) under quantized coordinate activations, we employ the Straight-Through Estimator (STE) \cite{bengio2013estimating} to bypass the non-differentiable rounding operator during few-shot offline calibration, preventing gradient vanishing and ensuring convergence.
3. **Error-Feedback Trajectory Stabilization (EF-Smooth):** When ensembling coefficients $\boldsymbol{\alpha}$ are quantized to low-precision discrete steps, rounding errors accumulate across deep networks, leading to representational drift. Inspired by delta-sigma modulation in signal processing, EF-Smooth tracks the rounding error of ensembling coefficients at layer $l-1$ and injects it into layer $l$ as a high-pass correction, stabilizing ensembling trajectories with near-zero compute overhead.

---

## 3. Mathematical Formulation

### A. Quantization Operator
We model symmetric uniform quantization of activations/representations and ensembling weights. Let $x \in \mathbb{R}$ represent a continuous value, $s > 0$ represent the scaling factor, and $b$ represent the bit-width (e.g., $b=8$ for INT8, $b=4$ for INT4). The quantization operator $Q(x, s, b)$ is defined as:
$$Q(x, s, b) = \text{clip}\left( \left\lfloor \frac{x}{s} \right\rceil, -2^{b-1}, 2^{b-1}-1 \right)$$
where $\lfloor \cdot \rceil$ represents the nearest-integer rounding function, and the de-quantized value is $\tilde{x} = Q(x, s, b) \times s$.

### B. Quantized Centroid Calibration (QCC)
Let $\mathcal{D}_k$ represent the few-shot calibration set for task $k$. The calibrated quantized centroid $c'_k \in \mathbb{Z}^D$ is computed directly on the quantized early feature activations $h_i^{(3)} \in \mathbb{R}^D$ extracted at Layer 3:
$$c'_k = \left\lfloor \frac{1}{|\mathcal{D}_k|} \sum_{i \in \mathcal{D}_k} Q(h_i^{(3)}, s_{act}, 8) \right\rceil$$

### C. Quantization-Aware Gating with STE
During offline optimization, the gating logit for expert $k$ under sample $b$ is computed using quantized features:
$$z_{k, b} = \mathbf{w}_k^T Q(h_b^{(3)}, s_{act}, 8) + b_{g, k}$$
In the backward pass, the gradient of the rounding function is approximated as an identity mapping:
$$\frac{\partial \tilde{h}}{\partial h} \approx 1$$
This allows us to update $\{W_g, b_g\}$ via standard backpropagation on the quantized representation space.

### D. Error-Feedback Trajectory Stabilization (EF-Smooth)
Let $\boldsymbol{\alpha}_b^{(l)} \in [0, 1]^K$ represent the continuous routing weight vector computed at layer $l$, and $\tilde{\boldsymbol{\alpha}}_b^{(l)}$ represent its quantized, low-precision counterpart (quantized to 4-bit, giving 16 discrete blending levels per expert).
The quantization error vector $\mathbf{e}_b^{(l)}$ is defined as:
$$\mathbf{e}_b^{(l)} = \boldsymbol{\alpha}_{b, \text{corrected}}^{(l)} - \tilde{\boldsymbol{\alpha}}_b^{(l)}$$
For the subsequent layer $l+1$, we add the error feedback discounted by a decay factor $\beta \in [0, 1]$:
$$\boldsymbol{\alpha}_{b, \text{corrected}}^{(l+1)} = \boldsymbol{\alpha}_b^{(l+1)} + \beta \mathbf{e}_b^{(l)}$$
$$\tilde{\boldsymbol{\alpha}}_b^{(l+1)} = \text{ProjectToSimplex}\left( Q(\boldsymbol{\alpha}_{b, \text{corrected}}^{(l+1)}, s_{\alpha}, 4) \right)$$
where $\text{ProjectToSimplex}$ ensures the quantized blending weights are normalized and sum to 1. This error diffusion keeps the average representational blending path extremely close to the optimal continuous trajectory.

---

## 4. Architecture Specifications
We instantiate QA-Merge inside the 14-layer Coordinate Sandbox (ICS):
- **Hidden Dimension:** $D = 192$, depth $L = 14$ layers.
- **Frozen Boundary:** Feature extraction and centroid evaluation occur at Layer 3.
- **Quantization Precision:**
  - **Activations/Representations $h^{(l)}$:** Quantized to INT8 (8-bit, symmetric range $[-128, 127]$) dynamically layer-by-layer.
  - **Ensembling Coefficients $\boldsymbol{\alpha}$:** Quantized to 4-bit (INT4, symmetric range $[-8, 7]$) to represent 16 discrete levels of adapter contribution per expert.
- **Dynamic Blending Dynamics:**
  For layers $l \in [4, 14]$, representations are propagated and blended using low-precision integer-arithmetic operations:
  $$h_b^{(l)} = h_b^{(l-1)} + \sum_{k=1}^K \tilde{\alpha}_{k, b}^{(l)} \gamma_V (v'_k - h_b^{(l-1)})$$
  where $\gamma_V = 0.05$ is the constant adapter scaling factor.

---

## 5. Baselines
To demonstrate the practical superiority of QA-Merge, we will evaluate it against the following baselines across both float32 and quantized (INT8/INT4) settings:
1. **Standard SABLE (Float32):** The original, unquantized stateless ensembling method.
2. **Quantized-Naive SABLE (INT8 / INT4):** Standard SABLE evaluated directly on quantized activations without QCC or EF-Smooth (exposes the quantization collapse bottleneck).
3. **Standard ChemMerge (Float32):** The unquantized, stateful kinetics ensembling baseline.
4. **Quantized-Naive ChemMerge (INT8 / INT4):** Stateful chemical kinetics executed directly with quantized representations and weights.
5. **Zero-Init Parametric Softmax Router (Float32 vs. Quantized):** The classical linear router optimized under unregularized, regularized, and quantization-aware conditions.
6. **Momentum-Merge (Float32 vs. Quantized):** The unquantized and quantized-naive constant EMA weight smoothing baseline.

---

## 6. Step-by-Step Interaction
1. **Feature Extraction & Quantization:** At Layer 3, the continuous activation vector $h_b^{(3)}$ is extracted and quantized to INT8 using $Q(h_b^{(3)}, s_{act}, 8)$.
2. **Quantized Distance Gating:** The quantized feature vector is compared against the pre-calibrated quantized centroids $c'_k$ to compute the L2 distance logit:
   $$d_{k, b} = - \| Q(h_b^{(3)}, s_{act}, 8) - c'_k \|_2^2$$
3. **Routing Coefficient Generation:** Distance logits are scaled by temperature $\tau$ and mapped via Softmax to produce continuous ensembling weights $\boldsymbol{\alpha}_b^{(4)}$.
4. **Error Diffusion & Coefficient Quantization:** At layer $l$ (starting with $l=4$), the continuous ensembling weights are corrected with the discounted previous error $\beta \mathbf{e}_b^{(l-1)}$ to form $\boldsymbol{\alpha}_{b, \text{corrected}}^{(l)}$. This is quantized to 4-bit precision to yield $\tilde{\boldsymbol{\alpha}}_b^{(l)}$, and the new quantization error $\mathbf{e}_b^{(l)}$ is updated and stored.
5. **Quantized Activation Blending:** Low-precision activation ensembling is computed via integer matrix multiplication:
   $$h_b^{(l)} = h_b^{(l-1)} + \sum_{k=1}^K \tilde{\alpha}_{k, b}^{(l)} \gamma_V (v'_k - h_b^{(l-1)})$$
6. **Iterative Layer Propagation:** Steps 2-5 are repeated for all dynamic ensembling layers $l \in [4, 14]$.
7. **Unified Low-Precision Distance Classifier:** The final quantized representation $h_b^{(14)}$ is classified using the negative Euclidean distance classifier.
