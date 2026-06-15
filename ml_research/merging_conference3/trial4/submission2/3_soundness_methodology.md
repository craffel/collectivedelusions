# 3. Soundness and Methodology Check

## Assessment of Technical Soundness
The technical soundness of the paper's methodology is **fair to good**. The mathematical formulation is structured cleanly, and the proposed techniques address the target problem directly. However, there are several subtle mathematical ambiguities, potential flaws in the gradient flow, and logical contradictions in the regularization design.

---

## Detailed Methodological Analysis

### 1. Scale and Zero-Point Noise Perturbation (SZNP) Formulation
The equations for asymmetric quantization (Equations 5 and 6) are formulated as:
$$s_{\text{asym}} = \frac{\max(W) - \min(W)}{2^b - 1} \cdot (1 + \epsilon_s)$$
$$z_{\text{asym}} = \left\lfloor \frac{-\min(W)}{s_{\text{asym}}} \right\rceil - 2^{b-1} + \epsilon_z$$
where $\epsilon_s \sim \mathcal{N}(0, \sigma^2_{\text{scale}})$ and $\epsilon_z \sim \mathcal{N}(0, \sigma^2_{\text{zero}})$.

#### Mathematical Issues:
- **Continuous Zero-Point Noise during Calibration:** 
  Adding continuous Gaussian noise $\epsilon_z$ to $z_{\text{asym}}$ makes the zero-point continuous during calibration. In standard PTQ, the zero-point $z$ must be a strict integer so that the quantized integers can be processed purely with integer math (avoiding floating-point accumulators). While the authors clarify that $\epsilon_s = 0, \epsilon_z = 0$ during deployment, injecting continuous noise during calibration creates a **training-test mismatch**. The optimizer finds merging coefficients $\Lambda$ that are optimal for *fractional* zero-points, but when deployed, these zero-points are rounded to integers. A mathematically more rigorous approach would be to round $z_{\text{asym}} + \epsilon_z$ to the nearest integer or use discrete uniform noise to ensure the model optimizes under valid integer grids even during calibration.
- **Infinite Support of Gaussian Noise:**
  Scale factor $s$ must be strictly positive. Since Gaussian noise $\epsilon_s$ has infinite support, $1 + \epsilon_s$ can theoretically be negative or zero (leading to division-by-zero errors in Equation 7). Although this has extremely low probability for $\sigma_{\text{scale}} = 0.01$, using a log-normal distribution or a truncated normal distribution for the scale perturbation would be mathematically more sound.

---

### 2. Autograd Graph & Gradient Flow through Scale/Zero-Point
The paper relies on the Straight-Through Estimator (STE) to propagate gradients through the rounding operation:
$$\frac{\partial Q^{(t)}(W, b)}{\partial W} \approx 1$$

#### The Missing Detail (Detaching Scale/Zero-Point):
- In PyTorch, if $s$ and $z$ are computed dynamically from $W$ (which depends on the optimized coefficients $\Lambda$), the autograd graph will propagate gradients through two paths:
  1. The direct weight division path: $\frac{W}{s}$.
  2. The scale factor $s(W)$ and zero-point $z(W)$ paths, which involve non-differentiable $\max(W)$ and $\min(W)$ operations.
- The derivative of $\max(W)$ or $\min(W)$ with respect to $W$ is a Dirac delta function (non-zero only at the argmax/argmin coordinate). If these paths are not detached, the gradients will be extremely sparse and noisy, leading to optimization instability.
- In standard PTQ literature, the scale and zero-point tensors are **detached** from the autograd graph during backpropagation (i.e., treated as constant parameters at each step). 
- The paper is completely silent on whether they detached $s$ and $z$ from the autograd graph. If they did not detach them (which PyTorch's default behavior does if not explicitly told otherwise), the backpropagation contains highly unstable subgradients through min/max, which could explain the optimization oscillations they observed at larger learning rates. This is a critical omission for reproducibility and mathematical rigor.

---

### 3. Task-Consensus Regularization (TCR) Contradiction
The TCR penalty is formulated as:
$$\mathcal{R}_{\text{con}}(\Lambda) = \frac{1}{K L} \sum_{k, l} \left[ \beta (\lambda^l_k - \lambda_{\text{init}})^2 + \gamma (\lambda^l_k - \bar{\lambda}^l)^2 \right]$$
where $\bar{\lambda}^l = \frac{1}{K} \sum_{k=1}^K \lambda^l_k$ is the average task blending coefficient at layer $l$, and $\gamma = 0.5$ penalizes task-specific deviation from this average.

#### Logical Contradiction:
- The fundamental premise of optimizing task-specific layer-wise merging coefficients $\Lambda \in [0, 1]^{K \times L}$ is to allow the model to assign *different* weights to different tasks at different layers depending on their parameter sensitivity.
- However, by setting $\gamma = 0.5$ (which is 5x larger than the proximity penalty $\beta = 0.1$), the TCR heavily penalizes any deviation of a task's coefficient from the group average $\bar{\lambda}^l$.
- This forces all task-specific coefficients at layer $l$ to be nearly identical: $\lambda^l_1 \approx \lambda^l_2 \approx \lambda^l_3 \approx \lambda^l_4 \approx \bar{\lambda}^l$.
- If the coefficients are forced to be almost identical across tasks, the framework is essentially performing **layer-wise uniform scaling** of the task vectors rather than true task-specific blending. If so, why optimize $K \times L$ coefficients instead of just $L$ layer-wise scaling factors? The strong consensus penalty contradicts the stated goal of multi-task weight blending and severely limits the expressivity of the ensembling.

---

### 4. Per-Channel Double Quantization on ViT-Tiny
The paper evaluates Double Quantization (DQ) and applies it **per-channel**:
"Double quantization is applied per-channel to maintain high representation capacity."

#### Logical Flaw:
- Double Quantization (DQ) is designed to compress the scale factors of quantized tensors to save memory. 
- However, DQ is only practically useful for fine-grained **block-wise or group-wise** quantization (e.g., block size 32 or 64), where scale factors are stored for every small group of weights, creating substantial scale overhead.
- For **per-channel** quantization, there is only one scale factor per output channel. In `ViT-Tiny`'s projection layers, the channel dimension is 192. Storing 192 FP16 scale factors takes just 384 bytes, while storing the weights takes 36,864 bytes. The scale overhead is **$< 1\%$** of the model size!
- Compressing these 192 scales to 8-bit integers saves only 192 bytes ($< 0.5\%$ of the layer size) while introducing significant compiler dequantization overhead on hardware.
- Applying Double Quantization per-channel is practically nonsensical, as the storage savings are completely negligible and do not justify the added complexity. This contradicts the authors' stated "Pragmatist" persona which claims to focus on real-world edge-deployment utility.
