# Idea Proposal: OmniMerge (Multi-Schema Stochastic Co-Optimization for Robust Model Merging)

## 1. Persona Alignment
As **The Pragmatist**, my research is guided by real-world, practical edge-deployment utility. In actual deployment environments, hardware ASICs and runtime compilers (TPUs, DSPs, Apple Neural Engine, TensorRT) utilize highly heterogeneous, incompatible post-training quantization (PTQ) standards (ranging from symmetric per-channel to asymmetric per-tensor). 

Prior quantization-aware model merging methods (like Q-Merge) optimize coefficients strictly under a single simulated operator. The Robustness Audit (`trial3_submission1`) exposed that this triggers **catastrophic cross-operator overfitting**, where learned coefficients collapse to random-guess performance (~10%) when deployed on mismatched target hardware.

**OmniMerge** directly solves this bottleneck by optimizing merging coefficients under a **stochastically sampled mixture of quantization operators** during test-time calibration. 
- It is **calibration-robust** and handles hardware heterogeneity elegantly.
- It requires **zero extra hardware metadata**, is fully training-free, and adds **zero latency or memory overhead at inference time** because the final optimized coefficients are used to create a standard low-bit merged model.
- It is incredibly simple, reliable, and directly enables multi-hardware deployment from a single optimization sweep.

## 2. Core Techniques
OmniMerge introduces three main technical contributions:
1. **Stochastic Operator Sampling (SOS):** Instead of optimizing under a single, static quantization schema, OmniMerge stochastically selects an active quantization operator $Q^{(t)}$ from a discrete pool of hardware-relevant schemas at each optimization step. This acts as parameter-space data augmentation, preventing the coefficients from overfitting to any single set of rounding boundaries.
2. **Scale and Zero-Point Noise Perturbation (SZNP):** To enhance generalization to unseen target compilers, the scale factors $s^{(t)}$ and zero-point offsets $z^{(t)}$ computed dynamically for each layer are perturbed with multiplicative Gaussian noise, ensuring that the coefficients are resilient to slight boundary shifts.
3. **Straight-Through Estimator (STE) with Operator-Dropout:** Gradients are backpropagated using STE. The stochastic switching of schemas creates a form of "gradient dropout", smoothing the loss landscape and allowing the optimizer to find a flatter, more generalizable weight-space minimum.

## 3. Mathematical Formulation

### Unquantized Model Merging
Let $\theta^l_{\text{pre}} \in \mathbb{R}^D$ be the pre-trained weights and $\tau^l_k = \theta^l_k - \theta^l_{\text{pre}}$ be the task vector of expert $k \in \{1, \dots, K\}$ at layer $l \in \{1, \dots, L\}$. The unquantized merged weights at layer $l$ under coefficients $\Lambda \in [0, 1]^{K \times L}$ are:
\begin{equation}
    \theta^l_{\text{merged}}(\Lambda) = \theta^l_{\text{pre}} + \sum_{k=1}^K \lambda^l_k \tau^l_k
\end{equation}

### Stochastic Operator Pool
At step $t$ of adaptation, we stochastically sample a quantization schema $Q^{(t)}$ from a discrete pool $\mathcal{Q}$:
\begin{equation}
    Q^{(t)} \sim \mathcal{Q} = \{Q_{\text{sym, per-channel}}, Q_{\text{sym, per-tensor}}, Q_{\text{asym, per-channel}}, Q_{\text{asym, per-tensor}}\}
\end{equation}
with uniform probability $P(Q^{(t)} = Q_j) = 0.25$.

### Quantization Equations
For asymmetric quantization of a weight tensor $W = \theta^l_{\text{merged}}(\Lambda)$ at bit-width $b$:
\begin{align}
    s_{\text{asym}} &= \frac{\max(W) - \min(W)}{2^b - 1} \cdot (1 + \epsilon_s), \quad \epsilon_s \sim \mathcal{N}(0, \sigma^2_{\text{scale}}) \\
    z_{\text{asym}} &= \left\lfloor \frac{-\min(W)}{s_{\text{asym}}} \right\rceil - 2^{b-1} + \epsilon_z, \quad \epsilon_z \sim \mathcal{N}(0, \sigma^2_{\text{zero}}) \\
    Q_{\text{asym}}(W, b) &= s_{\text{asym}} \cdot \left( \left[ \left\lfloor \frac{W}{s_{\text{asym}}} \right\rceil + z_{\text{asym}} \right]_{-2^{b-1}}^{2^{b-1}-1} - z_{\text{asym}} \right)
\end{align}
For symmetric quantization ($z_{\text{sym}} = 0$):
\begin{align}
    s_{\text{sym}} &= \frac{\max(|W|)}{2^{b-1} - 1} \cdot (1 + \epsilon_s), \quad \epsilon_s \sim \mathcal{N}(0, \sigma^2_{\text{scale}}) \\
    Q_{\text{sym}}(W, b) &= s_{\text{sym}} \cdot \left[ \left\lfloor \frac{W}{s_{\text{sym}}} \right\rceil \right]_{-2^{b-1}+1}^{2^{b-1}-1}
\end{align}

### Unsupervised Joint Objective
We minimize prediction entropy over a tiny, unlabeled calibration stream $D_{\text{cal}} = \{X_{i, k}\}_{i=1}^N$:
\begin{equation}
    \mathcal{L}_{\text{entropy}}(\Lambda) = -\frac{1}{N \cdot K} \sum_{k=1}^K \sum_{i=1}^N \sum_{c=1}^C p_{k, i}(c \mid \Lambda) \log p_{k, i}(c \mid \Lambda)
\end{equation}
where $p_{k, i}(c \mid \Lambda)$ are predicted softmax probabilities computed using the stochastically quantized model $\theta_{\text{quant}, t}(\Lambda) = Q^{(t)}\left(\theta_{\text{merged}}(\Lambda), b\right)$.

To regulate parameter drift, we apply Elastic Spatial Regularization (ESR):
\begin{equation}
    \mathcal{R}_{\text{spatial}}(\Lambda) = \frac{\beta}{K L} \sum_{k=1}^K \sum_{l=1}^L (\lambda^l_k - \lambda_{\text{init}})^2 + \frac{\gamma}{K L} \sum_{k=1}^K \sum_{l=1}^L (\lambda^l_k - \bar{\lambda}_k)^2
\end{equation}
where $\beta = 0.1$, $\gamma = 0.5$, and $\lambda_{\text{init}} = 0.3$.
The total objective is:
\begin{equation}
    \mathcal{L}_{\text{total}}(\Lambda) = \mathcal{L}_{\text{entropy}}(\Lambda) + \mathcal{R}_{\text{spatial}}(\Lambda)
\end{equation}

### Straight-Through Optimization
Using the Straight-Through Estimator (STE) $\frac{\partial Q^{(t)}(W, b)}{\partial W} \approx 1$, the coefficients are updated via:
\begin{equation}
    \Lambda^{(t+1)} = \text{clamp}\left( \Lambda^{(t)} - \eta \cdot \text{Adam}\left( \nabla_{\Lambda} \mathcal{L}_{\text{total}}(\Lambda^{(t)}) \right), [0, 1] \right)
\end{equation}

## 4. Architecture Specifications
- **Model Backbone:** `timm` Vision Transformer (`ViT-Tiny` with patch size 16, embedding dimension 192, 12 attention heads, 12 layers).
- **Target Quantization Precision:** $b = 4$ (aggressive) and $b = 8$ (standard).
- **Optimization Parameters:**
  - Coefficient matrix $\Lambda \in [0, 1]^{4 \times 12}$ (for 4 classification tasks, 12 layers).
  - Initialization: $\lambda_{\text{init}} = 0.3$ (uniform).
  - Learning rate $\eta = 10^{-3}$, Adam optimizer.
  - Scale perturbation noise: $\sigma^2_{\text{scale}} = 0.01$.
  - Zero-point perturbation noise: $\sigma^2_{\text{zero}} = 0.05$.
- **Calibration stream:** $N = 64$ unlabeled images per task.

## 5. Baselines
We will evaluate OmniMerge against:
1. **Unquantized Uniform FP16 Baseline:** Model Soup / Task Arithmetic without post-training quantization.
2. **Naive Merge-then-Quantize (M-then-Q):** Uniform Task Arithmetic coefficients followed by post-hoc target quantization.
3. **Q-Merge (STE under Source Operator):** Merging coefficients optimized strictly under a single source schema (e.g., Symmetric Per-Channel) and evaluated on target schemas, illustrating the cross-schema generalization gap.
4. **Quantized AdaMerging:** Coefficient optimization in full-precision (FP16) followed by post-hoc target quantization.
5. **Zero-Order 1+1 ES Comparator:** Stochastic random-walk optimization directly under the source schema (illustrating extreme boundary overfitting).

## 6. Step-by-Step Interaction
1. **Calibration Stream Input:** Unlabeled calibration images from the 4 classification tasks (MNIST, FashionMNIST, CIFAR-10, SVHN) are batched.
2. **Weight Blending:** At each step $t$, the merging coefficients $\Lambda^{(t)}$ are used to construct the dynamic unquantized merged weight tensor for each layer:
   $$W^l = \theta^l_{\text{pre}} + \sum_{k=1}^K \lambda^l_k \tau^l_k$$
3. **Stochastic Quantization Operator Selection:** For each layer $l$, an operator $Q^{(t)}$ is stochastically sampled from $\mathcal{Q}$. The weights are quantized and dequantized under this operator:
   $$W^l_{\text{quant}} = Q^{(t)}(W^l, b)$$
4. **Forward Evaluation:** Calibration batches are passed through the model with weights $\{W^l_{\text{quant}}\}$ to compute class probability distributions.
5. **Loss & Regularization Computation:** Shannon entropy is computed over the batch, combined with the proximity and smoothing penalties of ESR, yielding the joint scalar loss $\mathcal{L}_{\text{total}}$.
6. **STE Backpropagation:** Autograd propagates gradients through the non-differentiable rounding operator using STE.
7. **Coefficient Update:** The Adam optimizer updates the continuous coefficients $\Lambda^{(t)}$ based on the gradients, and the updated coefficients are clamped to $[0, 1]$.
8. **Mismatched Target Evaluation:** After optimization converges, the final unquantized merged model (using the optimized coefficients) is compiled and quantized under the unseen *target* hardware schemas (e.g. Symmetric Per-Tensor or Double Quantization) to verify cross-operator robustness.
