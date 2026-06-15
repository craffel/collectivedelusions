# Q-PolyMerge: Quantization-Aware Continuous Polynomial Subspace Model Merging for Extreme Edge Efficiency

## 1. Persona Alignment
Q-PolyMerge directly aligns with the core principles of **The Pragmatist** persona. In real-world enterprise deployments, deep learning models are strictly bound by hardware storage, memory bandwidth, and inference latency constraints. To satisfy these budgets, post-training quantization (PTQ) to low-bit formats (such as 8-bit INT8 or aggressive 4-bit INT4) is an absolute operational necessity. 

However, practitioners face a challenging dilemma: merging full-precision expert models followed by post-hoc quantization introduces severe quantization noise that degrades multi-task accuracy, whereas merging pre-quantized experts completely breaks linear mode connectivity. 

While unconstrained test-time adaptation (TTA) frameworks (like AdaMerging or standard Q-Merge) attempt to optimize coefficients to align layers under quantization, they suffer from the **Overfitting-Optimizer Paradox**. Because test-time calibration streams are small and unlabeled, unconstrained optimizers (optimizing $L \times K$ parameters) easily fit transductive statistical noise, resulting in highly jagged layer-wise coefficient profiles that collapse on held-out or out-of-distribution test data. 

Q-PolyMerge resolves this bottleneck by projecting the continuous merging coefficients onto a low-dimensional, smooth polynomial subspace of normalized layer depth. By reducing the learnable dimensions to just $(d+1) \times K$ parameters, we mathematically filter out high-frequency optimization noise and prevent degenerate entropy collapse states. Q-PolyMerge requires zero training data or parameters during inference, serving as a robust, highly stable, and easily integratable compression and alignment pipeline for edge deployment.

## 2. Core Techniques
The Q-PolyMerge framework integrates four core technical pillars to enable robust, low-bit multi-task merging:

1. **Quantization-Aware Model Merging (Q-Merge under PTQ):** Optimizing merging coefficients directly under the non-differentiable quantization operator. We evaluate two optimization paradigms:
   - **First-Order Adam Gradient Descent with a Straight-Through Estimator (STE):** Bypasses the non-differentiable rounding operation by copying gradients directly during backpropagation.
   - **Zero-Order 1+1 Evolution Strategy (1+1 ES):** A derivative-free search algorithm that treats the quantized network as a black-box oracle, completely avoiding backpropagation and activation caching.
2. **Polynomial Subspace Parameterization (PolyMerge):** Restricting the search space of merging coefficients to a continuous, low-degree polynomial trajectory of normalized layer depth. This low-pass filtering effect removes high-frequency optimization noise and mathematically prevents degenerate states.
3. **Symmetric Uniform Weight Quantization (per-channel for INT4):** Using standard symmetric PTQ to represent model weights in low-bit integer formats. Per-channel scaling is treated as a strict architectural mandate under 4-bit quantization to prevent outlier expert weights from crushing the dynamic range of entire layers.
4. **Post-Hoc Classification Head Quantization:** Quantizing the task-specific linear heads post-hoc to 8-bit precision (INT8) after continuous coefficient optimization. This ensures a 100% integer weight pipeline, removing the need to store any floating-point parameters on the physical edge device.

## 3. Mathematical Formulation

### Preliminaries and Task Vectors
Let $\Theta_{\text{base}} \in \mathbb{R}^M$ represent the weights of a pre-trained base model (e.g., a CLIP ViT or pre-trained Vision Transformer backbone). Let $\Theta_k \in \mathbb{R}^M$ represent the weights of expert $k \in \{1, 2, \dots, K\}$, fine-tuned from the same base checkpoint on a distinct task. We partition the model into $L$ sequential layer-specific parameter blocks. For layer block $l \in \{0, 1, \dots, L-1\}$, the expert task vector $\mathbf{\Delta}_{k, l} \in \mathbb{R}^{M_l}$ is defined as:
$$\mathbf{\Delta}_{k, l} = \Theta_{k, l} - \Theta_{\text{base}, l}$$

### Polynomial Coefficient Parameterization
Instead of optimizing independent coefficients $\lambda_{k, l} \in \mathbb{R}$ for each task $k$ and layer $l$, Q-PolyMerge parameterizes the merging coefficients as a continuous polynomial of normalized depth $\bar{l} = \frac{l}{L-1} \in [0, 1]$ of degree $d$:
$$\lambda_{k, l}(\boldsymbol{\alpha}) = \sum_{j=0}^d \alpha_{k, j} \cdot \left( \frac{l}{L-1} \right)^j$$
where $\boldsymbol{\alpha} = \{ \alpha_{k, j} \} \in \mathbb{R}^{K \times (d+1)}$ represents the small set of learnable polynomial parameters. Bounding the layer index to the compact interval $[0, 1]$ is crucial for numerical stability (preventing overflow in deep models) and scale invariance across architectures of varying depths.

The continuous merged weights at layer $l$ before quantization are defined as:
$$\Theta_{\text{merged}, l}(\boldsymbol{\alpha}) = \Theta_{\text{base}, l} + \sum_{k=1}^K \lambda_{k, l}(\boldsymbol{\alpha}) \mathbf{\Delta}_{k, l}$$

### Quantization Operator
We compress the merged weights to $b$ bits (e.g., $b=4$ or $b=8$) using a symmetric uniform post-training quantization operator $q(\cdot)$:
$$\Theta^q_{\text{merged}, l}(\boldsymbol{\alpha}) = q\left( \Theta_{\text{merged}, l}(\boldsymbol{\alpha}) \right)$$
For 8-bit quantization (INT8), per-tensor scaling is used. For aggressive 4-bit quantization (INT4), to preserve coordinate structures and linear mode connectivity, per-channel weight quantization is used. The per-channel quantization operator is formulated as:
$$q(W_{c, \cdot}) = \text{clamp}\left( \text{round}\left( \frac{W_{c, \cdot}}{S_c} \right), -2^{b-1}, 2^{b-1}-1 \right) \cdot S_c$$
where $W_{c, \cdot}$ represents the weights of output channel $c$, and $S_c$ is the per-channel scale factor defined by the absolute maximum weight value in that channel:
$$S_c = \frac{\max_i(|W_{c, i}|)}{2^{b-1} - 1}$$

### Optimization Objective
At test-time, the model receives a stream of unlabeled target images. We optimize the continuous polynomial parameters $\boldsymbol{\alpha}$ by minimizing the Shannon entropy of predictions:
$$\mathcal{L}_{\text{TTA}}(\boldsymbol{\alpha}) = \sum_{k=1}^K \mathbb{E}_{x \sim \mathcal{D}_k^{\text{unlabeled}}} \left[ H\left( f_{\Theta^q_{\text{merged}}(\boldsymbol{\alpha})}(x) \right) \right]$$
where $H(\mathbf{p}) = -\sum_{c=1}^C p_c \log (p_c + \epsilon)$, and $\epsilon = 10^{-8}$ is a numerical stability parameter.

### Gradient Optimization via Straight-Through Estimator (STE)
Since the $\text{round}$ function in the quantization operator has zero derivative almost everywhere, standard backpropagation fails. Q-PolyMerge resolves this using the Straight-Through Estimator (STE), which approximates the derivative of the rounding operation as the identity mapping during the backward pass:
$$\frac{\partial q(W)}{\partial W} \approx \mathbf{I}$$
Applying the chain rule, the gradient of the surrogate loss with respect to the learnable polynomial parameter $\alpha_{k, j}$ is computed as:
$$\frac{\partial \mathcal{L}_{\text{TTA}}}{\partial \alpha_{k, j}} = \sum_{l=0}^{L-1} \frac{\partial \mathcal{L}_{\text{TTA}}}{\partial \Theta^q_{\text{merged}, l}} \cdot \frac{\partial \Theta^q_{\text{merged}, l}}{\partial \lambda_{k, l}} \cdot \frac{\partial \lambda_{k, l}}{\partial \alpha_{k, j}}$$
Using the STE approximation, the gradient flows directly through quantization:
$$\frac{\partial \mathcal{L}_{\text{TTA}}}{\partial \alpha_{k, j}} \approx \sum_{l=0}^{L-1} \left( \frac{\partial \mathcal{L}_{\text{TTA}}}{\partial \Theta^q_{\text{merged}, l}} \cdot \mathbf{\Delta}_{k, l} \right) \cdot \left( \frac{l}{L-1} \right)^j$$
The parameters $\boldsymbol{\alpha}$ are updated iteratively using the first-order Adam optimizer.

### Derivative-Free Optimization via 1+1 Evolution Strategy (1+1 ES)
Under zero-order optimization, we treat the quantized model as a black-box oracle, completely bypassing the need for backpropagation and activation caching. At each step, we perturb the active parameters $\boldsymbol{\alpha}_k \to \boldsymbol{\alpha}_k + \sigma \cdot \mathbf{z}$ where $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$, and evaluate the target entropy. Because the search dimensionality is reduced from $L \times K$ to $(d+1) \times K$ parameters, the black-box search is highly efficient, avoiding the curse of dimensionality.

## 4. Architecture Specifications
We utilize a standardized and reproducible deep vision architecture to evaluate our framework:

- **Network Backbone:** Pre-trained Vision Transformer (timm `vit_tiny_patch16_224`), containing approximately 5.7M parameters.
- **Layer-wise Grouping:** The backbone parameters are grouped into $L=14$ discrete layers:
  - Layer 0: Patch embeddings.
  - Layers 1–12: Twelve independent Transformer blocks.
  - Layer 13: Final layer normalization layer.
- **Model Quantization Configurations:**
  - **INT8 Pipeline:** Backbone weights quantized to 8-bit symmetric uniform per-tensor; classification heads post-hoc quantized to 8-bit.
  - **INT4 Pipeline:** Backbone weights quantized to 4-bit symmetric uniform per-channel (strict mandate); classification heads post-hoc quantized to 8-bit.
  - **Activations:** Maintained in FP16/FP32 precision during inference to preserve representational precision.
- **Learnable Parameter Spaces:**
  - Unconstrained Merging (AdaMerging / Q-Merge): $14 \text{ layers} \times K \text{ tasks}$ parameters. For $K=4$, this yields **56 parameters**.
  - Q-PolyMerge ($d=2$ quadratic polynomial): $(2+1) \text{ parameters} \times K \text{ tasks}$ parameters. For $K=4$, this yields **12 parameters** (a **78.6% search space reduction**).

## 5. Baselines
We compare Q-PolyMerge against several state-of-the-art and foundational baselines under identical quantization and data regimes:

1. **Task Arithmetic (Unquantized FP16 Baseline):** Naive weight averaging using a static uniform scalar coefficient $\lambda_k = 0.3$ per task across all layers (unquantized performance ceiling).
2. **Task Arithmetic (Quantized Baseline):** Naive post-merge quantization (INT8 or INT4) of a model merged via standard Task Arithmetic using uniform coefficients, providing the unoptimized low-bit baseline.
3. **AdaMerging (Post-Hoc Quantized):** Optimizing merging coefficients layer-by-layer on the full-precision model using standard AdaMerging (Adam GD), and subsequently quantizing the final merged weights. This demonstrates the "alignment loss" and severe representation degradation of post-hoc quantization.
4. **Q-Merge (Unconstrained Quantization-Aware Merging):** Optimizing merging coefficients layer-by-layer directly under the quantization operator (Adam with STE or 1+1 ES), utilizing $L \times K = 56$ parameters. This isolates the effect of our continuous polynomial constraint on overfitting and generalization.
5. **PolyMerge (Unquantized Baseline):** Standard PolyMerge ($d=2$) executed on the full-precision weights, serving as the continuous unquantized ceiling.

## 6. Step-by-Step Interaction
During test-time adaptation, the data and gradient flow through the Q-PolyMerge framework as follows:

```
                  [ Calibration Batch x_t ]
                              │
                              ▼
[ Poly Parameters α ] ──► [ Vandermonde Matrix V ] ──► [ Merging Coefficients λ ]
                                                              │
                                                              ▼
[ Base Weights Θ_base ] ◄────────────────────────────── [ Task Vectors Δ ]
         │
         ▼
[ Continuous Merged Weights Θ_merged ] ──► [ Quantize Operator q(·) ]
                                                        │
                                                        ▼
                                           [ Quantized Weights Θ^q_merged ]
                                                        │
                                                        ▼
                                           [ Forward Pass: Compute Predictions ]
                                                        │
                                                        ▼
                                           [ Compute Entropy Loss L_TTA ]
                                                        │
                                                        ▼
[ Update α via Adam ] ◄── [ STE Gradient Flow: Backpropagate Gradients ]
```

1. **Calibration Stream Arrival:** At test-time, a small stream of unlabeled calibration images $x_t \sim \mathcal{D}_k^{\text{unlabeled}}$ arrives at the device (e.g., $S=32$ images per task).
2. **Subspace Mapping:** The continuous polynomial parameters $\boldsymbol{\alpha} \in \mathbb{R}^{K \times (d+1)}$ are mapped to layer-specific coefficients $\boldsymbol{\lambda} \in \mathbb{R}^{K \times L}$ via multiplication with the Vandermonde matrix $\mathbf{V}$:
   $$\boldsymbol{\lambda}_k = \mathbf{V}\boldsymbol{\alpha}_k$$
3. **Continuous Weight Merging:** The expert task vectors $\mathbf{\Delta}_{k, l}$ are scaled and added to the pre-trained base model weights $\Theta_{\text{base}, l}$ to generate the continuous merged weights:
   $$\Theta_{\text{merged}, l} = \Theta_{\text{base}, l} + \sum_{k=1}^K \lambda_{k, l} \mathbf{\Delta}_{k, l}$$
4. **Symmetric Uniform Quantization:** The continuous merged weights are passed through the rounding and clamping operator $q(\cdot)$ with scale factor $S_c$ (per-channel for INT4) to output the integer-quantized weights $\Theta^q_{\text{merged}, l}$.
5. **Inference Forward Pass:** The calibration batch $x_t$ is routed through the quantized model to compute output logit distributions $f_{\Theta^q_{\text{merged}}}(x_t)$. 
6. **Loss Evaluation:** The Shannon entropy loss $\mathcal{L}_{\text{TTA}}$ is computed over the batch.
7. **Backpropagation with STE:** The backward pass is executed. When the gradient reaches the quantization operator, the non-differentiable rounding function is bypassed via the Straight-Through Estimator (STE), which copies gradients back directly.
8. **Subspace Parameter Update:** The gradient of the loss with respect to the low-dimensional parameters $\boldsymbol{\alpha}$ is calculated, and $\boldsymbol{\alpha}$ is updated via the Adam optimizer.
9. **Inference Execution:** Once test-time adaptation converges (e.g., after 40 iterations), the final continuous parameters $\boldsymbol{\alpha}$ are frozen and used to construct a static, fully quantized, memory-efficient merged multi-task network $\Theta^q_{\text{merged}}(\boldsymbol{\alpha})$ for downstream edge deployment.
