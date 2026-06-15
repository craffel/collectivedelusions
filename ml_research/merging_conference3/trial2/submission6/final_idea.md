# Quantization-Aware Model Merging (Q-Merge)

## 1. Persona Alignment
This technical implementation directly aligns with the core philosophy and traits of **The Pragmatist**:
- **Real-World Deployment Constraints**: Quantization (INT8 or INT4) is the industry standard for deploying deep learning models on edge devices, mobile platforms, and low-latency servers. Storing and serving uncompressed FP16/FP32 multi-task models is often commercially and technically unfeasible.
- **Solving Real-World Bottlenecks**: While model merging successfully fuses expert models without retraining, the resulting models suffer severe performance degradation when quantized post-merge (due to quantization noise disrupting aligned weight spaces). Conversely, merging pre-quantized models causes catastrophic interference because discrete quantized intervals do not align. Q-Merge solves this exact real-world bottleneck by optimizing merging parameters *directly* under the quantization operator.
- **Robust and Simple Integration**: We utilize standard Round-to-Nearest (RTN) post-training quantization, avoiding expensive Quantization-Aware Training (QAT) loops. Optimization of merging coefficients is performed at test-time using unlabeled data, keeping the pipeline extremely simple, highly robust, and easy to integrate into existing model-serving systems.

## 2. Core Techniques
- **Task Arithmetic (TA)**: Fuses task-specific capabilities by extracting task vectors $\tau_k = \theta_k - \theta_{\text{base}}$ and adding them to the base model with scaled coefficients $\Lambda$~\cite{ilharco2022editing}.
- **Post-Training Quantization (PTQ)**: Compresses full-precision weights into low-bit representations (INT8 or INT4) using uniform symmetric/asymmetric quantization.
- **Zero-Order Test-Time Adaptation (1+1 ES)**: A derivative-free black-box optimizer to search for optimal merging coefficients $\Lambda$ directly on the non-differentiable quantized model, bypassing the rugged, discontinuous landscape that breaks standard first-order optimizers~\cite{yang2024adamerging}.
- **First-Order Straight-Through Estimator (STE)**: Backpropagation-based optimization (Adam GD) using the Straight-Through Estimator~\cite{bengio2013estimating} to pass gradients through the non-differentiable `round` function, enabling a direct empirical study of the *Overfitting-Optimizer Paradox* in quantized spaces.

## 3. Mathematical Formulation
Let $\theta^l_{\text{base}} \in \mathbb{R}^D$ be the weights of the pre-trained base model at layer $l \in \{1, \dots, L\}$. For each of the $K$ downstream tasks, let $\theta^l_k$ be the fine-tuned expert weights, and define the task vector:
$$ \tau^l_k = \theta^l_k - \theta^l_{\text{base}} $$

We combine these task vectors using layer-wise merging coefficients $\lambda^l_k \in [0, 1]$ to compute the full-precision merged weights:
$$ \theta^l_{\text{merged}}(\Lambda) = \theta^l_{\text{base}} + \sum_{k=1}^K \lambda^l_k \tau^l_k $$

We then apply symmetric, uniform quantization to scale and map the merged weights to $b$-bit integer representations (where $b \in \{4, 8\}$):
$$ \theta^l_{\text{quant}}(\Lambda) = \text{clip}\left(\text{round}\left(\frac{\theta^l_{\text{merged}}(\Lambda)}{S^l}\right), -2^{b-1}, 2^{b-1}-1\right) \times S^l $$

where the scaling factor $S^l$ is computed dynamically per layer to preserve the range of the merged weights:
$$ S^l = \frac{\max(|\theta^l_{\text{merged}}(\Lambda)|)}{2^{b-1} - 1} $$

The objective is to optimize the coefficient set $\Lambda = \{\lambda^l_k\}$ at test-time on an unlabeled calibration stream to minimize the joint entropy loss of the quantized model:
$$ \min_{\Lambda} \mathcal{L}(\Lambda) = \frac{1}{N} \sum_{i=1}^N \mathcal{H}\left(f\left(x_i; \theta_{\text{quant}}(\Lambda)\right)\right) $$

where $\mathcal{H}(p) = -\sum_c p_c \log p_c$ represents Shannon entropy, and $f(x; \theta)$ denotes the classification probability output.

We evaluate two optimization paradigms:
1. **1+1 ES (Zero-Order)**: Proposes mutations $\Lambda_{cand} = \text{clip}(\Lambda + \mathcal{N}(0, \sigma^2 I), 0, 1)$ and accepts them if $\mathcal{L}(\Lambda_{cand}) < \mathcal{L}(\Lambda)$.
2. **Adam GD with STE (First-Order)**: Computes gradients using the approximation $\frac{\partial \text{round}(x)}{\partial x} \approx 1$, propagating gradients from the classification loss back to the full-precision coefficients $\Lambda$.

## 4. Architecture Specifications
- **Backbone Network**: CLIP ViT-B/32 (or ViT-Tiny from `timm`), comprising 12 Transformer layers and 1 visual projection layer ($L = 13$ parameter groups).
- **Inputs**: Multi-task unlabeled calibration streams $X$ (e.g., MNIST, FashionMNIST, CIFAR-10, SVHN).
- **Target Quantization Layers**: Visual projection weights (`model.visual.proj`) or full-network attention and feed-forward weight projection matrices.
- **Bit-Width Configurations**: INT8 (standard precision) and INT4 (aggressive compression).
- **Outputs**: High-fidelity classification logits and predictions matching the multi-task targets.

## 5. Baselines
We evaluate Q-Merge against the following appropriate baselines:
1. **FP16 Merged Model (Upper Bound)**: Fusing full-precision experts without quantization. This defines the ideal performance target.
2. **Quantize-then-Merge (Q-then-M)**: Quantizing each expert first to INT8/INT4, and then merging them via standard Task Arithmetic. This baseline validates whether pre-quantization destroys linear mode connectivity.
3. **Merge-then-Quantize (M-then-Q)**: Merging the full-precision experts using unoptimized Task Arithmetic coefficients, and then quantizing the resulting merged model. This is the standard naive deployment pipeline.
4. **Post-Merge Naive Quantization with AdaMerging (FP16 Optimized)**: Optimizing the coefficients $\Lambda$ on the FP16 model first (using standard AdaMerging), and then quantizing the resulting model. This baseline tests whether FP16-optimized coefficients are robust to subsequent quantization noise.

## 6. Step-by-Step Interaction
1. **Extraction**: Task experts $\theta_k$ and the base model $\theta_{\text{base}}$ are loaded, and task-specific vectors $\tau_k$ are extracted.
2. **Fusing**: The full-precision merged weights $\theta_{\text{merged}}(\Lambda)$ are computed dynamically using the current merging coefficients $\Lambda$.
3. **Quantization**: The merged weights are scaled and quantized to INT8 or INT4, yielding the quantized model weights $\theta_{\text{quant}}(\Lambda)$.
4. **Forward Pass**: Calibration batches are fed forward through the quantized model $\theta_{\text{quant}}(\Lambda)$ to compute predictions and calculate the entropy loss $\mathcal{L}(\Lambda)$.
5. **Optimization Step**:
   - Under **1+1 ES**, candidate coefficients are proposed, and the quantized model is re-evaluated; successful mutations are accepted.
   - Under **Adam GD with STE**, backpropagation is performed; gradients flow through the Straight-Through Estimator of the rounding operator to update the continuous coefficients $\Lambda$.
6. **Inference**: Once the optimization budget is exhausted, the optimized coefficients $\Lambda^*$ are locked, and the final quantized multi-task model $\theta_{\text{quant}}(\Lambda^*)$ is evaluated on the unseen multi-task test sets.
