# Idea Proposal: A Methodological Deconstruction and Robustness Audit of Quantization-Aware Model Merging

## 1. Persona Alignment
This technical implementation is deeply grounded in the core traits and instructions of the **Methodologist** persona:
- **Skepticism of "State-of-the-Art" Claims:** Recent works like Q-Merge claim that optimizing model-merging coefficients directly under a specific quantization operator (using Straight-Through Estimators, STE) achieves "near-lossless" low-bit merging. We approach this claim with methodological skepticism, questioning whether the learned coefficient configurations are highly overfitted to the specific mathematical formulation of that quantization operator (e.g., Symmetric Per-Channel Round-to-Nearest).
- **Exposing Flaws and Confounding Variables:** We identify "Quantization-Operator Overfitting" as a critical, unstudied confounding variable. If the optimized coefficients fail to generalize across standard post-training quantization (PTQ) formats or hardware backends, then the claimed robustness is a brittle artifact rather than a generalizable weight-space alignment.
- **Rigor and Comprehensive Evaluation:** We propose a multi-axial evaluation pipeline spanning diverse quantization backends, calibration stream characteristics (sample sizes, label skew, and corruptions), and regularization methods, setting up a far more rigorous testing ground than existing studies.

## 2. Core Techniques
- **Post-Training Quantization (PTQ) Operators:** We implement uniform symmetric/asymmetric, per-tensor/per-channel, and double-quantization (DQ) operators at 8-bit and 4-bit precision.
- **Straight-Through Estimator (STE) Optimization:** First-order gradient-based optimization using Adam is guided by STE to propagate gradients backward through the non-differentiable rounding operator:
  $$\frac{\partial \text{round}(x)}{\partial x} \approx 1$$
- **Black-Box Optimization (1+1 ES):** A derivative-free 1+1 Evolution Strategy serves as a robust comparator to evaluate whether gradient-based optimization or derivative-free search is more resilient to PTQ schema shift.
- **Cross-Schema Evaluation Framework:** The core technique of our audit. Coefficients are optimized under a source quantization schema $Q_{\text{opt}}$ but evaluated under a different target deployment schema $Q_{\text{eval}}$.
- **Regularization Smoothers:** We integrate Elastic Spatial Regularization (ESR) and Class-Capacity Normalization (CCN) to stabilize the optimization trajectory and mitigate cross-operator overfitting.

## 3. Mathematical Formulation

Let $\theta_{\text{pre}}$ be the pre-trained backbone weights, and let $\{\theta_1, \dots, \theta_K\}$ be $K$ task-specific expert weights. The unquantized merged weights at layer $l \in \{1, \dots, L\}$ are defined as:
$$\theta^l_{\text{merged}}(\Lambda) = \theta^l_{\text{pre}} + \sum_{k=1}^K \lambda^l_k (\theta^k_l - \theta^l_{\text{pre}})$$
where $\Lambda \in [0, 1]^{K \times L}$ represents the matrix of merging coefficients.

### Quantization Formulations
We define the uniform asymmetric quantization operator $Q_{\text{asym}}(W, b)$ for weight tensor $W$ and bit-width $b$ as:
$$s = \frac{\max(W) - \min(W)}{2^b - 1}$$
$$z = \text{round}\left( \frac{-\min(W)}{s} \right) - 2^{b-1}$$
$$W_{\text{quant}} = \text{clamp}\left( \text{round}\left( \frac{W}{s} \right) + z, -2^{b-1}, 2^{b-1}-1 \right)$$
$$Q_{\text{asym}}(W, b) = s \cdot (W_{\text{quant}} - z)$$

For symmetric quantization $Q_{\text{sym}}(W, b)$, $z = 0$, and scale is:
$$s = \frac{\max(|W|)}{2^{b-1} - 1}$$
$$Q_{\text{sym}}(W, b) = s \cdot \text{clamp}\left( \text{round}\left( \frac{W}{s} \right), -2^{b-1}, 2^{b-1}-1 \right)$$

For per-channel quantization, scales $s_c$ and zero-points $z_c$ are calculated independently for each output channel slice $W[c, \dots]$.

### Optimization Objective
The unsupervised adaptation objective is to minimize prediction entropy $\mathcal{L}_{\text{entropy}}(\Lambda)$ on a calibration set of $N$ samples:
$$\mathcal{L}_{\text{entropy}}(\Lambda) = \frac{-1}{N} \sum_{i=1}^N \sum_{c=1}^C P(y=c \mid X_i; \theta_{\text{quant}}(\Lambda)) \log P(y=c \mid X_i; \theta_{\text{quant}}(\Lambda))$$
where:
$$\theta_{\text{quant}}(\Lambda) = Q_{\text{opt}}(\theta_{\text{merged}}(\Lambda), b)$$

Using the Straight-Through Estimator, gradient updates follow:
$$\Lambda^{(t+1)} = \Lambda^{(t)} - \eta \cdot \text{Adam}\left( \nabla_{\Lambda} \mathcal{L}_{\text{entropy}}(\Lambda^{(t)}) \right)$$

### Cross-Schema Generalization Gap
To measure how much the optimized coefficients overfit to $Q_{\text{opt}}$, we define the **Cross-Schema Generalization Gap**:
$$\Delta \text{Acc}(Q_{\text{opt}} \to Q_{\text{eval}}) = \text{Acc}(\theta_{\text{quant}}(\Lambda^*); Q_{\text{eval}}) - \text{Acc}(\theta_{\text{quant}}(\Lambda^*); Q_{\text{opt}})$$
where $\Lambda^* = \arg\min_{\Lambda} \mathcal{L}_{\text{entropy}}(\Lambda)$ under $Q_{\text{opt}}$.

## 4. Architecture Specifications
- **Backbone Model:** pre-trained `timm ViT-Tiny` (`vit_tiny_patch16_224`, 5.7M parameters).
- **Layers:** $L=14$ discrete layer groups (patch embeddings, 12 Transformer blocks, and the final layer normalization layer).
- **Expert Models:** $K=4$ experts representing classification heads and fine-tuned backbones for MNIST, FashionMNIST, CIFAR-10, and SVHN.
- **Search Space:** $\Lambda \in [0, 1]^{4 \times 14}$, representing 56 continuous parameters initialized to 0.3.
- **Inputs:** 224x224 normalized images from the calibration splits.
- **Outputs:** Softmax probability vectors representing class predictions.

## 5. Baselines
- **Uniform Task Arithmetic (FP16 Baseline):** The standard, unoptimized, unquantized baseline with uniform coefficients $\lambda^l_k = 0.3$.
- **AdaMerging (FP16 Optimized, Unquantized):** Standard unquantized AdaMerging with Adam GD, establishing the unquantized performance ceiling.
- **Naive Merge-then-Quantize (M-then-Q):** Merging full-precision experts with uniform coefficients, then applying the target evaluation quantization $Q_{\text{eval}}$ without any test-time optimization.
- **Quantized AdaMerging (AdaMerging + Quantized):** Optimizing coefficients in full-precision, then applying post-hoc quantization $Q_{\text{eval}}$.
- **Symmetric Per-Channel Q-Merge (Source Baseline):** Standard Q-Merge optimized and evaluated under Symmetric Per-Channel RTN quantization.

## 6. Step-by-Step Interaction
1. **Initialize Weights:** Load pre-trained ViT-Tiny backbone $\theta_{\text{pre}}$ and the 4 expert checkpoints $\{\theta_1, \dots, \theta_4\}$. Initialize coefficients $\Lambda$ to a uniform matrix of 0.3.
2. **Sample Calibration Stream:** Retrieve an unlabeled calibration stream of size $N$ (varying $N \in \{1, 2, 4, 8, 16, 32, 64, 128\}$) per task. Optionally inject distribution shifts (corruptions) or class imbalance.
3. **Weight Merging & Quantization:** Assemble the merged weights $\theta_{\text{merged}}(\Lambda)$ across all 14 layers. Apply the optimization quantization schema $Q_{\text{opt}}$ (e.g., Symmetric Per-Channel) to produce the quantized weights $\theta_{\text{quant}}(\Lambda)$.
4. **Forward Pass:** Run the calibration images through the quantized network. Compute logits and softmax prediction distributions for each task.
5. **Loss & Backpropagation:** Compute the multi-task prediction entropy loss. Propagate gradients back through the network and through the rounding operator using STE to update the continuous coefficients $\Lambda$. Repeat for 40-100 steps until convergence.
6. **Deploy & Cross-Evaluate:** Freeze the optimized coefficients $\Lambda^*$. Re-assemble the merged weights and apply a different target evaluation schema $Q_{\text{eval}}$ (e.g., Asymmetric Per-Tensor, double-quantization, or AdaRound).
7. **Metric Collection:** Run inference on the independent, unseen test splits. Record task-wise and average accuracies to compute the cross-schema generalization gap and analyze sensitivity to stream size, label skew, and corruptions.
