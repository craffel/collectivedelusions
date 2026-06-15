# 1. Summary of the Paper

This paper presents a rigorous, independent methodological deconstruction and multi-axial robustness audit of **Quantization-Aware Model Merging (Q-Merge)**. Utilizing a pre-trained `timm ViT-Tiny` backbone (5.7M parameters) fine-tuned on four diverse classification tasks (MNIST, FashionMNIST, CIFAR-10, and SVHN), the authors critically examine the standard assumptions underlying quantization-aware model merging under Post-Training Quantization (PTQ) constraints. 

The paper challenges several foundational premises of the current SOTA paradigm:
1. **Quantization-Operator Monomorphism:** The assumption that merging coefficients optimized under a simulated quantization operator can generalize to different physical hardware formats without loss of performance.
2. **Calibration Stream Purity:** The assumption that test-time calibration streams are always pristine and perfectly balanced.
3. **STE Gradient Path Fidelity:** The assumption that Straight-Through Estimators (STE) provide high-fidelity gradients for navigating highly non-convex, quantized landscapes.

---

## Key Methodology and Formulations

The paper studies a multi-task model merging setup where the dynamic unquantized merged weights at layer $l$ are defined by:
$$\theta^l_{\text{merged}}(\Lambda) = \theta^l_{\text{pre}} + \sum_{k=1}^K \lambda^l_k (\theta^l_k - \theta^l_{\text{pre}})$$
where $\Lambda \in [0, 1]^{K \times L}$ represents the layer-wise merging coefficients.

The authors evaluate these coefficients under several dynamic Post-Training Quantization (PTQ) operators:
* **Uniform Asymmetric Quantization ($Q_{\text{asym}}$):** Maps weight ranges to $b$-bit integer intervals, where scales ($s$) and zero-points ($z$) are dynamically re-calculated during every forward pass.
* **Uniform Symmetric Quantization ($Q_{\text{sym}}$):** Symmetrizes the range around zero with zero-points fixed at zero.
* **Granularity Variations:** Tensor-wise (global scale/offset) vs. Channel-wise (independent scales/offsets per output channel slice).
* **Double Quantization (DQ):** Quantizes scale factors themselves down to 8-bit precision.

### Optimization Objectives and Optimizers
* **Base Objective (Unsupervised Entropy Minimization):**
  $$\mathcal{L}_{\text{entropy}}(\Lambda) = -\frac{1}{N \cdot K} \sum_{k=1}^K \sum_{i=1}^N \sum_{c=1}^C P(y=c \mid X_{i,k}; \theta_{\text{quant}}(\Lambda)) \log P(y=c \mid X_{i,k}; \theta_{\text{quant}}(\Lambda))$$
  where $\theta_{\text{quant}}(\Lambda) = Q_{\text{opt}}(\theta^l_{\text{merged}}(\Lambda), b)$.
* **Regularized Objective (TV Spatial Smoother):**
  $$\mathcal{L}_{\text{total}}(\Lambda) = \mathcal{L}_{\text{entropy}}(\Lambda) + \alpha \sum_{k=1}^K \sum_{l=1}^{L-1} (\lambda_k^{l+1} - \lambda_k^l)^2$$
* **Optimizers:** First-order gradient descent via Straight-Through Estimators (STE) using Adam, and a derivative-free **1+1 Evolution Strategy (1+1 ES)** as a stochastic black-box comparator.

---

## Crucial Empirical Findings

The paper organizes its evaluation into four main axes:

### Axis 1: Calibration Stream Size Sweep
* Sweeps $N \in \{1, 4, 16, 64\}$ per task under Symmetric Per-Channel quantization.
* **Key Finding:** Direct low-bit optimization via STE is consistently outperformed by **Quantized AdaMerging** (optimizing coefficients in full FP16 precision, then applying post-hoc quantization), which achieves **30.00%** average accuracy versus Q-Merge's peak of **26.25%** (at $N=16$).
* Lowering the learning rate fails to resolve this gap, indicating a fundamental mismatch between straight-through gradient approximations and discontinuous quantized landscapes.

### Axis 2: Cross-Schema Generalization Matrix
* Evaluates coefficients optimized under a source schema $Q_{\text{opt}}$ but deployed under five mismatched target schemas $Q_{\text{eval}}$.
* **Key Finding:** Continuous merging coefficients overfit catastrophically to the exact mathematical operator used during optimization (e.g., coefficients optimized under `sym_channel` collapse to **10.13%** accuracy under `sym_tensor`). 
* Double Quantization is uniquely resilient to schema mismatch, dropping only 2.00% performance under mismatch due to the high precision retention of weight scales.

### Axis 3: Spatial Regularization vs. Black-Box Search
* Compares TV regularized STE and 1+1 ES under severe schema shift (`sym_channel` $\to$ `sym_tensor`).
* **Key Finding:** Spatial regularization fails to bridge discretization gaps. 1+1 ES achieves superior performance on the source schema (**20.75%**) but suffers from more severe target collapse (**8.62%** vs. STE's **10.12%**), showing that derivative-free search overfits intensely to localized rounding thresholds.

### Axis 4: Stream Distortion and Skew Robustness
* Stress-tests optimization under out-of-distribution (OOD) Gaussian input noise and severe Gini class imbalance.
* **Key Finding:** Under severe class skew, unsupervised entropy minimization collapses to **15.50%** average accuracy. Gaussian input noise, however, acts as an accidental stochastic regularizer, smoothing out rounding boundaries and preserving performance.

### Additional Extensions
* **Supervised Calibration Baseline:** Replacing entropy minimization with supervised cross-entropy on $N=16$ samples boosts performance to **35.00%** (clean) and **23.75%** (skewed), isolating the failure of prediction entropy minimization from pure data-scarcity limits.
* **CNN Architecture (ResNet-18):** Demonstrates that convolutional models exhibit a much smaller cross-schema generalization gap ($-4.25\%$), due to translation-invariant localized kernels.
* **Subspace-Constrained Merging:** Projecting expert task vectors into a low-rank subspace (mathematically simulating PEFT/LoRA) achieves a positive generalization gap of $+0.50\%$, confirming that low-intrinsic dimension constraints serve as an excellent structural defense against quantization-operator overfitting.
