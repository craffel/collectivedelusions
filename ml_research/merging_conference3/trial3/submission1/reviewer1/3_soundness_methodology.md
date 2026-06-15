# Soundness and Methodology Evaluation

This evaluation assesses the mathematical and empirical rigor, clarity, appropriateness of methods, potential technical flaws, and reproducibility of the paper under review.

---

## Clarity of the Description
The methodology and experimental sections are written with **exceptional clarity and rigor**. Every mathematical component is explicitly defined and formalized:
- **Model Merging:** The layer-wise blending coefficient parameterization $\Lambda \in [0, 1]^{K \times L}$ is mathematically precise, generalizing standard Task Arithmetic and AdaMerging.
- **PTQ Operators:** The equations for Uniform Asymmetric and Symmetric Quantization are highly accurate. Crucially, the authors explain that the scale $s$ and zero-point $z$ are dynamically re-calculated during the forward pass, and clarify the asymmetric gradient flow (autograd propagating through $s$ but not $z$). This is an outstanding level of detail rarely seen in quantization papers.
- **Optimization Backends:** Both the first-order Adam-based STE optimization and the derivative-free 1+1 Evolution Strategy (including Rechenberg's 1/5-th success rule equations) are fully detailed.
- **Real-world Context:** The inclusion of hardware deployment scenarios (Edge TPUs Mandating symmetric tensor-wise versus Qualcomm Hexagon DSPs/NVIDIA TensorRT supporting asymmetric channel-wise) grounds the theoretical work in physical engineering realities.

---

## Appropriateness of Methods
The evaluation methodology is highly appropriate and designed with a high level of experimental control:
- **Standardized Benchmark:** Using a standardized pre-trained backbone (`timm ViT-Tiny`, 5.7M parameters) with four classification heads fine-tuned on diverse datasets (MNIST, FashionMNIST, CIFAR-10, SVHN) allows the authors to study the optimization in a controlled, non-confounded setting.
- **Comprehensive Baselines:** The authors include:
  1. *FP16 Task Arithmetic:* Setting a full-precision baseline.
  2. *Naive Merge-then-Quantize (M-then-Q):* Setting a standard post-hoc compressed baseline.
  3. *Quantized AdaMerging:* Isolating whether quantization-aware search is actually necessary (by searching in FP16 and quantizing post-hoc).
  4. *Supervised Calibration Baseline:* To decouple the effects of data scarcity from the weaknesses of unsupervised objectives.
- **Multi-Axial Design:** Evaluating across Calibration Size, Cross-Schema Generalization, Regularization/Optimizer choice, and Stream Skew/Corruptions isolates the failure modes of quantization-aware model merging perfectly.

---

## Potential Technical Flaws and Critical Observations

### 1. Receptive Field and Architectural Factors
The paper observes that ResNet-18 exhibits a smaller cross-schema generalization gap ($-4.25\%$) than ViT-Tiny ($-7.76\%$). The authors explain this using localized translation-invariant kernels in CNNs smoothing out weights compared to global self-attention maps. This is an appropriate and insightful explanation.

### 2. Low-Rank Subspace-Constrained (LoRA-like) Merging
The authors demonstrate that projecting the expert task vectors into a low-rank subspace (SVD projection) closes the cross-schema generalization gap. However, they rightly identify this as a **"Low-Capacity Generalization Illusion"** because the SVD projection is highly destructive, dropping the absolute performance of the model to $13.00\%$. The authors' critical honesty here is outstanding; they recognize that a flat gap is a confounding artifact of severe representation degradation rather than an active, robust alignment of expert weights.

### 3. Scale-Up Methodological Bottlenecks
A minor limitation of the paper is that the empirical evaluations are conducted on a lightweight model (`ViT-Tiny`). However, the authors defend this choice rigorously:
- Weight-space merging requires experts to be fine-tuned from the *exact same* pre-trained initialization, and these expert checkpoints are strictly provided for `ViT-Tiny` in the workspace. Training new experts from scratch on larger architectures would be computationally prohibitive.
- They provide an analytical scaling projection explaining why scaling up parameter count is mathematically expected to *expand* rather than shrink the cross-schema generalization gap due to the exponential explosion of independent discrete rounding thresholds.

---

## Reproducibility
The reproducibility of this paper is **excellent**:
- The authors detail all hyperparameters: learning rate ($10^{-2}$), optimization steps (100), initialization values (0.3), initial step size for 1+1 ES ($\sigma^{(0)} = 0.05$), and TV regularization coefficient ($\alpha=0.5$).
- All data preparation, normalization, and resizing ($224 \times 224$ pixels) are specified.
- The reporting of mean $\pm$ standard deviation over three random seeds in Tables 2, 4, and 5 ensures empirical reliability.
- Access to the LaTeX source and compiled PDF ensures that any researcher has the exact definitions and experimental parameters needed for replication.

---

## Conclusion of Soundness Evaluation
The paper is technically flawless. The math is precise, the baseline comparisons are thorough and fair, and the scientific honesty regarding potential confounders (like the low-capacity generalization illusion) is exceptionally high. The soundness of the technical claims, experimental methodology, and support for the central claims are rated as **excellent**.
