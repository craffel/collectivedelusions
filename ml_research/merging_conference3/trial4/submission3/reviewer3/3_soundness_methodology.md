# 3. Soundness and Methodology

## Clarity of the Description
The mathematical and procedural descriptions in this paper are exceptionally clear and rigorous:
- **QLoRA Merging & Naive Re-Quantization:** The equations describing the dequantization of $Q_b(W_0)$ to high-precision, the summation of adapters under blending coefficients, and subsequent uniform symmetric quantization are mathematically complete and easy to follow.
- **The Closed-Form Derivation of SAWS Alignment Factor ($c^l$):** The minimization of the squared Frobenius norm distance between the original unquantized merged weight tensor and the scaled, re-quantized weight tensor is derived cleanly and elegantly.
- **Representation Scale Preservation Dilemma:** The proof showing that dividing the entire layer output by the scaling factor $\gamma^l$ during inference scales down and collapses the pre-trained base features is mathematically sound and conceptually instructive.
- **QA-ACS Optimization:** The application of the Straight-Through Estimator (STE) and the prediction entropy objective are clearly formulated, and the discussion on how the Adam optimizer's running averages filter the high-frequency gradient noise caused by STE mismatch is highly intuitive and technically sound.

## Appropriateness of Methods
The methodology employed is highly appropriate for auditing and mitigating quantization errors in model merging:
- **Multi-Axial Auditing (RQA):** Testing across multiple dimensions—bit-widths (4-bit vs. 8-bit), quantization granularities (per-tensor vs. per-channel), and formats (symmetric vs. asymmetric)—provides a highly complete picture of performance.
- **Backbone and Experts:** Using a standard Vision Transformer backbone (`vit_tiny_patch16_224` from `timm`) and training four task-specific adapters on distinct classification datasets (MNIST, FashionMNIST, CIFAR-10, SVHN) is a valid, well-understood setup.
- **Baselines:** The paper compares proposed methods against multiple necessary baselines, including unmerged FP16 experts, naive FP16 merge, naive re-quantization, decoupled co-existence (Quantize-then-Merge), and post-hoc quantized AdaMerging.

## Technical Flaws & Limitations
While there are no major mathematical errors or logical flaws, the methodology exhibits a few notable limitations that the authors themselves transparently and self-critically deconstruct:

1. **Small-Scale Vision Transformer Backbone:**
   The primary evaluation is restricted to a small-scale Vision Transformer (`vit_tiny`, 5.7M parameters). This is a toy model compared to the multi-billion parameter architectures (such as LLaMA or Mistral) where QLoRA and model merging are typically deployed. While the authors mitigate this by measuring double-quantization noise on the larger `vit_base` (86M parameters) in Table 1, and outlining large-scale scaling hypotheses in Appendix C, the lack of complete multi-task auditing results on a multi-billion parameter model is a notable limitation.

2. **Severe Pre-existing Task Interference Confounder:**
   The continuous, full-precision baseline (Naive FP16 Merge) already exhibits severe performance degradation ($66.65\%$ mean accuracy compared to the unmerged expert ceiling of $93.85\%$). Merging four highly disparate datasets (MNIST, FashionMNIST, CIFAR-10, SVHN) via task arithmetic introduces massive weight-space conflicts. Studying downstream quantization on a model that is already severely degraded makes it difficult to isolate quantization-induced representation erasure from pre-existing task interference. The authors' proposed **"Zero-Interference RQA Protocol"** (Appendix Section 5.1) is an excellent methodological suggestion to decouple these issues, but it is not implemented in the main experimental results.

3. **Fragility of Unsupervised QA-ACS:**
   The unconstrained unsupervised QA-ACS variant is shown to be highly unstable under severe 4-bit per-tensor and per-channel quantization noise, suffering from **entropy collapse** and dropping MNIST accuracy below Naive-RQ. While the authors resolve this in Appendix A.5 using supervised labels and $L_2$ regularization, this highlights that the core unsupervised test-time optimization formulation is fundamentally fragile in low-bit regimes.

## Reproducibility
The paper achieves an outstanding standard of reproducibility:
- Appendix A provides comprehensive detail on expert training hyperparameters (optimizer, learning rate, weight decay, epochs, batch size, LoRA rank/alpha/dropout), SAWS scaling constant ($\alpha = 0.08$), and QA-ACS optimization parameters (unlabeled calibration size $N=16$, learning rate $0.02$, Adam optimizer, $T=40$ steps).
- The authors conduct exhaustive sensitivity analyses, including:
  - Sensitivity of QA-ACS to calibration dataset size ($N \in \{16, 64, 128\}$).
  - Sensitivity of QA-ACS to different optimizers (SGD vs. Adam) and learning rates.
  - A hyperparameter sensitivity sweep of the SAWS scaling constant $\alpha \in [0.01, 0.50]$ (Table 7).
- Ablations between Global SAWS and Channel-wise SAWS are thoroughly documented (Table 10).

The level of detail and empirical transparency ensures that any machine learning researcher could easily reproduce the findings.
