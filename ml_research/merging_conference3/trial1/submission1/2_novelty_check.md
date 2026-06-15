# 2. Novelty Check

An analysis of the novelty and originality of the proposed **QP-Merge** framework highlights that while it is highly practical, robust, and represents the first co-design of model merging and quantization, its core technical components are elegant adaptations and combinations of established techniques from the post-training quantization (PTQ) literature.

## 1. Outlier-Residual Decoupling (ORD) vs. Existing Literature
- **Existing Concept:** Separating extreme weight outliers into high-precision, highly sparse matrices while keeping the remaining dense weights in low-bit quantized format is a well-known technique in single-model PTQ. Specifically, **SqueezeLLM** (Kim et al., 2023) and **SpQR** (Dettmers et al., 2023) introduced this exact hybrid representation (dense quantized + sparse high-precision) to preserve performance under extreme 3-bit or 4-bit compression.
- **QP-Merge Adaptation:** QP-Merge applies this concept to *task vectors* (the parameter differences $\Delta W_t = W_t - W_{\text{base}}$) rather than the final model weights themselves. While applying outlier decoupling to weight updates is a highly practical and logical transition for model merging, the underlying mathematical representation is conceptually identical to SqueezeLLM and SpQR.
- **Novelty Assessment:** *Moderate to Incremental.* The conceptual transfer from full-weight outlier decoupling to delta-weight outlier decoupling is intuitive and straightforward. The paper properly cites SqueezeLLM and SpQR, but its primary distinction is the application domain (multi-task model merging) rather than a novel algorithmic formulation of outlier handling.

## 2. Quantization-Error Aware Scale Calibration (QE-Calib) vs. Existing Literature
- **Existing Concept:** Optimizing diagonal weight scaling matrices or scale parameters to minimize activation reconstruction error over a small, unlabeled calibration dataset is a standard practice in post-training quantization. For instance:
  - **SmoothQuant** (Xiao et al., 2023) and **AWQ** (Lin et al., 2023) utilize channel-wise weight scaling to migrate quantization difficulty from activations to weights.
  - **AdaRound** (Nagel et al., 2020) and **BRECQ** (Li et al., 2021) optimize rounding or scale parameters using layer-wise reconstruction loss over a small calibration set (typically 1024 or 128 samples).
- **QP-Merge Adaptation:** QE-Calib optimizes layer-wise diagonal scalers $D_l$ and merging coefficients $\lambda$ over $M=128$ unlabeled samples using Adam to minimize end-to-end embedding MSE.
- **Novelty Assessment:** *Moderate.* Jointly optimizing merging coefficients $\lambda$ and weight scalers $D_l$ is a natural extension of PTQ calibration to model merging. However, optimizing merging coefficients post-hoc is a known heuristic, and applying standard gradient-based PTQ optimization to them is a direct application of existing techniques.

## Summary of Originality
The originality of the paper does not lie in introducing entirely new mathematical primitives, but rather in the **creative, high-impact co-design and engineering integration** of these two separate research areas (PTQ and Model Merging). 
- **Strengths:** It is the first work to explicitly co-design model merging and post-training quantization. It identifies the physical reasons why merged models fail under low-bit quantization (weight scale stretching by delta outliers and activation mismatches) and applies the right tools from the PTQ toolbox to solve them.
- **Weaknesses:** The theoretical novelty is limited. The paper does not provide new theoretical insights or convergence guarantees. It relies on existing concepts (SqueezeLLM's outlier decoupling and AdaRound/SmoothQuant-style reconstruction loss optimization) without substantial algorithmic modification.
- **Verdict on Novelty:** Highly appropriate for an applied machine learning conference (such as ICML or NeurIPS), where practical, highly effective, and well-motivated co-designs of existing techniques are highly valued.
