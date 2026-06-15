# 1. Summary of the Paper

## Main Topic
The paper investigates the intersection of post-training quantization (PTQ) and model merging (specifically, Low-Rank Adaptation (LoRA) and QLoRA). It addresses a potential methodological blindspot: evaluating model merging in full-precision (FP16/FP32) while ignoring the downstream post-training quantization mandatory for actual edge deployment. The central focus is the **"Re-Quantization Silence"**—the phenomenon where the subtle, task-specific, low-magnitude updates in merged low-rank adapters are rounded to zero during low-bit (e.g., 4-bit) post-hoc quantization, returning the merged model's performance to the level of the unadapted base model.

## Approach
To systematically analyze and address this phenomenon, the authors propose:
1. **Multi-Axial Re-Quantization Auditing (RQA) Framework:** A systematic evaluation of model merging across different quantization granularities (per-tensor vs. per-channel), bit-widths (4-bit vs. 8-bit), and formats (symmetric vs. asymmetric).
2. **Two Mitigations:**
   - **Scale-Adaptive Weight Shifting (SAWS):** A data-free, closed-form weight-scaling method that boosts the magnitude of the adapter updates relative to the base weights before merging, to prevent them from being rounded to zero.
   - **Quantization-Aware Adapter Coefficient Search (QA-ACS):** An optimization-based method that tunes layer-wise merging coefficients directly through the quantization operator using the Straight-Through Estimator (STE) and prediction entropy minimization on a tiny calibration set of 16 unlabeled samples.
3. **In-Depth Deconstruction and Validation:** A self-critical mathematical and empirical analysis of both the proposed mitigations and the underlying hardware constraints, including:
   - Quantifying the confounding error introduced by **Double Quantization format shift** (transitioning from NF4 to INT4/INT8).
   - An individual expert auditing control experiment to decouple task interference from quantization noise.
   - A physical CPU latency profiling benchmark exploring the cache-fitting vs. DRAM-latency bifurcation for co-existence vs. merging.

## Key Findings
- **Quantization Granularity Bifurcation:** The "Re-Quantization Silence" is highly localized to aggressive, sub-optimal **per-tensor** configurations. Under standard **per-channel** grids, naive, unmitigated re-quantization is nearly lossless, losing only $0.15\%$ to $0.30\%$ mean accuracy in 8-bit and $1.80\%$ in 4-bit.
- **Double Quantization Noise:** The transition from a quantile-based non-linear format (NF4) to a uniform linear format (INT4/INT8) shifts quantization bin boundaries, selectively increasing weight-space reconstruction error (Frobenius error increases by up to $+17.6\%$ in 4-bit and $+29.4\%$ in 8-bit on ViT-Base).
- **SAWS Scale Dilemma:** True scale preservation in a single-path merged model is mathematically self-defeating because dividing the layer output by the scaling factor $\gamma^l$ collapses the pre-trained base features. Instead, SAWS' efficacy is driven by **selective task-vector boosting** (boosting the adapter updates to act as a regularizer).
- **Entropy Collapse in QA-ACS:** Test-time optimization of prediction entropy under high 4-bit discretization noise easily drives the model into "entropy collapse" (confidently predicting a single incorrect class). This is successfully mitigated by adding ground-truth labels (Supervised Cross-Entropy) or $L_2$ coefficient regularization.
- **Cache-Fitting vs. DRAM-Latency Bifurcation:** On a tiny model like ViT-Tiny (5.7M), sequential dual-path co-existence is competitive because the model fits entirely within the CPU cache. However, on multi-billion parameter LLMs, weight-space merging is physically mandatory since large-scale models are DRAM-bandwidth bound and sequential dual-path execution incurs a linear $O(K)$ latency penalty.

## Explicitly Claimed Contributions (with Evidence)
1. **Systematic Multi-Axial RQA Framework:** Evaluated across MNIST, FashionMNIST, CIFAR-10, and SVHN using a ViT-Tiny backbone, providing a comprehensive empirical audit of low-bit model merging.
2. **Identification of the Quantization Granularity Bifurcation:** Demonstrated through Tables 2, 3, 4, and 5 that "Re-Quantization Silence" is a highly localized per-tensor artifact rather than a universal blocker.
3. **Double Quantization Noise Deconstruction:** Backed by Table 1 which measures Relative Frobenius reconstruction error across attention layers for both `vit_tiny` and `vit\_base`.
4. **Scale-Adaptive Weight Shifting (SAWS):** Proposed as a data-free closed-form solution. The authors deconstructed its "scale preservation dilemma" and ablated Global vs. Channel-wise scaling (Table 10).
5. **Quantization-Aware Adapter Coefficient Search (QA-ACS):** Formulated and critically evaluated. The authors analyzed its risk of entropy collapse and validated supervised/regularized variants (Table 9) as well as optimizer sensitivity.
6. **Individual Expert Auditing Protocol:** Backed by Table 8, decoupling task interference from quantization noise.
7. **Physical CPU Latency Benchmarking:** Backed by Table 11 and Figure 4, proving the cache-fitting bifurcation.
