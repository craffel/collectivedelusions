# 4. Experiment Check

## Critical Evaluation of the Experimental Setup
The experimental setup designed by the authors is comprehensive and structured, featuring:
- **Consistent Architecture:** A standard Vision Transformer (`vit_tiny_patch16_224`) with LoRA rank $r=8$ targeting query, key, and value projections in all 12 blocks. This ensures a consistent weight distribution and parameter count (5.7M base, plus LoRA parameters) across all tasks.
- **Heterogeneous Tasks:** Evaluating on MNIST, FashionMNIST, CIFAR-10, and SVHN. These datasets represent very different visual domains (digits, clothing, natural objects, street numbers), which provides a challenging scenario for multi-task merging.
- **Robust Quantization Suite:** Auditing across four distinct post-training quantization configurations (INT8 Symmetric Per-Channel, INT4 Symmetric Per-Channel, INT4 Asymmetric Per-Channel, and INT4 Symmetric Per-Tensor) covers both industry standards and aggressive edge-compression limits.

## Baseline Comparisons
The paper compares its proposed methods against a highly complete set of baselines:
1. **Unmerged FP16 Experts ($93.85\%$):** Represents the upper-bound task performance before any merging or quantization.
2. **Naive FP16 Merge ($66.65\%$):** Establishes the standard task arithmetic full-precision merging ceiling.
3. **Naive Re-Quantized (Naive-RQ):** Represents the unmitigated pipeline.
4. **Quantize-then-Merge (Q-then-M):** Separately quantizes base and adapter weights, serving as the co-existence dual-path inference baseline.
5. **AdaMerging (PH-Q):** Optimizes coefficients in full precision followed by post-hoc quantization, isolating whether optimization must be quantization-aware.

*Critique on Merging Baselines:* While uniform task arithmetic (Naive FP16 Merge) is a standard baseline, the massive gap between unmerged experts ($93.85\%$) and the merged model ($66.65\%$) indicates severe task interference in weight space. The paper would have been strengthened by evaluating more advanced full-precision merging baselines (such as TIES-Merging or DARE) to see if they are more or less sensitive to re-quantization.

## Do the Results Support the Claims?
Yes, the quantitative results provide strong empirical support for the authors' core claims:

1. **Quantization Granularity Bifurcation:**
   Supported by Tables 2, 3, and 4, Naive-RQ is nearly lossless under per-channel configurations (e.g., in Table 3, INT4 Symmetric Per-Channel drops only $1.80\%$ mean accuracy compared to FP16). The catastrophic collapse is isolated to INT4 Per-Tensor (Table 5), where Naive-RQ drops to $56.75\%$.

2. **SAWS Performance and Limits:**
   In Table 3 (INT4 Symmetric Per-Channel), SAWS achieves $67.80\%$ mean accuracy (exceeding the $66.65\%$ FP16 ceiling), demonstrating that its selective task-vector boosting acts as a valuable regularizer. However, under INT4 Per-Tensor (Table 5), SAWS achieves $56.40\%$, which is slightly worse than Naive-RQ ($56.75\%$), confirming the authors' claim about the limitations of global scaling under aggressive non-smooth per-tensor constraints.

3. **TTA Robustness and Instability (QA-ACS vs. AdaMerging):**
   In Table 3, AdaMerging ($68.80\%$) and QA-ACS ($68.00\%$) achieve the best overall results. Under INT4 Per-Tensor (Table 5), unconstrained QA-ACS exhibits local instability, with MNIST dropping to $37.80\%$ (below Naive-RQ's $42.00\%$). Table 9 in Appendix A.5 successfully validates that adding supervised labels (Supervised QA-ACS) or $L_2$ regularization (Regularized QA-ACS) completely resolves this instability, boosting mean accuracy to $60.63\%$.

4. **Decoupling Task Interference from Quantization Noise:**
   Table 8 (Individual Unmerged Quantized Experts Control Experiment) is the most elegant experimental result in the paper. It shows that under INT4 Symmetric Per-Channel, individual experts achieve $93.15\%$ (a negligible $0.70\%$ drop from FP16). This definitively proves that the low performance of the merged quantized model is driven by pre-existing weight-space task interference, NOT by quantization noise. Conversely, under INT4 Per-Tensor, the expert performance drops to $82.95\%$ (a massive $10.90\%$ drop), proving that per-tensor grids destroy base representations independently of merging.

5. **Physical CPU Latency profiling:**
   Table 11 and Figure 4 show that for the tiny `vit_tiny` model, co-existence is competitive (976ms vs 1008ms for $K=1$) because the model fits in the CPU cache. This empirical profiling supports the cache-fitting vs. DRAM-latency bifurcation theory, confirming that for multi-billion parameter models (which do not fit in cache), weight-space merging is physically required to avoid linear DRAM transfer latencies.
