# Experiment Check

## Critical Evaluation of the Experimental Setup & Datasets
- **Backbone & Configuration:** The authors use a very small `vit_tiny_patch16_224` backbone (5.7M parameters) with LoRA rank 8 targeting $W_q, W_k, W_v$. This is a major limitation, as real-world LLMs and Vision-Language models operate at the scale of billions of parameters, and quantization/merging dynamics can change at scale. However, the multi-axial evaluation (Tables 2-6) is thorough, and the base-weight Frobenius error analysis on `vit_base` (Table 1) provides a useful scaling indicator.
- **Datasets:** The four datasets used (MNIST, FashionMNIST, CIFAR-10, SVHN) represent highly disparate domains. This diversity is excellent for creating a challenging multi-task scenario with high task interference in weight space.

## Baselines
The baselines are comprehensive and well-designed:
1. **Unmerged FP16 Experts (Ceiling):** Establishes the upper bound of separate task models before merging ($93.85\%$).
2. **Naive FP16 Merge:** Establishes the full-precision merging ceiling ($66.65\%$), showing that weight-space merging itself causes a massive performance drop (due to representation interference) before any quantization is applied.
3. **Naive Re-Quantized (Naive-RQ):** Represents the standard unmitigated deployment pipeline, acting as a crucial baseline to measure the "silence" effect.
4. **Quantize-then-Merge (Q-then-M):** Represents the dual-path co-existence baseline, measuring performance without weight-space merging.
5. **AdaMerging (PH-Q):** Optimizing blending coefficients in FP16 followed by post-hoc quantization, isolating whether quantization-aware optimization is necessary.
6. **Individual Unmerged Quantized Experts:** Isolating quantization noise on separate models (Table 7).

## Support for Claims
The experimental results strongly support the paper's claims:
1. **Re-Quantization Silence:** Supported by Table 6 (INT4 Symmetric Per-Tensor), where Naive-RQ drops from $66.65\%$ to $56.75\%$, and MNIST accuracy drops to $42.00\%$ (a massive drop from the unmerged ceiling).
2. **Quantization Granularity Bifurcation:** Strongly supported by Tables 3, 4, 5, and 6. Naive-RQ is nearly lossless under standard per-channel grids, dropping only $0.30\%$ in INT8 and $1.80\%$ in INT4 symmetric. Catastrophic collapse only occurs under per-tensor grids ($56.75\%$). This proves that "Re-Quantization Silence" is highly localized to sub-optimal per-tensor configurations.
3. **Double Quantization Format-Shift Noise:** Supported by Table 1, showing that transitioning base weights from NF4 to linear INT8/INT4 increases relative weight-space reconstruction error by up to $29.42\%$ on ViT-Base and $16.46\%$ on ViT-Tiny. This proves that format transitions introduce significant representational noise.
4. **Limits and Fragility of Proposed Mitigations:**
   - **SAWS:** Under per-tensor constraints, SAWS achieves $56.40\%$ accuracy (Table 6), which is actually *worse* than Naive-RQ ($56.75\%$). This confirms that its uniform global scaling multiplier is sub-optimal under aggressive per-tensor noise, failing when it is most needed.
   - **QA-ACS:** Under per-tensor grids, QA-ACS drops performance on MNIST to $37.80\%$ (Table 6) due to unsupervised entropy collapse under severe discretization noise.
5. **Decoupling Quantization Noise from Task Interference:** Supported by Table 7. Direct quantization of individual experts under per-channel grids shows almost zero degradation (mean accuracy of $93.15\%$ in INT4 symmetric PC vs. $93.85\%$ FP16 ceiling). This empirically proves that the drop in the merged model's performance under per-channel grids is due entirely to pre-existing weight-space task interference ($66.65\%$ ceiling), and NOT to quantization erasure.

## Minimalist Perspective on Results
The experimental results are highly revealing through a **Minimalist** lens:
- **Over-Engineering is Exposed:** The results prove that the complex scaling (SAWS) and test-time optimization (QA-ACS) methods are highly fragile and completely unnecessary in standard deployment scenarios.
- **The Simple Approach Wins:** Simply using per-channel quantization (which is standard and simple) is virtually lossless (losing only $0.15\%$ to $1.8\%$ accuracy), and requires absolutely zero added complexity, zero scaling constants, zero test-time optimization steps, and zero calibration datasets.
- **Praise for Empirical Honesty:** The authors are highly commended for including these results transparently in the paper, rather than trying to hide the limits of their own proposed methods to present an artificial SOTA. This rigorous empirical honesty prevents the field from chasing over-engineered and fragile methods.
