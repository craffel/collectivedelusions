# 1. Summary of the Paper

## Overview
This paper proposes **QP-Merge** (Quantization-Preserving Merging), a training-free framework designed to enable low-bit post-training quantization (PTQ) of merged multi-task deep neural networks. While model merging is an effective way to combine multiple specialized task models without retraining, standard PTQ (e.g., INT4 or INT8) applied to merged models leads to catastrophic performance degradation. The authors identify two primary drivers of this failure:
1. **Heavy-tailed task outliers:** Subtracting base weights from fine-tuned weights ($\Delta W_t$) highlights highly sparse, high-magnitude parameter updates representing specialized features. Naive uniform quantization stretches the symmetric quantization scale factor $S_c$ to accommodate these outliers, compressing standard-range weights into very few bins and introducing severe quantization noise.
2. **Activation scale mismatches:** Because different task models are fine-tuned on distinct data distributions, their internal intermediate activation scales are fundamentally mismatched. Linearly blending task vectors under a fixed global coefficient ignores these task-specific scale discrepancies, causing severe representation misalignment after quantization.

## Core Methodology
QP-Merge addresses these challenges through two synergistic, lightweight techniques:
- **Outlier-Residual Decoupling (ORD):** Identifies and isolates the top $\le 1\%$ (or 0.5%) highest-magnitude weight updates (outliers) from each task vector. These outliers are stored in a highly sparse, high-precision FP16 tensor ($W_{\text{outlier}}$), while the range-bounded dense remainder ($W_{\text{dense}}$) is quantized to INT4 or INT8. This prevents weight-range stretching and preserves low-bit precision for the bulk of the parameters.
- **Quantization-Error Aware Scale Calibration (QE-Calib):** Jointly optimizes layer-wise diagonal weight scaling parameters $D_l$ and merging coefficients $\lambda$ over a tiny, completely unlabeled set of $M = 128$ domain samples (64 from each task domain). It runs 100 steps of Adam optimization to minimize the activation reconstruction mean-squared error (MSE) relative to the unquantized FP32 merged model's final embeddings.

During inference, layer computation is executed as a hybrid path:
$$Y = X \cdot Q_b(W_{\text{quantized\_base}}) + X \cdot W_{\text{outlier}}$$
where the dense part is computed via optimized low-bit integer GEMM kernels (e.g., INT4/INT8 Tensor Cores) and the sparse outlier part is computed via Sparse Matrix-Matrix Multiplication (SpMM) kernels.

## Key Experimental Findings
The authors evaluate QP-Merge on dual-task vision classification (MNISTVal and SVHNVal) using a pre-trained **ViT-B-32** model.
- **Baseline Results:** 
  - Unquantized FP32 Merged Bound (Uniform): Avg Accuracy of **94.93%** (99.14% MNISTVal, 90.72% SVHNVal).
  - Unquantized FP32 Merged Bound (Optimized): Avg Accuracy of **95.12%** (99.04% MNISTVal, 91.20% SVHNVal).
  - Naive INT4 Quantization: Drops catastrophically to **91.51%** (a 3.61% overall drop, with SVHNVal dropping by 6.40%).
- **QP-Merge INT4 Performance:** Recovers over 88% of the performance drop relative to the optimized baseline, achieving an average accuracy of **94.70% $\pm$ 0.13%** (recovering SVHNVal to 90.37% $\pm$ 0.22%). It outperforms a strong optimization-based SmoothQuant baseline (94.23%).
- **QP-Merge INT8 Performance:** Achieves **95.14% $\pm$ 0.03%** average accuracy, which is virtually lossless (within 0.02% of the optimized FP32 baseline) and actually beats the unquantized uniform baseline by +0.21%.
- **Ablation Studies:** Confirms that both QE-Calib and ORD are highly synergistic. For INT4, removing QE-Calib drops average accuracy to 91.09%, while removing ORD drops it to 94.49%.
- **Sensitivity Sweeps:** 
  - Sweeping outlier threshold $\gamma$ shows an optimal sweet spot at 0.5% - 5.0% density, with 0.5% achieving 94.74% accuracy.
  - Sweeping calibration set size $M$ demonstrates extreme data efficiency, with even $M=16$ unlabeled samples achieving 94.24% average accuracy.
- **Hardware Profiling:** Showcases a **3.77$\times$ VRAM compression ratio** for a representative linear layer in INT4 mode. Physical GPU latency profiling shows a latency of 60.92 $\mu$s (an overhead of 50.44 $\mu$s over FP16), which the authors honestly analyze as being driven by PyTorch's kernel launch overhead rather than computational complexity.
