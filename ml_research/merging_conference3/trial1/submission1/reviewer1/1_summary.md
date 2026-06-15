# 1. Summary of the Paper

## Main Topic
The paper addresses the challenge of deploying merged, multi-task deep neural networks (specifically Vision Transformers) onto resource-constrained edge or mobile hardware. Model merging in high-precision (FP16 or FP32) combines task-specific weights to create a unified multi-task model without joint retraining. However, when standard post-training quantization (PTQ) is applied to compress these merged models to low-bit integers (INT4 or INT8), accuracy degrades catastrophically. This degradation is attributed to:
1. **Heavy-tailed task outliers** that stretch the uniform quantization scales, crushing normal-range weights into few bins.
2. **Activation scale mismatches** across different tasks due to distinct underlying feature distributions.

To bridge this gap, the authors propose **QP-Merge** (Quantization-Preserving Merging), which co-designs parameter merging and post-training quantization.

---

## Technical Approach
QP-Merge introduces two core training-free, unsupervised techniques:

1. **Outlier-Residual Decoupling (ORD):**
   - Identifies and isolates a very small fraction (top $\le 1\%$) of high-magnitude weight differences (outliers) from each task vector.
   - Routes these outliers to a sparse, high-precision (FP16) tensor $W_{\text{outlier}}$.
   - Quantizes the remaining outlier-free, tightly bounded dense weight matrix $W_{\text{dense}}$ to INT4 or INT8 using symmetric per-channel uniform quantization.
   - At inference time, the computation executes as a hybrid of a low-bit dense GEMM and an FP16 sparse matrix multiplication (SpMM).

2. **Quantization-Error Aware Scale Calibration (QE-Calib):**
   - Addresses activation scale conflicts using a small, completely unlabeled calibration dataset ($M=128$).
   - Optimizes layer-wise column-scaling diagonal parameters $D_l$ and learnable task coefficients $\lambda$ over 100 steps of Adam to minimize the mean squared error (MSE) between the latent embeddings of the unquantized FP32 merged model and the quantized hybrid model.
   - Right-multiplies $D_l$ directly to the dense task vector updates without requiring an inverse scaling on activations during inference, avoiding runtime activation scaling complexity.

---

## Key Findings & Empirical Results
- **Severe Degradation from Naive Quantization:** Naively quantizing a merged ViT-B-32 model to INT4 causes the average classification accuracy to drop by **3.61%** (with SVHN classification dropping by 6.40%).
- **Accuracy Recovery in INT4:** In 4-bit (INT4) mode, QP-Merge achieves an average accuracy of **94.70%** (across 3 random seeds), recovering over 88% of the drop relative to the unquantized optimized FP32 baseline and performing within **0.42%** of it. It outperforms a strong post-hoc SmoothQuant baseline (94.23%).
- **Lossless INT8 Merging:** In 8-bit (INT8) mode, QP-Merge achieves **95.14%** average accuracy, exceeding the uniform unquantized FP32 baseline (94.93%) and performing virtually lossless (+0.02% gain) compared to the optimized unquantized FP32 baseline.
- **Data & Param Efficiency:** The calibration generalizes robustly with as few as $M=16$ unlabeled samples, showing stable and monotonic convergence.

---

## Explicitly Claimed Contributions (with Evidence)
1. **Co-designed Merging and Quantization:** First framework to explicitly address joint heavy-tailed weight distributions of merged models.
2. **Outlier-Residual Decoupling (ORD):** Shown empirically to prevent quantization grid stretching. The ablation of *No ORD* in INT4 drops average accuracy from 94.52% to 94.49% (and 0.18% on the harder SVHN task).
3. **Quantization-Error Aware Scale Calibration (QE-Calib):** Demonstrated to align activation scales across tasks without labeled downstream data. Disabling calibration (*No QE-Calib*) in INT4 drops accuracy to 91.09%.
4. **Hardware-Friendly and Scalable Formulations:** Argues that the sparse path ($W_{\text{outlier}}$ density $\le 1.0\%$) and INT4/INT8 dense representations scale well, offering a theoretical 3.77$\times$ VRAM compression ratio and physical speedups on memory-bound edge processors.
