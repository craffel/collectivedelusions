# 3. Soundness & Methodology

This section provides a rigorous critique of the technical soundness of the QP-Merge framework, focusing on methodological clarity, potential technical flaws, and reproducibility from a deployment engineer's perspective.

---

## Clarity of Description
The methodology is exceptionally well-described and mathematically structured.
- The standard model merging formulation (Eqs. 1–4) and the quantization operations (Eqs. 5–6) are laid out clearly.
- The decomposition of task vectors into sparse and dense elements (Eqs. 7–11) is mathematically clean.
- The explanation of column-wise diagonal scaling via right-multiplication ($D_l$) in Eq. 13 is precise, aligning with standard PyTorch linear layer dimensions ($d_{\text{out}} \times d_{\text{in}}$).
- The trade-offs and physical details of the architecture (such as the CSR/COO sparse formats and SpMM kernel deployment) are clearly addressed.

---

## Appropriateness of Methods
- **Outlier-Residual Decoupling (ORD):** Decoupling high-magnitude task vector updates is an extremely appropriate and direct response to the "stretched quantization scales" issue. Isolating the top $\le 1\%$ updates allows symmetric per-channel quantization to achieve tight, high-precision grids for the remaining 99% of parameters.
- **End-to-End Latent Alignment:** For vision encoders (like ViT-B-32), aligning the final output embedding space via MSE minimization (Eq. 12) is highly appropriate. It directly optimizes downstream task performance by ensuring semantic similarity to the unquantized FP32 model.

---

## Potential Technical Flaws & Practitioner Concerns

### 1. Risk of Representation Drift (Lack of Mathematical Equivalence)
Standard PTQ scale search (e.g., SmoothQuant) uses mathematically equivalent weight scaling by applying the inverse factor to activations during runtime ($Y = (X D^{-1})(D W)$). 
QP-Merge permanently scales the weight updates ($W_{\text{base}} + \Delta W \cdot D_l$) **without** adjusting activations. The authors argue that this is a necessary approximation because different merged tasks have conflicting activation ranges, preventing a single equivalent scaling factor. 
While their empirical results show that the model generalizes well on MNIST/SVHN, changing weight scales without inverse activation adjustments alters the unquantized mapping. For more complex, high-dimensional manifolds (such as LLM generation or dense visual segmentation), this non-equivalent scaling poses a high risk of **representation drift** or localized overfitting to the 128 calibration samples.

### 2. High PyTorch Latency Overhead (The "Deployment Bottleneck")
While the authors showcase 3.77$\times$ VRAM compression, their physical GPU profiling reveals a major deployment hurdle:
- **FP16 baseline latency:** 10.48 $\mu$s
- **QP-Merge INT4 (0.5% outliers) latency:** 60.92 $\mu$s (**5.8$\times$ slower than FP16**)
This massive slowdown in standard PyTorch is caused by CUDA kernel launch overhead from sequentially running separate dense INT4 and sparse FP16 SpMM operators at batch size 1. 
The authors claim that compile-time optimization tools (TensorRT, Triton) can fuse these kernels and eliminate this overhead. However, they do not provide or demonstrate a fused kernel. For a practitioner, deploying a model that is 5.8$\times$ slower out-of-the-box is a major negative, severely limiting its "instant" deployability.

### 3. Scalability of Outlier Density
If QP-Merge is scaled to $T$ tasks (e.g., an 8-task vision suite), and we extract 1% outliers per task, the combined outlier matrix density could scale up to $T\%$ (or $8\%$) if the outliers are disjoint. 
While the authors propose a "global thresholding scheme" in their future work to bound the density to a fixed 1%, they did not evaluate or implement it. If outlier density scales past 2-3%, the SpMM execution overhead will compound rapidly, eroding both the memory bandwidth benefits and execution speeds.

---

## Reproducibility
The methodology is highly reproducible. The paper specifies:
- Base model (`ViT-B-32` from OpenAI / timm).
- Datasets (MNIST and SVHN) and fine-tuning hyperparameters.
- Calibration size ($M=128$), iterations (100), learning rate ($1\times 10^{-3}$), and optimizer (Adam).
- Mask thresholds ($\gamma \in [0.99, 0.995]$).
The mathematical descriptions are detailed enough for an engineer to replicate the pipeline in PyTorch.
