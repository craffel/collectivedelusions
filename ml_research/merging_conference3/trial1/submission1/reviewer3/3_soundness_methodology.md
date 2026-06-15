# Soundness and Methodology Evaluation

This document presents a rigorous, adversarial critique of the technical soundness, mathematical formulation, physical hardware claims, and reproducibility of the "QP-Merge" framework.

## 1. Mathematical Flaws and Non-Equivalent Scaling
The core of the Quantization-Preserving Scale Calibration (QE-Calib) rests on a mathematically inconsistent and non-equivalent formulation.

### A. Unequal Treatment of Base and Task Weights
In Equation 11, the diagonal scaling matrix $D_l$ is applied only to the dense task vector updates, leaving the pre-trained base weight $W_{l, \text{base}}$ completely unscaled:
$$W_{l, \text{hybrid}}(D_l, \lambda) = Q_b\left(W_{l, \text{base}} + \left( \sum_{t=1}^T \lambda_t \Delta W_{t, l, \text{dense}} \right) D_l \right) + \sum_{t=1}^T \lambda_t \Delta W_{t, l, \text{outlier}}$$
There is no mathematical justification for scaling only the task vector residual while leaving the base weight unscaled. Because the base weights $W_{\text{base}}$ represent the foundational representation space, multiplying only the task-specific adjustments by $D_l$ severely warps the relative scales between base features and task adjustments. This violates the core assumption of linear mode connectivity in parameter space, which underpins task arithmetic.

### B. Lack of Activation Inverse Scaling (Loss of Mathematical Equivalence)
In traditional activation-weight co-scaling (e.g., SmoothQuant), scaling weights channel-wise is mathematically neutralized by applying the inverse scaling factor to activations:
$$Y = (X D_l^{-1}) (D_l W)$$
This maintains exact mathematical equivalence in floating-point, ensuring no function drift occurs before quantization. 
In QP-Merge, the authors permanently multiply the weight updates by $D_l$ but **do not apply inverse scaling to the activations**. This means they have permanently altered the floating-point network's mapping.
- **The Core Illusion:** The authors frame QE-Calib as a "quantization scale calibration" technique. In reality, because mathematical equivalence is lost, it is acting as a **low-parameter gradient-based fine-tuning/adaptation step** on the calibration dataset.
- **Overfitting Risk:** For a ViT-B-32 model, optimizing a diagonal scaler $D_l$ of size $d_{\text{in}} \times d_{\text{in}}$ (768 parameters) for each of the approximately 4–6 linear layers per block across 12 blocks yields approximately **55,000 learnable parameters**. Optimizing 55,000 parameters over a tiny calibration set of only **128 samples** is a massive overfitting risk. While the authors claim this does not overfit on their toy digit-classification suites, this generalization is highly unlikely to hold if scaled to massive LLMs (e.g., LLaMA-7B, where this would represent ~900,000 parameters) or diverse multi-task suites, where localized representation drift and catastrophic forgetting of uncalibrated domains would occur.

---

## 2. Hard Physical Hardware & Latency Contradictions
The paper exhibits a massive disconnect between its high-level hardware marketing claims and its actual physical measurements.

### A. The 6x Latency Slowdown (600% Overhead)
The abstract claims that QP-Merge introduces **"near-zero storage or computational overhead on modern accelerator hardware."** However, Section 4.5's physical GPU latency profiling reveals:
- **FP16 baseline latency:** $10.48\ \mu\text{s}$
- **QP-Merge (0.5% outliers) latency:** $60.92\ \mu\text{s}$
This represents a **600% increase in forward pass latency (a 6x slowdown)**. For any real-world edge deployment under tight latency SLAs (which the authors highlight as their core motivation), a 6x slowdown is a complete dealbreaker.

### B. Speculative, Unimplemented Compilations
The authors brush this catastrophic slowdown aside by claiming it is merely "high-level PyTorch API overhead" and that "low-level, deployment-optimized execution engines (such as TensorRT, vLLM, or custom Triton kernels)... eliminate this."
- As a Critic, we must point out that the authors **did not implement or measure** any such custom fused Triton kernels or TensorRT compilations. Their defense is entirely speculative.
- In hardware execution at batch size 1 (the online serving constraint), running a dense INT4 GEMM followed by a sparse FP16 SpMM (via `torch.sparse.mm`) introduces severe kernel launch latencies, memory hierarchy bottlenecks (loading separate dense and sparse matrices into SRAM), and thread-level contention that custom compilers cannot easily hand-wave away.
- The analytical weight-transfer modeling in Section 4.5 is highly oversimplified, assuming that weight transfer latency scales perfectly linearly with weight size and completely ignoring the physical instruction-level overhead of sparse indexing and address calculations in COO/CSR formats.

---

## 3. Reproducibility and Experimental Gaps
The methodology is vague on several crucial details, which severely hampers reproducibility:

1. **The Head-Merging Mystery:**
   The paper evaluates on MNISTVal and SVHNVal. Both datasets represent digit classification (0-9). 
   - Did the fine-tuned task models share a single classification head, or did they have separate linear classification heads? 
   - If they shared a head, how was it fine-tuned without cross-task interference? 
   - If they had separate heads, how were they merged, evaluated, or routed during the joint inference phase? 
   The paper is completely silent on this critical architectural detail.
2. **Task Coefficient Discrepancy:**
   The methodology states: "We initialize ... $\lambda_t = 0.3$" for optimization. However, the uniform FP32 baseline uses $\lambda_t = 0.5$. The authors do not explain how the learnable task coefficients are bounded, regularized, or if they are prone to collapsing to trivial solutions (such as setting task weights to zero or exploding to extreme scales) during the 100-step representation alignment.
3. **Training Hyperparameters:**
   The authors provide the training epochs and learning rates for fine-tuning the ViT-B-32 checkpoints, but omit critical training configurations such as weight decay, batch size during fine-tuning, learning rate schedules, data augmentation, and the exact split of train/validation/test sets used, making full reproduction of their base checkpoints impossible.
