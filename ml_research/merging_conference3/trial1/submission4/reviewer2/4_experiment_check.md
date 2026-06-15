# Experimental Check and Critical Evaluation

## Experimental Setup & Datasets
The experimental evaluation is highly structured and utilizes a standard, widely accepted benchmark suite:
- **Architecture:** Vision Transformer (ViT-B-32) with 86M parameters, which is a representative architecture for model merging research.
- **Datasets:** Eight diverse image classification datasets (SUN397, Stanford Cars, RESISC45, EuroSAT, SVHN, GTSRB, MNIST, and DTD) capturing different domains (natural, satellite, digits, traffic signs, texture).
- **Unlabeled TTA splits:** Utilizing 1000 randomly sampled unlabeled images with a standard batch size of 32 provides a realistic, unified testbed for post-hoc adaptation.

## Evaluation Baselines
The paper includes a robust, exhaustive set of baselines:
1. **Static Merging:** Task Arithmetic (TA), Ties-Merging, and OrthoMerge.
2. **Test-Time Adaptation (TTA):** AdaMerging, SyMerge, and Task Surgery, all evaluated under a standardized Synergy-Refinement Protocol (initialized at $\theta_{\text{TA}}$).
3. **Regularization Controls:** $L_2$ Weight Anchoring at TA, and FluidMerge with Spatial Laplacian viscosity.
4. **Optimization Controls:** Static TA + Head-Only Tuning (with frozen encoder).

This is a comprehensive and fair comparison that strictly isolates the impact of different components.

## Critical Evaluation of Empirical Claims & Results

### 1. Verification of the "Domain Shift Barrier"
The claim that post-hoc adaptation cannot easily reconstruct representations from scratch is strongly supported by the "Boundary Stress-Test" (Table 2). When starting from raw base weights $\theta_0$, all adaptation methods fail to exceed random-guess accuracy (~5%). The Expected Calibration Error (ECE) analysis (where ECE explodes to over 90%) further supports the authors' diagnosis: without aligned representation features, the highly flexible classification heads overfit to high-frequency noise, leading to extremely confident but incorrect predictions on validation data. This is a highly valuable, practical insight for practitioners.

### 2. Effectiveness of Fisher-Information Viscosity
The claim that Fisher Viscosity prevents representation collapse and stabilizes trajectory integration is fully supported by the comparative results in Table 1:
- **vs. Spatial Laplacian:** FluidMerge with Fisher Viscosity achieves **59.34%** accuracy and **7.18%** ECE, whereas the Spatial Laplacian baseline collapses to **54.76%** accuracy and **16.02%** ECE. This confirms that spatial coordinate smoothing over PyTorch index layouts triggers catastrophic representation tearing.
- **vs. L2 Weight Anchoring:** Fisher Viscosity outperforms standard $L_2$ weight anchoring (**58.48%** accuracy, **8.75%** ECE). The coordinate-wise, sensitivity-scaled updates of EWC are superior to a flat Euclidean penalty.

### 3. Practical Utility and Cost-Benefit Trade-off (The Practitioner's View)
While FluidMerge achieves state-of-the-art results, a critical practitioner analysis of the empirical gains reveals a severe cost-benefit limitation:
- **Small Absolute Margins:** FluidMerge (59.34%) only outperforms the frozen-encoder baseline **Static TA + Head-Only Tuning** (58.12%) by a narrow margin of **1.22%** absolute top-1 accuracy. It outperforms static, zero-compute Task Arithmetic (57.74%) by **1.60%**.
- **Prohibitive Computational Overhead:** As detailed in Table 3, achieving this 1.22%–1.60% improvement requires **20.5 minutes** of wall-clock time on a single NVIDIA A100 GPU and **14.8 GB** of GPU memory overhead per run. This is because full-encoder adaptation requires computing empirical Fisher coordinates and performing backpropagation through all 86M parameters of the ViT-B-32 image encoder across 100 epochs, while evaluating $K=8$ task teachers.
- **Inference/Deployment Mismatch:** In practical industrial settings, model merging is highly valued precisely because it is a *zero-compute*, instant deployment strategy. Spending 20.5 minutes of premium GPU time for a marginal ~1.2% accuracy improvement is highly unlikely to be practical, especially since 90% of the TTA adaptation benefit can be obtained by simply tuning the classification heads at zero encoder backpropagation cost (58.12% accuracy). 
- **Scale Limitations:** The evaluation is restricted to a small 86M-parameter ViT-B-32 model. For modern production models containing billions of parameters (e.g., Llama-3, CLIP ViT-L/ViT-G), computing Fisher matrices and backpropagating gradients for multiple tasks during test-time deployment would be completely intractable.

### 4. LoRA-based Adaptation Gaps
In Section 4.6, the authors present a parameter-efficient LoRA-FluidMerge wrapper as a future horizon, demonstrating a **64.1$\times$ parameter reduction** and a **1.32$\times$ backward-pass speedup**. However, the authors **do not report any classification accuracies or ECE results** for this LoRA-FluidMerge setup. Without showing whether the low-rank constraint maintains the 59.34% adaptation accuracy or triggers performance decay, this practical validation remains incomplete.
