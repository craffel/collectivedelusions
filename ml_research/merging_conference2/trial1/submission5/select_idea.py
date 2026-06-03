import random

# Use a fixed seed for reproducibility
random.seed(42)

ideas = [
    "Quantization-Robust Model Merging via Outlier Smoothing (QR-Merge)",
    "Compute-Efficient Feature-Distribution Alignment for Merged Models (CE-FDA)",
    "Budget-Constrained Sparsity-Aware Merging (BC-SAM)",
    "Out-of-Distribution Robust Model Merging (OOD-Merge)",
    "Inference-Latency Optimized Layer-Wise Merging (ILO-Merge)",
    "Task Vector Quantization for Multi-Task Merging (TV-Quant)",
    "Post-Hoc Uncertainty Calibration for Merged Classifiers (PUC-Merge)",
    "Closed-Form Isotropic Scaling for Multi-Task Integration (CF-Isotropic)",
    "Extreme Outlier Pruning for Quantized Task Arithmetic (EOP-QTA)",
    "Low-Memory Continual Weight Merging (LM-CWM)"
]

selected_idx = random.randint(0, len(ideas) - 1)
selected_idea = ideas[selected_idx]

print(f"Selected index: {selected_idx}")
print(f"Selected idea: {selected_idea}")
