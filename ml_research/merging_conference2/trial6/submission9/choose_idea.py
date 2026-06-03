import random

ideas = [
    "The Ultimate Empirical Merging Unification (UEMU)",
    "Empirical Robustness of Calibration under Severe Distribution Shift (ER-CSDS)",
    "Multi-Seed and Hyperparameter Sensitivity Landscape of Model Merging Calibration",
    "Task-Interference Localization and Empirical Layer-Wise Sensitivity Analysis (TILE-LSA)",
    "Empirical Validation of Data-Free vs. Data-Driven Calibration in Merged Models",
    "Cross-Architecture Empirical Generalizability of Merging Calibration",
    "Robust Multi-Task Mixture of Experts (MoE) via Empirical Routing Calibration (R-MoE)",
    "An Empirical Study of Activation Sparsity and Quantization in Merged and Calibrated Models",
    "Empirical Calibration of Model Merging under Scale: From ResNet-18 to ResNet-101",
    "A Unified Empirical Framework for Hybrid Spatial-Spectral Calibration"
]

random.seed(42)
selected_index = random.randint(0, len(ideas) - 1)
selected_idea = ideas[selected_index]

print(f"Selected Index: {selected_index}")
print(f"Selected Idea: {selected_idea}")
