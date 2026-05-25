import random

ideas = [
    "Idea 1: Sharpness-Aware Gating (SAG) - Combine SAM-TTMM's sharpness-aware optimization with routing-based merging.",
    "Idea 2: Continuous Sparsity-Aware Interpolated Routing (CSAIR) - Soft blending of Euclidean and Angular distances based on Hoyer sparsity.",
    "Idea 3: Bayesian Model Merging with Active Learning Rate Scaling (BMM-ALRS) - Formulate merging coefficients as a posterior and use particle-based Bayesian methods.",
    "Idea 4: Curvature-Aware Covariance Preconditioning (CACP) - Local Hessian-diagonal diagonal estimation to precondition merging coefficient updates.",
    "Idea 5: Moment-Preserving Soft Batch Normalization with Temporal Decay (MP-SBN-TD) - Extend soft MoG BN fusion with temporal decay for fast adaptation.",
    "Idea 6: Self-Supervised Prototype Adaptation (SSPA) - Adapt prototypes on-the-fly using a self-supervised contrastive loss.",
    "Idea 7: Cross-Attention Multi-Expert Feature Merging (CA-MEFM) - Dynamic cross-attention feature-level expert merging.",
    "Idea 8: Orthogonal Gradient Projection for Continual TTMM (OGP-CTTMM) - Gradient projection onto the orthogonal complement of past domain null-spaces.",
    "Idea 9: Contrastive Uncertainty-Guided Temperature Scaling (CUGTS) - Temperature scaling based on the ratio of nearest and second-nearest prototype distances.",
    "Idea 10: Sparsity-Aware Sharpness-Regularized Merging (SASRM) - Cohesive framework combining sparsity-aware hybrid routing (AHR) and sharpness-aware minimization (SAM)."
]

# Set a seed, e.g., 42
random.seed(42)
selected_index = random.randint(0, len(ideas) - 1)
print(f"Selected Idea Index: {selected_index}")
print(f"Selected Idea: {ideas[selected_index]}")
