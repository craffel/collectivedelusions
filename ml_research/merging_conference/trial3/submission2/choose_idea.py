import random

ideas = [
    {
        "id": 1,
        "title": "Elastic Fisher-Guided Spectral Test-Time Adaptation for LoRA Merging",
        "description": "Computes diagonal Fisher Information on the unmerged LoRA expert heads. During test-time adaptation, uses this Fisher Information to regularize the test-time adaptation of the merged classifier heads and the merging coefficients, preventing prediction collapse while optimizing sharpness."
    },
    {
        "id": 2,
        "title": "Curvature-Aware Synergistic Test-Time Model Merging (CA-SyMerge)",
        "description": "Employs dynamic test-time curvature estimation (e.g., through a lightweight diagonal Fisher) to guide the layer-wise merging coefficients. Layers with high curvature are merged conservatively, while flat layers are merged more aggressively, combined with expert-guided self-labeling."
    },
    {
        "id": 3,
        "title": "Adaptive Orthogonal Procrustes Test-Time Adaptation (AOP-TTA)",
        "description": "Extends OrthoMerge to test-time adaptation. Instead of optimizing the merging coefficients in Euclidean space, it performs gradient descent directly on the manifold of the orthogonal group at test-time using Procrustes alignment on the batch representations, ensuring the adapted model maintains geometric consistency."
    },
    {
        "id": 4,
        "title": "Sharpness-Aware Self-Labeling with SOSR for Robust TTA (SASS-TTA)",
        "description": "Combines Soft-Orthogonality Spectral Regularization (SOSR) and Sharpness-Aware Minimization (SAM) in an online self-labeled TTA loop. It uses the unmerged experts to compute confident pseudo-labels and performs SAM updates restricted to the spectral-regularized subspace, maintaining low rank compatibility."
    },
    {
        "id": 5,
        "title": "Fisher-Regularized Procrustes Alignment for Deep Model Merging (FRPA)",
        "description": "A training-side regularizer that uses Fisher Information to weight the Procrustes alignment. Parameters with high Fisher Information (most important for a task) are constrained to be close to the orthogonal manifold of the pre-trained base model, while less important parameters are allowed to drift."
    },
    {
        "id": 6,
        "title": "Decoupled Sharpness-Aware Test-Time Classifier Adaptation (D-SATA)",
        "description": "Decouples the test-time adaptation of task classification heads and layer-wise merging coefficients. It uses different learning rates and distinct SAM perturbation radii for coefficients and heads, regularized by a soft bounded distance from the original unmerged experts."
    },
    {
        "id": 7,
        "title": "Spectral-Regularized Fisher-Guided TTA (SR-FG-TTA)",
        "description": "Uses a diagonal running Fisher Information of experts to selectively apply SOSR (Soft-Orthogonality Spectral Regularization). Tensors with high Fisher Information are strongly regularized towards orthogonality, preventing representation distortion under test-time gradient steps."
    },
    {
        "id": 8,
        "title": "Low-Rank Orthogonal Projection for Test-Time Adaptation (LROP-TTA)",
        "description": "Restricts the test-time gradients of LoRA merging coefficients to lie in the orthogonal complement of the experts' shared null space, preventing task-interference at the parameter level."
    },
    {
        "id": 9,
        "title": "Sharpness-Aware Fisher-Bounded Synergistic Merging (SAF-SyMerge)",
        "description": "A unified test-time framework that uses running diagonal Fisher estimates to scale SAM's perturbation step. In directions where the Fisher is large (indicating high sensitivity), the SAM perturbation is scaled down, preventing destructive parameter updates."
    },
    {
        "id": 10,
        "title": "Multi-Objective Pareto-Optimal Test-Time Adaptation (MO-TTA)",
        "description": "Formulates test-time model merging as a multi-objective optimization problem, dynamically balancing classification entropy, expert-guided self-labeling consistency, and orthogonality/spectral penalties on a per-batch basis."
    }
]

# Seed with a fixed value for perfect reproducibility
random.seed(2026)
selected_index = random.randint(0, len(ideas) - 1)
selected_idea = ideas[selected_index]

print(f"Total ideas: {len(ideas)}")
print(f"Selected Idea ID: {selected_idea['id']}")
print(f"Selected Title: {selected_idea['title']}")
print(f"Selected Description: {selected_idea['description']}")
