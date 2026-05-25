import random

# List of 10 novel research ideas on model merging, test-time adaptation, geometry, and sharpness
ideas = [
    {
        "id": 1,
        "title": "Sharpness-Aware Test-Time Adaptive Merging (SATT-Merge)",
        "description": "During test-time adaptation of model merging coefficients and classifiers on unlabeled data (as in SyMerge), enforce a sharpness-aware regularization. Specifically, perturb the merging coefficients and/or adapted classifiers with worst-case gradient-based noise to minimize a robust loss, guiding the merged model to a flatter minimum.",
        "expected_results": "Significantly better generalization to out-of-distribution (OOD) test data and corrupted test sets (e.g., ImageNet-C), with higher stability and lower variance during test-time adaptation.",
        "impact": "Addresses the key vulnerability of test-time adaptation (overfitting/instability on small test batches) by connecting test-time merging with loss landscape flatness."
    },
    {
        "id": 2,
        "title": "Orthogonal Test-Time Adaptive Merging (OrthoSyMerge)",
        "description": "Instead of optimizing model merging coefficients and classifiers in Euclidean space during test-time adaptation, perform joint adaptation on the Riemannian manifold of the orthogonal group using Lie algebra so(d) updates and Cayley transforms.",
        "expected_results": "Preserves weight and representation geometry (such as hyperspherical energy) during test-time adaptation, preventing the representation collapse often observed in training-free or standard test-time adaptive merging.",
        "impact": "Introduces geometry preservation to test-time model adaptation and merging, offering a principled way to adapt models without distorting pre-trained features."
    },
    {
        "id": 3,
        "title": "Isotropic Test-Time Adaptive Merging (Iso-SyMerge)",
        "description": "Integrate adaptive isotropic SVD balancing into the test-time adaptation loop. Dynamically balance the singular value spectrum of the task-specific classifiers or task vectors during self-labeled test-time adaptation to prevent dominant tasks from overshadowing weaker tasks.",
        "expected_results": "More balanced multi-task performance during test-time adaptation, especially in highly heterogeneous or many-task scenarios.",
        "impact": "Prevents representation bias and 'task hijacking' in unsupervised/test-time adaptive merging."
    },
    {
        "id": 4,
        "title": "Curvature-Guided Test-Time Model Merging (CG-Merge)",
        "description": "Use estimated Fisher Information Matrices (FIM) or diagonal Hessian values of individual expert models to weight the gradients of merging coefficients and task classifiers during test-time adaptation.",
        "expected_results": "Better retention of important task-specific parameters and structure during test-time adaptation, resulting in less catastrophic forgetting of general knowledge.",
        "impact": "Bridges curvature-aware fine-tuning methods (like EWC) with unsupervised test-time model merging."
    },
    {
        "id": 5,
        "title": "Self-Labeled Subspace-Boosted Merging (SL-SubMerge)",
        "description": "Identify principal task subspaces using SVD on task vectors (as in Task Singular Vectors) and perform self-labeled test-time adaptation strictly within these low-rank subspaces instead of the full parameter space.",
        "expected_results": "Much faster test-time adaptation with highly reduced parameter footprint, leading to better optimization stability and resistance to test-time overfitting.",
        "impact": "Demonstrates that test-time adaptation is most effective when restricted to the low-rank subspace of task-specific variations."
    },
    {
        "id": 6,
        "title": "Sharpness-Aware Orthogonal Merging (SA-Ortho)",
        "description": "Perform Orthogonal Model Merging (OrthoMerge) but incorporate a sharpness-aware loss directly on the Lie algebra representation. That is, search for a merged rotation in the Lie algebra that minimizes the worst-case sharpness of the merged model.",
        "expected_results": "Finds a merged rotation that lies in a flatter region of the joint loss landscape, enhancing out-of-distribution robustness.",
        "impact": "First method to integrate sharpness/flatness optimization directly with manifold-based model merging."
    },
    {
        "id": 7,
        "title": "Procrustes-Aligned Test-Time Adaptation",
        "description": "Align the representations or task-specific layers of different models using Orthogonal Procrustes before performing self-labeled test-time adaptation, ensuring starting points are functionally aligned.",
        "expected_results": "Significantly reduced alignment discrepancy and smoother test-time optimization curves.",
        "impact": "Highlights the importance of functional alignment prior to test-time adaptation."
    },
    {
        "id": 8,
        "title": "Isotropic Orthogonal Merging (Iso-Ortho)",
        "description": "Perform Orthogonal Model Merging to combine the rotational components of the models on the Lie group manifold, while applying isotropic SVD balancing to the residual linear components.",
        "expected_results": "More balanced singular value representation in residual features, leading to higher performance on long-tail tasks.",
        "impact": "A comprehensive geometric merging framework addressing both structural rotation and singular value scaling."
    },
    {
        "id": 9,
        "title": "Adaptive Flatness-Seeking Model Soups (AFS-Soups)",
        "description": "Evaluate multiple model soups (combinations of weights) on test data and adaptively select the soup that exhibits the flatest loss landscape on the test batch, without running gradient descent.",
        "expected_results": "Extremely fast, training-free test-time model selection that is robust to corrupted or out-of-distribution data.",
        "impact": "Provides a zero-training alternative to test-time adaptation that is guided by flatness metrics."
    },
    {
        "id": 10,
        "title": "Bayesian Synergistic Test-Time Adaptive Merging (Bayes-SyMerge)",
        "description": "Treat the joint optimization of merging coefficients and classifiers as a Bayesian inference problem during self-labeled test-time adaptation, using MC dropout to capture uncertainty of the teacher models.",
        "expected_results": "Higher robustness to noisy self-labels from incorrect teacher model predictions, particularly on ambiguous test samples.",
        "impact": "First Bayesian formulation of test-time adaptive model merging, addressing the fundamental limitation of self-labeling noise."
    }
]

# Write Phase 1 details and the 10 ideas to progress.md
header = """# Research Progress Log

## Phase 1: Foundation (Read & Formulate)

### Literature Synthesis
I have read the three provided papers:
1. **SyMerge (arXiv:2412.19098)**: Proposes test-time adaptive model merging by jointly optimizing merging coefficients and a task-specific layer using expert models as teacher guides for self-labeling. Avoids entropy-minimization instability.
2. **OrthoMerge (arXiv:2602.05943)**: Merges models on the orthogonal group manifold (Lie algebra so(d)) via magnitude correction. Employs Orthogonal Procrustes for non-OFT models to decouple weights into orthogonal (rotational) and residual (linear) updates.
3. **SAIM (Sharpness-Aware Isotropic Merging)**: Focuses on continual learning via sharpness-aware block coordinate descent (SA-BCD) fine-tuning to find flatter minima, and adaptive isotropic merging to balance the singular value spectrum.

### Ten Novel Research Ideas
"""

progress_content = [header]
for idea in ideas:
    progress_content.append(f"#### Idea {idea['id']}: {idea['title']}\n- **Core Idea**: {idea['description']}\n- **Expected Results**: {idea['expected_results']}\n- **Expected Impact**: {idea['impact']}\n")

# Selection via pseudo-random number generator
random.seed(2)  # Use seed 2
selected_idx = random.randint(0, len(ideas) - 1)
selected_idea = ideas[selected_idx]

selection_text = f"""### Pseudo-Random Selection
Using python `random.seed(2)` and `randint(0, 9)`, the selected idea is **Idea {selected_idea['id']}: {selected_idea['title']}**.

### Refined Idea & Hypothesis
- **Hypothesis**: By incorporating a **Sharpness-Aware** objective during the **Test-Time Adaptation** of model merging coefficients and task-specific classifiers, we can guide the adaptation toward flatter regions of the joint loss landscape on unlabeled test data. This flatness will dramatically improve robustness to distribution shifts, corrupted inputs (e.g., ImageNet-C), and noisy self-labels from teacher models.
- **Rationale**: 
  - Standard test-time adaptive merging (like SyMerge or AdaMerging) optimizes a proxy objective (e.g., self-labeling or entropy) on a small, unlabeled test batch. This process is highly prone to overfitting the small test batch, resulting in weight configurations that are sharp and brittle.
  - SAIM and other papers show that sharpness-aware optimization leads to flatter minima that are robust to parameter perturbations and show significantly better generalization.
  - SATT-Merge (Sharpness-Aware Test-Time Adaptive Merging) will perturb the model parameters (adapted classifier or merging coefficients) by a small adversarial step $\\epsilon$ during test-time adaptation, and minimize the worst-case self-labeled cross-entropy loss. This enforces flatness during the unsupervised test-time adaptation phase itself.
- **Experimental Plan**:
  - Implement a classification test-time adaptation model where we merge expert models (e.g. on MNIST, SVHN, CIFAR-10) using task-arithmetic.
  - Implement a self-labeled test-time adaptation loop (similar to SyMerge) where we jointly optimize merging coefficients and task classifiers.
  - Introduce sharpness-aware perturbation (SAM-like step) on the parameters during test-time adaptation.
  - Evaluate on clean and corrupted test sets (adding noise/corruption to simulate distribution shift) to demonstrate that Sharpness-Aware test-time adaptation achieves superior generalization and robustness.
"""

progress_content.append(selection_text)

# Write to progress.md
with open("progress.md", "w") as f:
    f.write("\n".join(progress_content))

print(f"Successfully generated 10 ideas and selected Idea {selected_idea['id']}: {selected_idea['title']}")
