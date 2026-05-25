import random

# Ten research ideas
ideas = [
    {
        "title": "VAAM-Merge: Variance-Aware Adaptive Momentum for Test-Time Model Merging",
        "motivation": "Test-time adaptation streams are non-stationary and noisy. Fixed learning rates for merging weights either adapt too slowly to task shifts or overfit to corruptions.",
        "method": "Track the running variance of prediction entropy. Use this variance to dynamically adjust the learning rate/momentum of layer-wise merging coefficients on-the-fly (higher step size at task boundaries, lower within stable tasks).",
        "expected_results": "Faster recovery from task switches and greater stability under OOD corruptions compared to standard gradient descent on merging weights.",
        "impact": "Improves real-time adaptability of merged models in dynamic and unpredictable environments."
    },
    {
        "title": "CAbA-Merge: Contrastive Anchor-Based Alignment for Teacher-Free TTMM",
        "motivation": "In teacher-free merging (S2C-Merge), classification heads are kept frozen to prevent decision boundary collapse, but this limits the model's ability to adapt to intra-task domain shifts.",
        "method": "Pre-compute class centroid representations (anchors) from expert validation sets. At test-time, optimize both merging weights and classification heads by minimizing a contrastive loss between test features and these anchors.",
        "expected_results": "Enables safe classification head adaptation without decision boundary collapse or teacher models in memory, surpassing frozen-head baselines on OOD shifts.",
        "impact": "Breaks the Teacher-Overhead Paradox while still allowing representation tuning."
    },
    {
        "title": "B-TTMM: Bayesian Test-Time Model Merging",
        "motivation": "Deterministic updates on small, noisy test-time batches are prone to sharp local minima and poor calibration.",
        "method": "Model layer-wise merging weights as a variational distribution (e.g., Gaussian) and optimize its parameters using Monte Carlo dropout or variational inference under a self-supervised prior.",
        "expected_results": "Better uncertainty estimation (calibration), lower prediction entropy on unseen tasks, and increased robustness to noisy/corrupted test batches.",
        "impact": "Introduces probabilistic reasoning to test-time model merging, improving safety."
    },
    {
        "title": "CAT-Merge: Cross-Attention Task Routing for Dynamic Multi-Task Model Merging",
        "motivation": "Existing test-time model merging assumes task boundaries are known or sequential, but real-world streams have blurred or unknown task boundaries.",
        "method": "Introduce a lightweight cross-attention router that takes input batch features and outputs a soft task routing distribution, which initializes and guides the merging coefficients.",
        "expected_results": "High multi-task performance on mixed/fuzzy streams without explicit task boundaries or domain labels.",
        "impact": "Enables realistic, boundary-free test-time model merging."
    },
    {
        "title": "CG-TTA: Curriculum-Guided Test-Time Weight Consolidation",
        "motivation": "Elastic weight consolidation (EWC) is typically static, but during test-time adaptation, the optimal balance between expert retention and adaptation changes dynamically.",
        "method": "Scale the EWC penalty factor dynamically based on a curriculum of prediction confidence (entropy) and adaptation time, permitting more flexibility as the stream stabilizes.",
        "expected_results": "Combines rapid initial adaptation to domain shifts with robust long-term parameter protection.",
        "impact": "Enhances the Pareto frontier of adaptation speed and forgetting prevention."
    },
    {
        "title": "OSP-Merge: Orthogonal Subspace Projection for Teacher-Free TTMM",
        "motivation": "Updating merging coefficients or heads on Task A can cause interference and performance degradation on other tasks (Task B).",
        "method": "Compute SVD of expert features to define task-specific orthogonal subspaces. During TTA, project the gradients of the merging weights/heads onto the active task's subspace.",
        "expected_results": "Virtually zero interference or degradation across tasks during alternating sequential streams.",
        "impact": "Ensures task isolation during dynamic, multi-task adaptation."
    },
    {
        "title": "Sparse-TTMM: Sparsity-Regularized Gated Test-Time Model Merging",
        "motivation": "Dense layer-wise model merging activates all experts for every layer, which increases interference and unnecessary blending.",
        "method": "Apply L1 regularization or a Concrete-distribution gate to the layer-wise merging weights to encourage sparse expert utilization per layer.",
        "expected_results": "Learns task-specific, sparse routing of expert layers on-the-fly, reducing interference and maintaining clean features.",
        "impact": "Provides interpretability and efficiency in layer-by-layer expert selection."
    },
    {
        "title": "SAT-Free: Sharpness-Aware Teacher-Free Consistency Merging",
        "motivation": "S2C-Merge (teacher-free) updates are sensitive to sharp local minima, while SATA-SBF (sharpness-aware) requires heavy teachers in memory.",
        "method": "Integrate Sharpness-Aware Minimization (SAM) into a teacher-free self-supervised objective (entropy minimization + consistency regularization) to optimize merging weights.",
        "expected_results": "Flatter loss landscape for merging coefficients, leading to significantly better OOD generalization without teacher overhead.",
        "impact": "Brings the benefits of flatness-regularized optimization to teacher-free model merging."
    },
    {
        "title": "EMA-TTMM: Exponential Moving Average Coeff-Smoothing for TTMM",
        "motivation": "Gradient updates on small test-time batches cause high variance and instability in the merging weights lambda from batch to batch.",
        "method": "Maintain an EMA of the merging weights lambda. Use the EMA weights for the forward pass/inference, while updating the online weights using self-supervised objectives.",
        "expected_results": "Drastically smoother performance curves over time and higher average multi-task accuracy on small batch sizes.",
        "impact": "Provides a simple, lightweight plug-and-play mechanism to stabilize any test-time merging framework."
    },
    {
        "title": "MT-AT-Merge: Multi-Teacher Distillation under Adaptive Temperature",
        "motivation": "In multi-teacher distillation-based model merging, soft labels from irrelevant teachers introduce noise and degrade performance.",
        "method": "Dynamically scale each expert teacher's distillation temperature based on its confidence (entropy) on the current batch, filtering out irrelevant expert guidance.",
        "expected_results": "Improved distillation quality and higher multi-task accuracy in complex multi-expert settings.",
        "impact": "Refines expert-guided model merging by filtering noisy expert advice."
    }
]

# Write ideas to progress.md
with open("progress.md", "w") as f:
    f.write("# Research Progress Log\n\n")
    f.write("## Phase 1: Foundation (Read & Formulate)\n\n")
    f.write("### Identified General Theme\n")
    f.write("Test-Time Adaptation (TTA) and Test-Time Model Merging (TTMM) for multi-task learning, OOD generalization, and teacher-free deployment.\n\n")
    f.write("### Ten Generated Research Ideas\n\n")
    for i, idea in enumerate(ideas):
        f.write(f"#### Idea {i+1}: {idea['title']}\n")
        f.write(f"- **Motivation:** {idea['motivation']}\n")
        f.write(f"- **Method:** {idea['method']}\n")
        f.write(f"- **Expected Results:** {idea['expected_results']}\n")
        f.write(f"- **Impact:** {idea['impact']}\n\n")

# Choose one idea using pseudo-random number generator
random.seed(2026)
selected_idx = random.randint(0, len(ideas) - 1)
selected_idea = ideas[selected_idx]

print(f"Selected Idea Index: {selected_idx + 1}")
print(f"Selected Idea Title: {selected_idea['title']}")

with open("progress.md", "append" if False else "a") as f:
    f.write("### Selection of Research Idea\n")
    f.write(f"Selected using a pseudo-random number generator with seed 2026.\n\n")
    f.write(f"**Selected Idea:** Idea {selected_idx + 1}: {selected_idea['title']}\n")
    f.write(f"- **Motivation:** {selected_idea['motivation']}\n")
    f.write(f"- **Method:** {selected_idea['method']}\n")
    f.write(f"- **Expected Results:** {selected_idea['expected_results']}\n")
    f.write(f"- **Impact:** {selected_idea['impact']}\n\n")
    f.write("### Improving the Chosen Idea (Iterate)\n")
    f.write("We can improve the selected idea by integrating concepts from the other submissions (e.g. C-TMM from submission1, EWC-TTA from submission7) to create a highly robust, unified, and mathematically principled approach.\n\n")
    f.write("### Final Project Hypothesis and Rationale\n")
    f.write("To be defined after selection.\n")
