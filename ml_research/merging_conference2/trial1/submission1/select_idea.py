import random

ideas = [
    {
        "index": 1,
        "title": "Rigorous Evaluation of Self-Labeling under Severe Teacher Degradation in Test-Time Merging",
        "description": "Tests SyMerge's assumption that expert models can act as reliable teachers under severe distribution shifts/corruption, proving that self-labeling is highly sensitive to teacher quality and proposing a robust confidence-weighted filtering baseline."
    },
    {
        "index": 2,
        "title": "Demystifying Orthogonal Model Merging: Is Manifold Geometry Doing the Heavy Lifting?",
        "description": "Performs a rigorous ablation study on OrthoMerge to see if the performance gains actually come from complex Riemannian manifold math or simply from conflict-aware neuron-level decoupling, and proposes a simple Euclidean counterpart."
    },
    {
        "index": 3,
        "title": "Isotropic Merging: Feature Distortion or True Subspace Alignment?",
        "description": "Evaluates the assumption that flattening the singular value spectrum (isotropic merging) is beneficial, testing it under controlled feature distortion metrics and comparing it to simple selective low-rank projection."
    },
    {
        "index": 4,
        "title": "Re-evaluating SOTA Model Merging: The Hidden Hyperparameter Optimization Gap",
        "description": "Exposes how SOTA model merging papers compare their highly tuned methods against untuned Task Arithmetic/TIES baselines, and shows that heavily tuned baselines can match or exceed SOTA."
    },
    {
        "index": 5,
        "title": "Confounding Factors in Multi-Task Model Merging: The Role of Dataset Scale and Class Imbalance",
        "description": "Evaluates merging under severe class and dataset size imbalances, showing SOTA methods are highly sensitive to these and proposing a dataset-scale scaling baseline."
    },
    {
        "index": 6,
        "title": "Systematic Analysis of Initialization Discrepancy in Model Merging",
        "description": "Rigorously tests claims that disjoint-basin models (different seeds) can be merged via test-time adaptive or SVD methods, showing they fail in realistic settings."
    },
    {
        "index": 7,
        "title": "The Myth of Task Independence in Task Vectors",
        "description": "Analyzes the non-linear interactions and correlations between task vectors and develops a diagnostic metric for 'task vector entanglement' to explain performance collapse."
    },
    {
        "index": 8,
        "title": "Flawed Metrics in Continual Model Merging: Average Accuracy vs. Task-Specific Degradation",
        "description": "Introduces 'Worst-Case Forgetting' and 'Minimum Task Performance' metrics to expose that SOTA continual merging methods often achieve high average accuracy by completely sacrificing specific tasks."
    },
    {
        "index": 9,
        "title": "Does Sharpness-Aware Fine-Tuning Actually Improve Mergability? A Critical Re-evaluation",
        "description": "Tests whether SAIM's SAM fine-tuning actually improves 'mergability' via flatness, or if it simply acts as a regularizer reducing the update's norm (comparable to simple L2/weight clipping)."
    },
    {
        "index": 10,
        "title": "On the Sensitivity of Test-Time Adaptive Merging to Unlabeled Test Data Distribution",
        "description": "Evaluates the fragility of SyMerge and AdaMerging to class imbalance and covariate shifts in the test-time unlabeled batches, showing they can easily collapse."
    }
]

# Use a fixed seed for reproducible pseudo-random selection
seed = 42
random.seed(seed)
selected_idx = random.randint(0, len(ideas) - 1)
selected_idea = ideas[selected_idx]

print(f"Random seed used: {seed}")
print(f"Selected Idea Index: {selected_idea['index']}")
print(f"Selected Idea Title: {selected_idea['title']}")
print(f"Selected Idea Description: {selected_idea['description']}")
