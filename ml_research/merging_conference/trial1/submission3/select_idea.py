import random
import os

# 10 ideas
ideas = {
    1: {
        "title": "SAM-SLA: Sharpness-Aware Single-Layer Adaptation for Model Merging",
        "description": "Fine-tuning task-specific models using Sharpness-Aware Minimization (SAM) to find flatter minima, which acts as a stabilizer and amplifier for test-time single-layer adaptation (SLA), resulting in superior task synergy, faster adaptation, and higher data efficiency."
    },
    2: {
        "title": "Ortho-SLA: Orthogonal Test-Time Adaptation on Lie Algebra",
        "description": "Constraining test-time single-layer adaptation to the manifold of the orthogonal group using Lie algebra, preventing the destruction of hyperspherical energy during adaptation."
    },
    3: {
        "title": "Iso-SLA: Isotropic Single-Layer Adaptation via SVD",
        "description": "Adapting only the singular values of a single critical layer at test-time to achieve task synergy without altering the singular vectors (preserving the underlying representation directions)."
    },
    4: {
        "title": "SA-Ortho: Sharpness-Aware Orthogonal Merging",
        "description": "Combining sharpness-aware fine-tuning with orthogonal model merging to investigate if flat minima preserve orthogonal structure in weight space and improve Riemannian merging."
    },
    5: {
        "title": "SS-Iso: Self-Supervised Isotropic Merging at Test-Time",
        "description": "Dynamically balancing the singular value spectrum of merged models at test-time using self-supervised objectives on unlabeled test data, removing the need for labeled validation data."
    },
    6: {
        "title": "Decoupled Low-Rank Task Synergy",
        "description": "Merging LoRA adapters by decoupling them into orthogonal and residual components, and adapting the residual component using test-time self-labeling."
    },
    7: {
        "title": "Curvature-Guided Single-Layer Selection for SLA",
        "description": "Dynamically choosing which layer to adapt during test-time model merging based on the trace of the Fisher Information Matrix of each layer, instead of using a heuristic/fixed layer."
    },
    8: {
        "title": "Momentum-Preserving Manifold Merging for Continual Learning",
        "description": "Performing orthogonal model merging over sequential training checkpoints (continual merging) by tracking momentum trajectories in the Lie algebra."
    },
    9: {
        "title": "Contrastive Self-Labeling for Test-Time Synergy",
        "description": "Using a contrastive learning loss on unlabeled test data to guide single-layer adaptation during merging, preventing representational collapse."
    },
    10: {
        "title": "Fisher-Weighted Isotropic Merging (F-Iso)",
        "description": "Weighting the singular value spectrum balancing in SAIM using the diagonal Fisher Information of the task-specific models to prioritize task-critical directions."
    }
}

# Seed using the Slurm Job ID if available, otherwise 42
slurm_job_id = os.environ.get("SLURM_JOB_ID", "42")
try:
    seed = int(slurm_job_id)
except ValueError:
    seed = 42

random.seed(seed)
selected_idx = random.randint(1, 10)

print(f"Seed used (Slurm Job ID): {seed}")
print(f"Selected Idea Index: {selected_idx}")
print(f"Selected Title: {ideas[selected_idx]['title']}")
print(f"Selected Description: {ideas[selected_idx]['description']}")
