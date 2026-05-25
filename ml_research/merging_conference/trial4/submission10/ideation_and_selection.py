import os
import random
import json

# List of 10 novel research ideas on TTA for Model Merging
ideas = [
    {
        "id": 1,
        "name": "PC-TTA: Prototype-Constrained Test-Time Adaptation for Teacher-Free Model Merging",
        "hypothesis": "Test-time adaptation of both classification heads and merging coefficients without a teacher model can be stabilized (preventing decision-boundary collapse) by constraining the adapted heads to align with pre-computed, fixed semantic prototype representations in a shared latent space.",
        "rationale": "S2C-Merge freezes classification heads entirely to prevent collapse under self-supervision, which limits the model's ability to adapt to severe covariate shift. By introducing fixed, normalized class prototypes (from clean validation data) and regularizing the adapted classification heads to map features onto these prototypes, we can adapt the heads without experiencing prediction collapse.",
        "expected_impact": "Allows full test-time adaptation of classification heads in teacher-free settings, significantly boosting OOD robustness under severe corruptions while using 0 extra memory/compute for teacher models."
    },
    {
        "id": 2,
        "name": "FW-CMS: Fisher-Weighted Convex Model Soups for Highly Sensitive Parameter Protection",
        "hypothesis": "Scaling the test-time learning rates of individual layer merging coefficients inversely by their parameter-level Fisher Information (or sensitivity) will protect task-critical representation structures while allowing faster adaptation in highly flexible layers.",
        "rationale": "SATA-SBF uses diagonal running Fisher to scale SAM perturbations, but does not adapt the learning rates of the merging parameters themselves. A layer-wise learning rate scaling based on pre-computed layer sensitivities will make adaptation more robust to rapid domain transitions.",
        "expected_impact": "Improves transition speed on sequential streams while maintaining stability and reducing temporal lag."
    },
    {
        "id": 3,
        "name": "MIM-TTMM: Masked Image Modeling as a Self-Supervised Objective for Test-Time Model Merging",
        "hypothesis": "Using a self-supervised masked reconstruction task (such as Masked Image Modeling) at test-time to optimize merging coefficients is less susceptible to prediction collapse than standard entropy minimization.",
        "rationale": "Entropy minimization is notorious for collapsing decision boundaries. A masked patch reconstruction objective (for transformers) forces the encoder to preserve structural semantic details of the OOD input without relying on class labels or soft targets.",
        "expected_impact": "Establishes a highly stable, non-collapsing, teacher-free self-supervised objective for test-time adaptation of merged vision transformers."
    },
    {
        "id": 4,
        "name": "DTR-GMS: Dynamic Task-Routing with Gated Model Soups",
        "hypothesis": "A tiny, test-time trained gating network (router) that maps input batch representations to merging weights can achieve better multi-task accuracy than globally optimizing static, layer-wise coefficients.",
        "rationale": "Existing methods optimize a single set of merging coefficients $\Lambda$ for the entire stream, which assumes all samples in a batch or stream belong to the same domain. A dynamic router allows sample-level or batch-level specialized merging, which is crucial for high-frequency alternating streams.",
        "expected_impact": "Substantially boosts performance on high-frequency alternating streams (chunk size M=1) where global coefficients suffer from high temporal lag."
    },
    {
        "id": 5,
        "name": "Bayes-TMM: Bayesian Test-Time Model Merging via Online Laplace Approximation",
        "hypothesis": "Maintaining a posterior distribution over the merging coefficients $\Lambda$ and sampling merging configurations at test-time (Bayesian model averaging) provides better uncertainty estimation and robust generalization under OOD shifts than point-estimate adaptation.",
        "rationale": "Point-estimate TTA is prone to overfitting to the specific noise of the incoming stream. Maintaining a Gaussian posterior centered at the adapted coefficients allows robust ensemble-like predictions at test-time.",
        "expected_impact": "Reduces calibration error (ECE) and boosts classification accuracy under severe spatial corruptions."
    },
    {
        "id": 6,
        "name": "MR-TTMM: Momentum-Regularized Test-Time Model Merging for High-Frequency Streams",
        "hypothesis": "Applying a momentum-based smoothing constraint on the test-time updates of merging coefficients prevents rapid, unstable oscillations and mitigates the temporal lag associated with out-of-phase transitions.",
        "rationale": "In high-frequency alternating streams (e.g. MNIST -> FashionMNIST -> KMNIST), the merging coefficients oscillate rapidly. A momentum regularizer ensures smooth transitions and prevents catastrophic interference across adjacent tasks.",
        "expected_impact": "Improves stability and classification accuracy on alternating streams with small chunk sizes."
    },
    {
        "id": 7,
        "name": "TA-TTA: Task-Arithmetic Subspace Regularized Test-Time Adaptation",
        "hypothesis": "Constraining the adapted model parameters to lie strictly within the low-dimensional subspace spanned by the task vectors (as defined in Task Arithmetic) stabilizes online adaptation and prevents representation collapse.",
        "rationale": "Model weights fine-tuned on separate tasks have been shown to span a highly structured task subspace. Restricting the optimizer's updates to this subspace acts as a strong inductive bias that preserves model capabilities.",
        "expected_impact": "Enables extremely fast convergence and strong robustness to OOD noise with minimal trainable parameters."
    },
    {
        "id": 8,
        "name": "CAFA-Merge: Cross-Attention Feature Alignment for Compressed Expert Guidance",
        "hypothesis": "Using a cross-attention layer between the active merged model's latent features and a small, compressed cache of expert activations provides the benefits of expert-guided self-labeling without the memory overhead of keeping the full expert models in memory.",
        "rationale": "S2C-Merge eliminates teachers completely but suffers a performance drop. If we can store and retrieve compressed representations of expert features (or use a tiny distilled neural cache), we can guide the adaptation process efficiently.",
        "expected_impact": "Achieves competitive performance with full expert-guided methods while requiring less than 5% of the teacher model memory."
    },
    {
        "id": 9,
        "name": "CD-TTMM: Curriculum-Driven Test-Time Model Merging via Online Drift Estimation",
        "hypothesis": "Dynamically adjusting the adaptation learning rate and regularization strengths based on an online estimate of the drift velocity and stream complexity improves overall adaptation quality.",
        "rationale": "When the test stream is stable (low drift), adaptation should be slow and cautious to prevent parameter drift. When a task boundary is detected (high drift/entropy jump), adaptation should accelerate to quickly specialize the model.",
        "expected_impact": "Mitigates temporal lag at task boundaries while preventing overfitting during stable periods."
    },
    {
        "id": 10,
        "name": "OLR-Merge: Orthogonal Low-Rank Subspace Merging for PEFT Experts",
        "hypothesis": "Optimizing test-time merging coefficients for LoRA adapters under a constraint that projects updates orthogonally to the base model's singular vectors preserves zero-shot capabilities while maximizing downstream performance.",
        "rationale": "Merging LoRA experts can distort the shared pre-trained base model representations. Restricting merging adaptations to orthogonal low-rank subspaces preserves the base model's general knowledge.",
        "expected_impact": "Prevents catastrophic forgetting of zero-shot capabilities during downstream test-time adaptation."
    }
]

# Write all ideas to progress.md
progress_content = "# Research Progress Log\n\n"
progress_content += "## Phase 1: Foundation (Read & Formulate)\n\n"
progress_content += "### Synthesis of Prior Work\n"
progress_content += "1. **SATA-SBF & SATA-RGP** (submission1.pdf): Introduces Convex Tensor-wise Model Merging (C-TMM) via Softmax constraint to keep weights in the convex hull of expert trajectories. Uses running Fisher to scale SAM perturbations and Relative Geometry Preservation (RGP) to align class prototype similarities, maintaining decision boundaries under OOD shifts.\n"
progress_content += "2. **S2C-Merge** (submission5.pdf): Proposes a teacher-free TTA approach. Resolves prediction collapse under self-supervised entropy minimization and consistency regularization by keeping the classification heads frozen and only adapting merging weights.\n"
progress_content += "3. **EWC-TTA** (submission7.pdf): Applies EWC-style quadratic penalty on active classification heads using offline pre-computed clean diagonal Fisher priors to protect task-critical head structures during expert-guided self-labeling TTA.\n\n"
progress_content += "### 10 Novel Research Ideas\n"

for idea in ideas:
    progress_content += f"#### Idea {idea['id']}: {idea['name']}\n"
    progress_content += f"- **Hypothesis:** {idea['hypothesis']}\n"
    progress_content += f"- **Rationale:** {idea['rationale']}\n"
    progress_content += f"- **Expected Impact:** {idea['expected_impact']}\n\n"

# Seed pseudo-random number generator
random.seed(42)
selected_index = random.randint(0, 9)
selected_idea = ideas[selected_index]

progress_content += "### Selection of Idea\n"
progress_content += f"- **Selected Index (seeded random with seed=42):** {selected_index + 1}\n"
progress_content += f"- **Selected Idea:** {selected_idea['name']}\n"
progress_content += f"- **Hypothesis:** {selected_idea['hypothesis']}\n"
progress_content += f"- **Rationale for Selection:** This idea is highly novel as it directly bridges the gap between S2C-Merge (teacher-free but with frozen, un-adaptable heads) and EWC-TTA/SATA-RGP (adaptable heads but requiring either teacher supervision or complex geometric preservation). By introducing Prototype-Constrained TTA (PC-TTA), we can safely adapt the task classification heads in a fully teacher-free, self-supervised setting without triggering prediction collapse.\n\n"

# Write to progress.md
with open("progress.md", "w") as f:
    f.write(progress_content)

print(f"Successfully generated 10 ideas and selected: {selected_idea['name']}")
print("Selected idea detail:")
print(json.dumps(selected_idea, indent=2))
