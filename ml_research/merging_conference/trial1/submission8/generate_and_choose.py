import random

# Seed the PRNG with our Slurm Job ID for reproducibility
random.seed(22158112)

ideas = [
    {
        "id": 1,
        "title": "Orthogonal Synergistic Merging (OrthoSymerge)",
        "hypothesis": "Merging the orthogonal components of fine-tuned expert models on the Riemannian manifold, while jointly optimizing their task-specific classifiers using unsupervised test-time self-labeling, will yield superior robustness to distribution shifts compared to Euclidean test-time adaptive merging.",
        "rationale": "Combining SyMerge's test-time self-labeling adaptation with OrthoMerge's geometry-preserving Riemannian manifold merging provides both representation-level stability (from orthogonal mapping) and task-level synergy (from classifier adaptation)."
    },
    {
        "id": 2,
        "title": "Sharpness-Aware Orthogonal Merging (SA-Ortho)",
        "hypothesis": "Applying a sharpness-aware perturbation directly on the Lie algebra representation so(d) during Orthogonal Fine-Tuning (OFT) will guide task adaptation toward flatter minima on the Riemannian manifold, significantly reducing parameter interference during subsequent merging.",
        "rationale": "SAIM proved that flatter minima in Euclidean space dramatically enhance merging performance. Extending sharpness-aware optimization to the Riemannian manifold of the orthogonal group will create geometrically stable, flat representations."
    },
    {
        "id": 3,
        "title": "Isotropic Orthogonal Model Merging (Iso-Ortho)",
        "hypothesis": "Performing SVD on the skew-symmetric Lie algebra matrix Q of expert models and adaptively balancing their singular value spectra to be more isotropic before Cayley reconstruction will prevent dominant task directions from causing representation collapse on the manifold.",
        "rationale": "Isotropic scaling in SAIM prevents task dominance in Euclidean space. Applying it to the Lie algebra representation Q within OrthoMerge ensures balanced rotation intensity across all merged tasks."
    },
    {
        "id": 4,
        "title": "Sharpness-Aware Self-Labeling Test-Time Adaptation (SA-SL-TTA)",
        "hypothesis": "Regularizing the flatness of the task-specific classifier during test-time adaptive merging via a sharpness-aware self-labeling loss will stabilize unsupervised adaptation and mitigate confirmation bias under severe distribution shifts.",
        "rationale": "SyMerge's self-labeling adaptation can be unstable when predictions are noisy. Sharpness regularization ensures the adapted classifier parameters converge to a flat region of the pseudo-labeled loss landscape, enhancing out-of-distribution robustness."
    },
    {
        "id": 5,
        "title": "Decoupled Orthogonal-Isotropic Merging (Ortho-Iso)",
        "hypothesis": "Decoupling fine-tuned model weights into orthogonal and residual components, then performing magnitude-corrected merging on the orthogonal group and adaptive isotropic SVD-based merging on the residual component, will optimize both structural rotation and additive updates.",
        "rationale": "Combining OrthoMerge's Orthogonal-Residual Decoupling (ORD) and SAIM's Isotropic Merging allows us to merge the clean structural (rotational) component on the manifold and separately balance the noisy residual (additive) component in Euclidean space."
    },
    {
        "id": 6,
        "title": "Synergistic Orthogonal Alignment (SynOrtho)",
        "hypothesis": "Actively optimizing the Lie algebra merging coefficients based on cross-task alignment metrics (pseudo-labeled cross-task performance) will foster positive task synergy on the Riemannian manifold.",
        "rationale": "OrthoMerge uses a static magnitude-corrected scaling factor. Optimizing these coefficients dynamically via a synergy-maximizing test-time objective will unlock the full potential of multi-task coordination on the manifold."
    },
    {
        "id": 7,
        "title": "Sharpness-Aware Test-Time Adaptive Merging (SA-AdaMerge)",
        "hypothesis": "Incorporating a sharpness-aware optimization step into the test-time adaptation of merging coefficients will prevent overfitting to the unlabeled test batch and improve generalization across temporal distribution drifts.",
        "rationale": "Unsupervised test-time optimization of merging weights can easily lead to sharp, overfitted local minima on the current batch. Flatness regularized adaptation ensures stable coefficients across changing environments."
    },
    {
        "id": 8,
        "title": "Isotropic Test-Time Adaptive Merging (Iso-AdaMerge)",
        "hypothesis": "Imposing an isotropic penalty on the singular values of the test-time adapted merged model features will prevent representation collapse during unsupervised coefficient search.",
        "rationale": "During test-time adaptation, model updates can collapse toward a dominant direction that minimizes entropy or self-labeled loss but harms other tasks. Isotropic regularization maintains a balanced representation space."
    },
    {
        "id": 9,
        "title": "Manifold Continual Merging (MCM)",
        "hypothesis": "A sequential merging algorithm that maps cumulative task-specific updates directly into the Lie algebra so(d) at each continual learning step will mitigate catastrophic forgetting without requiring the storage of historical task vectors.",
        "rationale": "Euclidean continual merging (like SAIM or TA) accumulates updates in a highly distorted space. Accumulating orthogonal matrices on the Riemannian manifold preserves the angular relations of parameters, inherently protecting historical knowledge."
    },
    {
        "id": 10,
        "title": "Selective Coordinate-Descent Orthogonal Finetuning (SCD-OFT)",
        "hypothesis": "Updating only the top p% of elements in the skew-symmetric Lie algebra representation Q with the highest gradient momentum during OFT fine-tuning will yield highly parameter-efficient and stable orthogonal adaptations.",
        "rationale": "Combining SA-BCD's coordinate-selection mechanism with OFT allows targeted adaptation of only the most task-sensitive geometric dimensions on the manifold."
    }
]

chosen = random.choice(ideas)
print("Chosen Idea:")
print(f"ID: {chosen['id']}")
print(f"Title: {chosen['title']}")
print(f"Hypothesis: {chosen['hypothesis']}")
print(f"Rationale: {chosen['rationale']}")
