import random

ideas = [
    "Idea 1: A Critical Deconstruction of Data-Free Parameter Calibration. Rigorously comparing HNS and IPR against a finely-tuned global Task Arithmetic baseline to show that complex layer-wise/channel-wise scaling does not outperform simple tuned global scaling.",
    "Idea 2: BatchNorm Re-estimation is all you need. Demonstrating that representation collapse in model merging is primarily a BatchNorm running statistics misalignment, and that recomputing stats with as few as 16 unlabeled samples outperforms all data-free parameter scaling methods by a massive margin.",
    "Idea 3: The Orthogonality Myth in Model Merging. Investigating the validity of the 'orthogonal task vectors' assumption. We show that task vectors are only orthogonal under specific, high-learning-rate or unregularized training, and when tasks are related or regularized, orthogonality breaks, causing HNS and IPR to fail.",
    "Idea 4: On the Flawed Evaluation of Data-Free Merging. Exposing how data-free merging papers select weak uncalibrated baselines (e.g., WA without scaling) to show dramatic performance gains, while a simple grid-search on lambda matches their performance without any complex math.",
    "Idea 5: Robustness of Model Merging under Out-of-Distribution (OOD) and Noise. Evaluating whether HNS and IPR degrade under noisy or out-of-distribution test conditions, compared to simple test-time BN calibration.",
    "Idea 6: Task-Specific vs. Unified Merging: The Hidden Storage Cost of HNS. Rigorously comparing HNS (which is task-specific and requires storing scale vectors per task) against unified merging methods like U-IPR, and questioning if HNS actually saves practical serving costs.",
    "Idea 7: Isotropic vs. Anisotropic Parameter Scaling. A systematic evaluation of whether spectral/anisotropic scaling (S-IPR, SA-IPR) provides any real-world benefit over isotropic scaling (U-IPR) or if it is just overfitting to the test set.",
    "Idea 8: An Empirical Analysis of Warm-up and Fine-tuning Hyperparameters on Mergeability. Showing that representation collapse is itself an artifact of poor fine-tuning practices (e.g., lack of weight decay or too high learning rate), and proper fine-tuning mitigates collapse before merging even occurs.",
    "Idea 9: Data-Free vs. Data-Efficient Calibration: A Pareto Frontier Analysis. Mapping the exact trade-off between the number of calibration samples (0 to 1024) and the classification accuracy, showing that even 4-8 samples of real data provide massive gains over 100% data-free methods.",
    "Idea 10: The Impact of Model Architecture on Parameter Resonance. Testing if U-IPR and HNS generalize to non-ResNet architectures (like Vision Transformers or MLPs) and showing that their assumptions are highly architecture-dependent and fail on models without BatchNorm."
]

# Use a deterministic seed for reproducibility
random.seed(101)
selected_idx = random.randint(0, len(ideas) - 1)
print(f"Selected Idea Index: {selected_idx + 1}")
print(f"Selected Idea: {ideas[selected_idx]}")
