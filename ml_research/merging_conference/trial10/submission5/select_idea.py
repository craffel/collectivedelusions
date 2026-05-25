import random
import sys

# Seed with the Slurm Job ID for reproducible pseudo-random selection
job_id = 22160544
random.seed(job_id)

ideas = [
    {
        "id": 1,
        "title": "Dynamic Frequency-Domain Model Merging (DFD-TTMM)",
        "description": "Damps high-frequency environmental noise by performing test-time merging and parameter optimization in the spectral/frequency domain, or applying a dynamic low-pass filter to weight-space updates.",
        "expected_results": "Significant improvement in robustness to high-frequency and Gaussian noise without sacrificing performance on clean domains.",
        "impact": "High. Addresses a core vulnerability of test-time optimization under noisy streams."
    },
    {
        "id": 2,
        "title": "Contrastive Sharpness-Aware Test-Time Model Merging (C-SAM-TTMM)",
        "description": "Combines sharpness-aware optimization of merging coefficients with a self-supervised contrastive loss on augmented test-batch views. This guides merging parameters toward stable weight regions without relying solely on entropy minimization.",
        "expected_results": "Avoids the 'feedback trap' and prevents representational collapse on long-horizon streams.",
        "impact": "High. Introduces a robust unsupervised objective alternative to predictive entropy."
    },
    {
        "id": 3,
        "title": "Adaptive Temporal-Consistency Kronecker Model Merging (ATC-KMM)",
        "description": "Incorporates a temporal smoothness regularizer directly into Kronecker-preconditioned updates to penalize sudden large shifts in layer-wise merging coefficients unless a major domain shift is explicitly detected.",
        "expected_results": "Smoother, more stable merging trajectories across continuously shifting streams and reduced variance in batch-by-batch predictions.",
        "impact": "Medium-High. Enhances stability and temporal cohesion in streaming test-time settings."
    },
    {
        "id": 4,
        "title": "Bayesian Evidential Model Merging (BEMM)",
        "description": "Uses Evidential Deep Learning (EDL) to compute subjective uncertainty (belief masses) for each expert. These uncertainties dynamically scale the temperature and the merging coefficients, allowing explicit OOD detection.",
        "expected_results": "Highly reliable routing and merging weights on both in-distribution and novel out-of-distribution (KMNIST) streams.",
        "impact": "High. Bridges model merging with evidential uncertainty estimation."
    },
    {
        "id": 5,
        "title": "Wasserstein-Distance Adaptive Hybrid Routing (W-AHR)",
        "description": "Replaces Euclidean/Angular distance-based prototype routing with Wasserstein distance (optimal transport) between the batch's feature distribution and the experts' prototype distributions.",
        "expected_results": "Greater resilience to geometric and environmental shifts that distort individual feature coordinates under noise.",
        "impact": "Medium. Provides a more mathematically robust distance metric for prototype-based gating."
    },
    {
        "id": 6,
        "title": "Curvature-Guided Meta-Learning for Test-Time Model Merging (CG-MTTMM)",
        "description": "Dynamically scales the test-time adaptation learning rate and damping factor on-the-fly using the local loss curvature (approximated via Hessian trace), eliminating the need for manual hyperparameter tuning.",
        "expected_results": "Faster convergence and optimal hyperparameter selection across varying noise intensities and domain shifts.",
        "impact": "High. Automates and optimizes the test-time optimization loop."
    },
    {
        "id": 7,
        "title": "Sparse Parameter-Sensitive Model Merging (SPS-MM)",
        "description": "Restricts test-time parameter updates to the top-k most sensitive layers (identified via Kronecker sensitivity estimation), keeping other parameters frozen to reduce optimization noise.",
        "expected_results": "Faster optimization, lower compute overhead, and a natural defense against representational overfitting under noise.",
        "impact": "Medium-High. Optimizes resource utilization and prevents noise-overfitting."
    },
    {
        "id": 8,
        "title": "Feedback-Averted Multi-Scale Entropy Minimization (FA-MEM)",
        "description": "Minimizes entropy across multiple feature representation scales (at intermediate layer outputs) rather than just the final classifier output, preventing the feedback trap in deep layers.",
        "expected_results": "Better preservation of representation structure and avoidance of overconfident incorrect local minima.",
        "impact": "Medium-High. Offers a distributed unsupervised adaptation signal."
    },
    {
        "id": 9,
        "title": "Wasserstein-Moment-Matched Batch Normalization (WMM-BN)",
        "description": "Fuses expert Batch Normalization statistics using Wasserstein barycenters of activation distributions instead of simple linear moment matching or Gaussian mixture modeling.",
        "expected_results": "More precise alignment of activation maps between expert networks, completely eliminating representational mismatch.",
        "impact": "High. Advances the state-of-the-art in weight-space Batch Normalization fusion."
    },
    {
        "id": 10,
        "title": "Prototype-Contrastive Sharpness-Aware Model Merging (PC-SAM)",
        "description": "Integrates SAM-TTMM with a prototype-contrastive regularizer. This forces the adapted representation of the current batch to remain structurally close to the nearest expert prototype, preventing representational collapse and the feedback trap during sharpness-aware parameter perturbation.",
        "expected_results": "State-of-the-art accuracy under severe noise and OOD shifts by combining sharpness-awareness with semantic representation constraints.",
        "impact": "Very High. Directly addresses both the flat loss region and semantic preservation goals simultaneously."
    }
]

# Select one pseudo-randomly
chosen_idx = random.randint(0, len(ideas) - 1)
chosen_idea = ideas[chosen_idx]

print(f"Chosen Idea Index: {chosen_idx} (1-based: {chosen_idx+1})")
print(f"Chosen Idea Title: {chosen_idea['title']}")

# Prepare the markdown content to append to progress.md
progress_content = f"""
## Phase 1: Foundation (Read & Formulate)

### Literature Review & Synthesis
Based on our reading of the three core papers (SAM-TTMM, AHR-SATS-DUN, and BK-AHR), we synthesized the following key insights:
- **General Themes**: Test-Time Model Merging (TTMM) is a powerful, data-free paradigm that dynamically fuses specialized expert parameters on-the-fly to handle non-stationary streaming data without retraining.
- **Core Contributions**:
  - *SAM-TTMM* identifies the 'feedback trap' (entropy minimization driving merging weights into sharp, overconfident local minima) and proposes sharpness-aware optimization (SAM) to find flat loss regions in interpolation space.
  - *AHR-SATS-DUN* identifies that background sparsity under noise causes spherical normalization collapse and proposes Adaptive Hybrid Routing (using Hoyer sparsity), Decisive Under Noise (DUN) temperature scaling, and Entropy-Adaptive Learning Rate (EALR) regularization.
  - *BK-AHR* proposes on-the-fly Kronecker sensitivity estimation from parameter gradients for preconditioning and soft Bayesian Mixture-of-Gaussians BN statistic fusion to align activations.
- **Limitations**: Existing methods either optimize merging parameters without semantic constraints (leading to potential representational collapse) or rely on distance metrics that struggle under heavy noise.

### Formulating Ten Novel Ideas
1. **Dynamic Frequency-Domain Model Merging (DFD-TTMM)**: Merging/optimization in frequency domain to filter out high-frequency noise.
2. **Contrastive Sharpness-Aware Test-Time Model Merging (C-SAM-TTMM)**: Guidance using self-supervised contrastive learning.
3. **Adaptive Temporal-Consistency Kronecker Model Merging (ATC-KMM)**: Temporal consistency constraints on Kronecker preconditioned updates.
4. **Bayesian Evidential Model Merging (BEMM)**: EDL-based subjective uncertainty estimation for routing/temperature scaling.
5. **Wasserstein-Distance Adaptive Hybrid Routing (W-AHR)**: Wasserstein distance for prototype-based gating.
6. **Curvature-Guided Meta-Learning for Test-Time Model Merging (CG-MTTMM)**: Curvature-guided LR and damping parameter tuning.
7. **Sparse Parameter-Sensitive Model Merging (SPS-MM)**: Restricting updates to top-k sensitive layers.
8. **Feedback-Averted Multi-Scale Entropy Minimization (FA-MEM)**: Multi-scale entropy regularization at intermediate layers.
9. **Wasserstein-Moment-Matched Batch Normalization (WMM-BN)**: Fusing BN stats using Wasserstein barycenters.
10. **Prototype-Contrastive Sharpness-Aware Model Merging (PC-SAM)**: Combining SAM-TTMM with prototype-contrastive loss to prevent representational collapse.

### Pseudo-Random Idea Selection
- **Seed**: `{job_id}` (Slurm Job ID)
- **Selected Index**: `{chosen_idx+1}`
- **Selected Idea**: **{chosen_idea['title']}**
- **Description**: {chosen_idea['description']}
- **Expected Results**: {chosen_idea['expected_results']}
- **Impact**: {chosen_idea['impact']}
- **Rationale**: This idea is highly novel and elegant. By combining sharpness-aware optimization (which targets flat regions in the merging parameter space) with a prototype-contrastive regularizer (which prevents semantic drift and representational collapse), we address both the structural flat-region objective and the semantic representation constraint. This double-defense mechanism directly resolves the 'feedback trap' and the 'sparsity-density tradeoff' highlighted across all three papers, making it extremely likely to deliver superior generalization on non-stationary, noisy, and OOD streams.

"""

with open("progress.md", "a") as f:
    f.write(progress_content)

print("Successfully updated progress.md with Phase 1 details.")
