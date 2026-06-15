# Norm-Equalized Task Arithmetic (NETA)

## 1. Persona Alignment
This proposal aligns perfectly with the core principles of **The Minimalist** (as specified in `persona.md`):
- **Aggressively Prunes Complexity**: Instead of proposing a convoluted test-time optimization pipeline involving unlabeled datasets, predictions, entropy minimization, gradient descent or derivative-free search (which we have shown to be highly prone to transductive overfitting and task-bias), NETA solves the problem *analytically* in a single, closed-form, training-free step.
- **Zero Hyperparameter Bloat**: Standard Test-Time Adaptation (TTA) techniques introduce multiple hyperparameters (learning rate, number of calibration steps, calibration batch sizes, optimization schedules, regularizer weights). NETA introduces exactly **zero** new hyperparameters. It operates directly on top of standard Task Arithmetic, utilizing the existing global scalar $\lambda_0$.
- **High Readability and Elegance**: The proposed method consists of a simple, three-line closed-form weight-space scaling formula. It is exceptionally clean, readable, reproducible, and mathematically elegant.
- **Skeptical of Convoluted Pipelines**: NETA departs from the trend of layer-wise coefficient search (such as AdaMerging or SyMerge) which we audited and proved to be overparameterized and fragile under shuffling or averaging on complex tasks. Instead of searching for parameters, NETA directly targets the physical imbalance of task updates.

---

## 2. Core Techniques
NETA modifies and builds directly upon the foundational **Task Arithmetic** framework [1] and the concept of magnitude-based balancing introduced in TIES-Merging [2], but does so without TIES's complex heuristic trimming, sign election, and rescaling pipeline.
- **Task Vector Extraction**: Following Ilharco et al. [1], we compute task vectors as the direct parameter difference between the independently fine-tuned expert weights and the pre-trained base network weights.
- **Layer-wise Isotropic Balancing**: We introduce a closed-form scale factor for each layer and task that scales the task vectors dynamically. This is inspired by the need for magnitude and variance alignment (as explored in isotropic weight merging [3]), but is implemented as a simple, direct weight-space normalizer.

**References:**
- [1] Ilharco, G., et al. "Editing Models with Task Arithmetic." ICLR 2023.
- [2] Yadav, P., et al. "Resolving Interference in Model Merging." NeurIPS 2023.
- [3] "Sharpness-Aware Isotropic Merging (SAIM)." 2026.

---

## 3. Mathematical Formulation
Let $\theta_{\text{pre}}^l \in \mathbb{R}^{d_l}$ represent the parameter weights of the pre-trained base model at layer $l \in \{1, \dots, L\}$.
Let $\theta_k^l \in \mathbb{R}^{d_l}$ represent the fine-tuned expert weights for task $k \in \{1, \dots, K\}$ at layer $l$.

The raw task vector $\tau_k^l$ is extracted as:
$$\tau_k^l = \theta_k^l - \theta_{\text{pre}}^l$$

We define the Frobenius norm of each task vector at layer $l$ as $\|\tau_k^l\|_F$.
The average task vector norm at layer $l$ across all $K$ tasks is computed analytically as:
$$\mu^l = \frac{1}{K} \sum_{j=1}^K \|\tau_j^l\|_F$$

To balance the relative representation scales of each task at layer $l$, we define the NETA scaling coefficient $w_k^l$ as:
$$w_k^l = \frac{\mu^l}{\|\tau_k^l\|_F + \epsilon}$$
where $\epsilon = 10^{-6}$ is a standard numerical stabilizer to prevent division by zero in layers with zero weight change.

The final merged parameter weights $\theta_{\text{merged}}^l$ for layer $l$ are given by the analytical closed-form:
$$\theta_{\text{merged}}^l = \theta_{\text{pre}}^l + \lambda_0 \sum_{k=1}^K w_k^l \tau_k^l$$
where $\lambda_0$ is the standard global scaling coefficient of Task Arithmetic (typically $\lambda_0 \in [0.1, 1.0]$).

### Theoretical Properties:
1. **Perfect Magnitude Isotropy**:
   The norm of each balanced task vector contribution is:
   $$\|\hat{\tau}_k^l\|_F = \|w_k^l \tau_k^l\|_F = w_k^l \|\tau_k^l\|_F = \frac{\mu^l}{\|\tau_k^l\|_F + \epsilon} \|\tau_k^l\|_F \approx \mu^l$$
   This guarantees that at every layer, each expert model contributes a weight update of *identical* Frobenius norm to the merged model.
2. **Preservation of Global Scale**:
   Summing the balanced task vectors preserves the expected scale of updates:
   $$\sum_{k=1}^K \|\hat{\tau}_k^l\|_F \approx K \mu^l = \sum_{k=1}^K \|\tau_k^l\|_F$$
   This ensures that NETA does not shrink or blow up the total update scale compared to standard Task Arithmetic, preventing representation collapse.

---

## 4. Architecture Specifications
NETA is a model-agnostic, layer-wise weight transformation. It applies to any standard neural network architecture (Transformers, CNNs, MLPs) divided into $L$ parameter groups.

- **Inputs**:
  - Pre-trained base weights: $\{\theta_{\text{pre}}^l\}_{l=1}^L$
  - Fine-tuned expert weights: $\{\{\theta_k^l\}_{k=1}^K\}_{l=1}^L$
  - Global scaling hyperparameter: $\lambda_0 \in \mathbb{R}$
- **Intermediate Representations**:
  - Layer-wise task vectors: $\tau_k^l = \theta_k^l - \theta_{\text{pre}}^l$
  - Task vector Frobenius norms: $\|\tau_k^l\|_F$
  - Layer-wise mean norms: $\mu^l$
  - Analytical scaling coefficients: $w_k^l$
- **Final Output**:
  - Merged parameter state dict: $\{\theta_{\text{merged}}^l\}_{l=1}^L$
- **Target Backbone**:
  - We target the standard **CLIP ViT-B/32** model. The model is partitioned into $L = 13$ discrete parameter groups: 12 Transformer Blocks (including self-attention and MLP weights) and 1 Visual Projection Layer (`model.visual.proj`). NETA scales parameters group-by-group (layer-by-layer) across these 13 partitions.

---

## 5. Baselines
We will evaluate NETA against the following appropriate baselines:
1. **Task Arithmetic (Vanilla)**: The foundational baseline using a constant global coefficient $\lambda = 0.3$ across all tasks and layers. This is appropriate as NETA aims to improve upon Task Arithmetic with zero extra data or training.
2. **Spatially Averaged AdaMerging (1+1 ES)**: The strong regularized baseline identified in Paper 3 (The Overfitting-Optimizer Paradox), which uses a single optimized scalar per task (4 parameters total) tuned via prediction entropy minimization on 256 unlabeled calibration images.
3. **Optimized AdaMerging (Adam GD)**: The standard layer-wise test-time adaptation baseline (52 parameters) tuned via continuous first-order backpropagation on 256 unlabeled calibration images. This represents the SOTA but prone-to-overfitting paradigm we aim to outperform.

---

## 6. Step-by-Step Interaction
The NETA model merging pipeline executes in the following deterministic, training-free steps:

1. **Extraction**:
   - For each layer $l \in \{1, \dots, 13\}$ and each task $k \in \{1, 2, 3, 4\}$:
     - Load the pre-trained CLIP base weights $\theta_{\text{pre}}^l$ and task expert weights $\theta_k^l$.
     - Compute the task vector: $\tau_k^l = \theta_k^l - \theta_{\text{pre}}^l$.

2. **Norm Calculation**:
   - For each layer $l$ and task $k$:
     - Compute the Frobenius norm of the task vector: $\|\tau_k^l\|_F = \sqrt{\sum_i (\tau_{k, i}^l)^2}$.
   - For each layer $l$:
     - Compute the average norm across the 4 tasks: $\mu^l = \frac{1}{4} \sum_{j=1}^4 \|\tau_j^l\|_F$.

3. **Coefficient Synthesis**:
   - For each layer $l$ and task $k$:
     - Compute the NETA scaling weight: $w_k^l = \frac{\mu^l}{\|\tau_k^l\|_F + 10^{-6}}$.

4. **Merging**:
   - For each layer $l$:
     - Compute the merged weights: $\theta_{\text{merged}}^l = \theta_{\text{pre}}^l + \lambda_0 \sum_{k=1}^4 w_k^l \tau_k^l$.

5. **Assembly**:
   - Assemble the merged layers into the final CLIP ViT-B/32 state dict.
   - Deploy the merged model for multi-task visual classification evaluation with zero additional adaptation or inference overhead.
