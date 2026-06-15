# 1. Summary of the Paper

## Main Topic and Problem Statement
The paper addresses a critical challenge in adaptive, test-time model merging (TTA). Direct parameter-space merging of task-specific expert models (which are fine-tuned from a shared pre-trained base model $\theta_0$) has emerged as a computationally efficient alternative to multi-task training from scratch. Recent adaptive model-merging methods like *AdaMerging* introduce learnable layer-wise merging coefficients $\boldsymbol{\lambda}$ that are optimized online during deployment using gradient-based minimization of predicted entropy on unlabeled, local test data streams.

The authors identify and define a fundamental failure mode of this approach, termed the **Overfitting-Optimizer Paradox**: unconstrained unsupervised optimization of merging coefficients online is highly susceptible to transductive noise and local distribution shifts. To minimize entropy on local streams, the optimizer introduces high-frequency spatial oscillations in merging coefficients across network depth. Under the physical laws of representation learning, these uncoordinated coefficient fluctuations across adjacent layers disrupt the internal representation manifold of the model, leading to catastrophic representation collapse and a severe drop in generalization performance (often performing worse than simple static uniform baselines).

## Proposed Approach: RCR-Merge
To resolve the Overfitting-Optimizer Paradox, the paper proposes **Riemannian Curvature-Regularized Test-Time Model Merging (RCR-Merge)**. The core philosophy is to model the deep neural network parameter space as a Riemannian manifold where distance is locally scaled by the second-order curvature of the loss landscape, rather than a flat, isotropic Euclidean surface.

The practical execution of RCR-Merge consists of a three-step sequential pipeline:
1. **Offline Base Curvature Estimation**: The diagonal trace of the Fisher Information Matrix (FIM) for each layer of the pre-trained base model $\theta_0$ is estimated offline using a minuscule calibration batch (e.g., $|D_{\text{cal}}| = 64$ samples). The resulting curvature vector is normalized across depth to act as a spatial Riemannian metric.
2. **Online Test-Time Adaptation**: As unlabeled test data streams arrive, the merging coefficients $\boldsymbol{\lambda}$ are optimized online via entropy minimization.
3. **Riemannian Curvature-Weighted Total Variation (RCR-TV) and Absolute Anchoring**: The joint optimization objective includes a spatial regularizer that penalizes the squared difference of merging coefficients between adjacent layers, scaled by the geometric mean of their pre-trained base curvatures ($\sqrt{c_l c_{l-1}}$). To prevent joint-drift (where all coefficients shift in tandem under correlated noise), an absolute coordinate anchoring penalty ($\gamma \|\boldsymbol{\lambda} - \boldsymbol{\lambda}_0\|_2^2$) is also added.

Additionally, the authors propose **Gradient Norm Balancing (GNB)** to dynamically initialize the spatial regularization strength $\beta$ at step 0 in a fully unsupervised manner. By evaluating the gradient of the regularizer at a worst-case spectral perturbation (alternating signs representing maximum spatial frequency), GNB standardizes the optimization coordinates to a unit perturbation sphere and calculates a scale-invariant ratio of the initial loss gradient to the regularizer's peak sensitivity.

## Key Findings
1. **Unconstrained Adaptation Collapses**: Unconstrained AdaMerging suffers severely from transductive noise, causing multi-task accuracy to drop catastrophically below the static uniform baseline across both coupled and uncoupled metrics.
2. **Curvature-Guided Smoothing Works**: Weighing the spatial Total Variation by pre-trained base curvatures provides an effective analytical barrier that protects sensitive bottleneck layers (early attention and late task heads) while allowing adaptability in flat, robust layers.
3. **GNB Drastically Simplifies Tuning**: GNB successfully automates hyperparameter selection, replacing a scale-dependent search over $\beta$ with a highly stable, dimensionless scale factor $\alpha$.
4. **Local Conformal Barriers Beat Global Subspaces**: On modular, stage-wise transition landscapes with discrete boundaries, RCR-Merge significantly outperforms PolyMerge, which suffers from global curve deformations (Runge's phenomenon).
5. **Static Approximations are Robust**: The pre-trained base curvature $G(\theta_0)$ remains an exceptionally stable and permanent geometric prior during adaptation, matching the performance of a dynamic oracle even under 50% cumulative parameter drift.

## Explicitly Claimed Contributions (with Evidence provided in Paper)
- **Formulation of the Overfitting-Optimizer Paradox**: The authors provide conceptual arguments and demonstrate empirically (Table 1) that unconstrained entropy minimization collapses representations.
- **Introduction of RCR-Merge & RCR-TV**: The paper outlines the second-order geometric regularizer (Equation 5) and details its K-FAC tensor-wise extensions.
- **Formal Theoretical Guarantees**:
  - *Lemma 3.1 (Coordinate-Level Spatial Barrier)*: Proves that squared spatial variation of coefficients is bounded by the inverse of the base curvatures.
  - *Theorem 3.2 (Representation Drift Bounding)*: Formally bounds activation-level adjacent-layer representation drift by the curvature-weighted spatial difference of coefficients.
  - *Spectral Analysis*: Models RCR-Merge as a Laplacian low-pass filter that blocks high-frequency noise from propagating.
- **Empirical Evaluations on Emulator**: A 30-seed simulation study on the Coupled Model II Landscape showing RCR-Merge outperforms the Uniform Baseline (+3.06% absolute), unconstrained AdaMerging (+10.33%), and flat TV (+5.01%).
- **Real-World BERT and ViT Pilot Studies**: Provides a pilot implementation on BERT-Base and ViT-B/16 on CPU, showing that RCR-Merge prevents representation collapse on real architectures and validating that pre-trained curvatures correlate heavily (99.00% cosine similarity) with online adapted curvatures.
