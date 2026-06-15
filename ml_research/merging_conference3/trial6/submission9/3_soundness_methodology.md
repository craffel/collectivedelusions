# 3. Soundness and Methodology Check

This document outlines the major methodological flaws, logical contradictions, and scientific integrity issues identified in the proposed paper.

---

## 1. Synthetic "Simulation" Presented as Real-World ViT Experiments
The paper's abstract, introduction, and section headings present the empirical results as being conducted on a real 14-layer compact Vision Transformer (`vit_tiny_patch16_224`) backbone across four standard datasets (MNIST, FashionMNIST, CIFAR-10, SVHN).
However, an inspection of the source code (`run_experiments_new.py`) reveals that **no real models are trained, no real datasets are processed, and no physical model merging is ever performed.**
Instead, the entire evaluation is run on a highly artificial, toy simulator (`MultiTaskFeatureSimulator` and `ModelMergingEnvironment`), where:
- Image patch features $H_0 \in \mathbb{R}^{B \times N \times D}$ are generated from random Gaussian noise with hardcoded task-specific masks.
- Class prediction accuracies are calculated using a closed-form algebraic formula based on a hardcoded "ceiling" value:
  $$\text{prob\_correct} = 0.1317 + (\text{ceiling} - 0.1317) \times \text{norm\_score}$$

While the authors briefly mention a "controlled coordinate-space sandbox" in Section 4, the overall paper framing is highly misleading. An unsuspecting reader would believe that the reported "53.07% Joint Mean Accuracy" and "robustness" were obtained from actual training and inference of a Vision Transformer on physical images. Real parameter-space dynamics are highly non-linear, and sudden representational collapse can occur from minor coefficient deviations, which this monotonic sigmoid-based proxy trivializes.

---

## 2. Evaluation Discrepancy & Structural Bias in Task Heterogeneity Sweep
There is a severe structural bias in how the models are evaluated in the Batch Size & Task Heterogeneity Sweep (Sweep 3):
- **Physical Merge Constraint:** When deploying a merged model on GPU, weights are loaded once in memory, and the entire batch is processed using the same weights. Thus, a dynamic router must predict a single set of merging coefficients of shape `[K]` for the entire batch. Standard routers (like `BSigmoidRouter`) adhere to this constraint by average pooling their predicted coefficients over the batch dimension, returning a single coefficient vector of shape `[K]`.
- **Bypassing the Constraint:** In Sweep 3, `BSigmoidRouter` is evaluated under this physical constraint (average-pooled coefficients), causing its accuracy to collapse because the mixed-task batch averages out task-specific signatures. However, `CAMRouter` is evaluated with `return_sample_alphas=True`, which returns individual sample-level coefficients of shape `[B, K]`. 
- **Unfair Comparison:** By evaluating `CAMRouter` using sample-specific coefficients and comparing it with a baseline restricted to batch-averaged coefficients, the authors introduce an unfair structural advantage. `CAMRouter` completely bypasses the physical batch weight-merging constraint. If `CAMRouter` were evaluated under the same physical constraint as `BSigmoidRouter` (returning a single averaged coefficient vector over the batch), it would also suffer from "heterogeneity collapse."
- **DHG Evaluation Omission:** While the authors propose "Decoupled Historical Gating (DHG)" in Section 3.3 as their core solution to batched inference in production, they do not actually evaluate `CAMRouter` using DHG in Sweep 3. Instead, they bypass it by using sample-specific alphas directly, leaving DHG completely unverified empirically under the physical batch-merging constraint.

---

## 3. Training-Inference Discrepancy in Decoupled Gating (DHG)
To address task heterogeneity, the authors introduce Decoupled Historical Gating (DHG) for batched inference, where per-sample coefficients are tracked via an exponential moving average (EMA) over historical steps.
However, in Section 3.3 ("Batched Calibration Training"), they state:
> "During calibration training, average-pooling is utilized over the small calibration batch as an efficient gradient-smoothing mechanism."

This introduces a severe **training-inference discrepancy**:
- During **training**, the router is optimized using average-pooled coefficients over active batch elements, meaning the gradients are smooth but backpropagation is conditioned on the average routing representation of the batch.
- During **inference**, the router is evaluated using a sample-level EMA (DHG) that completely bypasses the average pooling of the active batch.

Training a model under one feature-aggregation distribution (batch average pooling) and deploying it under a different sequential distribution (historical EMA) represents a major methodological flaw that can lead to severe covariate shift and unpredictable routing behavior in production.
