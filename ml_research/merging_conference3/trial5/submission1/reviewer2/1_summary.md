# Evaluation Phase 1: Comprehensive Summary of the Submission

## 1. Overview of the Submission
The submission, titled **"Riemannian Curvature-Regularized Test-Time Model Merging"**, addresses the problem of parameter-space model merging during test-time adaptation (TTA). Model merging combines independently fine-tuned, task-specific expert models (sharing a common pre-trained base model $\theta_0$) in parameter space to achieve multi-task learning without the computational cost of training from scratch. Recent adaptive merging methods (e.g., AdaMerging) introduce learnable, layer-wise merging coefficients $\boldsymbol{\lambda}$ and optimize them online on unlabeled test data streams via Shannon entropy minimization. 

The authors argue that unconstrained test-time optimization of merging coefficients is highly susceptible to transductive noise and local stream biases, leading to high-frequency spatial oscillations in coefficients across adjacent network layers. They define this failure mode as the **"Overfitting-Optimizer Paradox"**: unsupervised local optimization fits transductive noise, resulting in catastrophic representation collapse and degraded generalization performance compared to a static uniform baseline.

To resolve this, the authors propose **Riemannian Curvature-Regularized Test-Time Model Merging (RCR-Merge)**. RCR-Merge regularizes the spatial Total Variation (TV) of merging coefficients across layer depth, weighting adjacent-layer relative penalties by the geometric mean of their pre-trained base curvatures (estimated using the diagonal trace of the pre-trained base model's Fisher Information Matrix, or FIM). The authors also introduce an absolute coordinate anchoring penalty to prevent global coefficient joint-drift and a scale-invariant heuristic called Gradient Norm Balancing (GNB) to dynamically initialize the regularization weights.

---

## 2. Core Technical Approach and Pipeline
The RCR-Merge framework consists of a sequential three-step pipeline:
1. **Offline Base Curvature Estimation:** The diagonal trace of the Fisher Information Matrix (FIM) is computed for each of the $L$ layers/blocks of the pre-trained base model $\theta_0$ using a small calibration dataset ($D_{\text{cal}}$, typically $|D_{\text{cal}}| = 64$ samples). This serves as a static, block-diagonal Riemannian metric tensor.
2. **Online Test-Time Adaptation via Entropy Minimization:** During deployment, the layer-wise merging coefficients $\boldsymbol{\lambda} \in \mathbb{R}^{K \times L}$ are optimized online on incoming unlabeled test streams by minimizing the Shannon entropy of predicted probability distributions.
3. **Dual Regularization (Spatial + Absolute):** The optimization is regularized using a joint objective:
   $$\mathcal{L}_{\text{joint}}(\boldsymbol{\lambda}) = \mathcal{L}_{\text{TTA}}(\boldsymbol{\lambda}) + \beta \mathcal{R}_{\text{curv}}(\boldsymbol{\lambda}) + \gamma \mathcal{R}_{\text{anchor}}(\boldsymbol{\lambda})$$
   where:
   - $\mathcal{R}_{\text{curv}}(\boldsymbol{\lambda}) = \sum_{k} \sum_{l=2}^{L} \sqrt{c_l c_{l-1}} (\lambda_{k, l} - \lambda_{k, l-1})^2$ (Riemannian Curvature-Weighted TV penalty)
   - $\mathcal{R}_{\text{anchor}}(\boldsymbol{\lambda}) = \|\boldsymbol{\lambda} - \boldsymbol{\lambda}_0\|_2^2$ (Absolute coordinate anchoring penalty)
   - The spatial regularization weight $\beta$ and coordinate anchoring weight $\gamma$ are scaled at step 0 using Gradient Norm Balancing (GNB).

---

## 3. Explicitly Claimed Contributions
The submission explicitly claims the following primary contributions:
1. **Conceptual Formulation:** Defining and formalizing the *Overfitting-Optimizer Paradox* in adaptive model merging, detailing how unconstrained test-time entropy minimization under local stream bias collapses internal model representations.
2. **Methodological Framework:** Introducing *RCR-Merge*, a second-order geometric framework that regularizes the spatial total variation of merging coefficients by scaling adjacent-layer penalties with the pre-trained base model's diagonal FIM trace.
3. **Theoretical Guarantees:** Proving coordinate-level bounds showing that the curvature-weighted penalty acts as an analytical barrier protecting highly sensitive layers (Lemma 3.1) and a network-level output representation drift bound (Theorem 3.2).
4. **Laplacian Spectral Analysis:** Providing a spectral-theoretic explanation of the regularizer as a curvature-guided low-pass Laplacian smoothing filter that blocks high-frequency noise propagation.
5. **Empirical Evaluation:** Evaluating the framework on a rigorous 30-seed simulation study over synthetic Coupled Model II and Stage-wise Modular Transition landscapes, showing substantial improvements in multi-task accuracy and variance reduction over static and adaptive baselines.
6. **Real-World Proof-of-Concept:** Conducting pilot studies on full-scale architectures, namely `bert-base-uncased` (110M parameters) and Vision Transformer `vit-base-patch16-224` (86M parameters), under transductive noise to demonstrate the physical viability and scalability of RCR-Merge.

---

## 4. Key Findings and Evidence Presented
The paper presents the following key findings to support its claims:
- **Simulation Study (Table 1):** In a 12-layer, 4-expert Coupled Model II emulator over 30 independent seeds, unconstrained AdaMerging drops average accuracy below the Uniform Baseline (87.45%) to 80.18% (smoothness-biased metric) and 84.82% (unbiased decoupled metric). RCR-Merge resolves this collapse, achieving **90.51%** (coupled) and **90.50%** (decoupled Euclidean), outperforming AdaMerging by +10.33% and +5.68% respectively.
- **Stage-wise Modular Transition Landscape (Table 3):** Under a modular transition landscape, RCR-Merge achieves **93.53%** (coupled) and **93.85%** (decoupled), completely outperforming PolyMerge (91.21% and 91.41%), which suffers from global curve deformation (Runge's phenomenon).
- **Ablation Study (Table 4):** Disentangling the regularizers shows that combining both spatial TV and absolute anchoring is necessary. RCR-Merge's curvature-weighted TV combined with anchoring outperforms TV + Anchor (Flat Combined) by +0.39% on the decoupled metric, validating the role of second-order base sensitivities.
- **Robustness to Parameter Drift (Table 5 & 6):** Under continuous parameter drift up to 50%, the static pre-trained curvature approximation performing RCR-Merge remains virtually indistinguishable from a dynamic oracle. For extreme non-stationarity over 2,000 steps, a threshold-triggered re-estimation mechanism achieves 94.37% accuracy with only 1 trigger tripped on average.
- **Real-World Pilots (Section 4.5):** 
  - On a 12-layer `bert-base-uncased` model fine-tuned on sentiment and topic tasks, unconstrained AdaMerging exhibits extreme coefficient divergence (e.g., $\lambda_{1, 8} = 5.13$) and drops Task 2 accuracy to 50.00% (average 75.00%). RCR-Merge stabilizes coefficients ($0.43 \le \lambda_{k, l} \le 0.57$) and maintains 100.00% accuracy.
  - On `vit-base-patch16-224` (86M parameters), unconstrained AdaMerging collapses Task 2 accuracy to 35.00% (average 47.50%). RCR-Merge stabilizes the coefficients ($0.32 \le \lambda_{k, l} \le 0.73$) and maintains 55.00% accuracy on Task 2 (average 57.50%).
