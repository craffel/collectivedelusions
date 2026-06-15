# Intermediate Review Evaluation: Paper Summary (1_summary.md)

This document provides a comprehensive summary of the conference submission titled *"Information-Geometric Subspace Routing: A Provably Stable Parameter-Free Framework for Test-Time Model Merging"*, covering the main topic, approach, key findings, and explicitly claimed contributions with their corresponding empirical and theoretical evidence.

---

## 1. Main Topic
The paper addresses the challenge of **test-time model merging** (or parameter ensembling) of specialized modular domain-expert models (such as LoRAs) sharing a pre-trained backbone. At test-time, the objective is to dynamically ensemble these specialized weights to handle incoming heterogeneous, mixed-domain input streams without needing manual task boundaries or intensive retraining.

The paper identifies three primary vulnerabilities in existing test-time ensembling and dynamic routing frameworks:
1. **The Dynamic Routing Paradox (Few-Shot Overfitting):** Parametric routers (e.g., linear routers or wave superposition models) optimized on microscopic calibration/support sets (e.g., $N=64$ samples) suffer from catastrophic overfitting, frequently assigning incorrect, saturated routing weights.
2. **Vectorization Collapse under Sequential Streams:** Traditional dynamic routers rely on batch-average smoothing. In low-latency single-sample streaming ($B=1$), routing coefficients fluctuate wildly, degrading ensembling accuracy.
3. **Flat Euclidean Geometrical Misspecification:** Parameter-free subspace routing (PFSR) methods project representations onto class prototypes using unweighted cosine similarity, which geometrically and information-theoretically assumes a flat, isotropic weight/representation space. This ignores the asymmetric sensitivity and coordinate warping induced by specialized fine-tuning.

---

## 2. Proposed Approach: FIOSR
To resolve these issues, the paper proposes **Fisher-Information Optimal Subspace Routing (FIOSR)**, a training-free and parameter-free dynamic ensembling framework rooted in information geometry.

### Core Components:
* **Information-Geometric Riemannian Manifolds:** The parameter space of the expert models is modeled as a Riemannian manifold, where the local coordinate sensitivities are represented by a smoothed and power-scaled **diagonal empirical Fisher Information Matrix (dFIM)** of the expert heads, estimated on-the-fly over a tiny calibration split (default $N_c=16$ samples per task).
* **Fisher-Weighted Cosine Similarity:** Instead of flat Euclidean cosine projections, FIOSR projects test representations onto expert class prototypes using a warped inner product defined by the Riemannian metric tensor $\mathbf{g}_{k, c} = \text{diag}(\tilde{F}_{k, c})$. This analytically warps the coordinate space, suppressing noisy dimensions (low Fisher value/high variance) and magnifying highly discriminative directions (high Fisher value/low variance).
* **Class-Size Scaling Calibration (CSC):** This normalizes maximum coordinate similarities by their task-specific expected maximums under extreme value theory (scaling by $\sqrt{2\log C_k / d}$), correcting for statistical maximum bias under asymmetric expert vocabulary sizes $C_k$. The paper also proposes Correlation-Corrected Class-Size Scaling Calibration (CC-CSC) to adjust for correlated class prototypes.
* **Micro-Batch Homogenization (MBH):** An unsupervised batch-partitioning algorithm that groups incoming heterogeneous stream samples by their dominant expert task before executing forward passes. This prevents *heterogeneity collapse* and avoids running $B$ separate forward passes for individual routing coefficients.

---

## 3. Key Findings
* **Overfitting Mitigation:** Since FIOSR is completely training-free and parameter-free, it is entirely immune to the Dynamic Routing Paradox and few-shot overfitting. It achieves superior routing without optimization.
* **Streaming Stability:** Combined with MBH, FIOSR maintains flat-line stability across all stream batch sizes ($B=1$ to $512$), completely eliminating Vectorization Collapse and Heterogeneity Collapse.
* **Accuracy Improvements:** In a simulated 192-dimensional *Analytical Coordinate Sandbox* with 10 random seeds, FIOSR significantly outperforms unweighted cosine projection (PFSR+MBH) by **8.56%** absolute joint accuracy ($76.86\%$ vs $68.30\%$) and outperforms complex parametric routers (which collapse to near-uniform ensembling accuracy of $\sim 36-39\%$).
* **Oracle Performance Recovery:** FIOSR successfully recovers individual specialized expert performance ceilings, achieving $\approx 100\%$ routing accuracy on Simulated Task 0 (MNIST-equivalent) and Simulated Task 1 (FashionMNIST-equivalent), and near-ceiling accuracy on noisy Simulated Task 2 (CIFAR-equivalent) and Simulated Task 3 (SVHN-equivalent).
* **Real-world Validation:** FIOSR's viability is successfully demonstrated on (i) a 64-dimensional simulated LoRA activation space, achieving a $95.00\%$ routing accuracy and recovering $98.30\%$ of the theoretical expert oracle performance, and (ii) an end-to-end physical ResNet-18 deployment on MNIST, FashionMNIST, and SVHN, outperforming flat ensembling.

---

## 4. Explicit Claims & Evidence

| Explicit Claim | Type of Evidence | Detailed Evidence from the Text |
| :--- | :--- | :--- |
| **Immunity to Few-Shot Overfitting / Dynamic Routing Paradox** | Quantitative Empirical & Theoretical | Bypasses training parameters entirely. In the homogeneous $B=256$ sandbox, FIOSR achieves **76.86%** joint accuracy, while parametric routers collapse due to overfitting (Linear Unreg: 38.91%, QWS-Merge: 36.42%, L3-Softmax: 38.85%). |
| **Immunity to Vectorization Collapse** | Quantitative Empirical | Evaluated on mixed-task streams of 1,000 samples across batch sizes $B=1$ to $512$. FIOSR maintains perfectly flat performance of **76.83%** (at $B=1$) to **76.83%** (at $B=512$), whereas parametric methods are highly volatile at low batch sizes. |
| **Necessity of Class-Size Scaling Calibration (CSC)** | Empirical Ablation & Extreme Value Theory | Under asymmetric vocabularies $C_{\text{tasks}} = [10, 10, 10, 4]$, removing CSC results in a statistically significant drop of **1.02% - 1.15%** absolute joint accuracy across all 10 seeds due to false-positive routing toward larger-vocabulary experts. |
| **Pooled Class-Conditional vs. Task-Level Variance** | Quantitative Empirical Ablation | Using pooled within-class covariance to isolate pure coordinate noise from class discriminative spread improves joint ensembling accuracy from **75.77%** to **76.88%** ($+1.11\%$ absolute gain) under $B=256$. |
| **Sample Efficiency / Sensitivity to Calibration Size ($N_c$)** | Quantitative Empirical Sweep | A sweep over $N_c \in \{2, 4, 8, 16, 32, 64, 128\}$ shows a sharp phase transition: underdetermined at $N_c \le 4$ (accuracy $\le 65.16\%$), but once $N_c \ge 8$, the estimators stabilize, jumping to **74.34%** joint accuracy ($+9.35\%$ gain over flat baseline), and saturating at $N_c=16$ (**75.61%**). |
| **Computational Scalability via Top-$M$ Gating** | Quantitative Empirical Ablation | Restricting ensembling to the Top-$M$ experts bounds the maximum active forward passes at $M$. At $M=1$ (hard routing selection with 1 forward pass, eliminating sequential MBH overhead), FIOSR achieves **76.87%** joint accuracy (outperforming flat Cosine by **8.84%**). |
| **Robustness under Rotated, Correlated Noise** | Quantitative Empirical Stress-Test | On a rotated noise sandbox, standard diagonal Fisher collapses below flat Cosine (67.38% vs 67.50%). However, an online covariance EVD shrinkage estimator (**FIOSR-Online**) achieves **67.68%**, and an Oracle K-FAC equivalent (**FIOSR-Rotated**) recovers **73.07%** ($+5.57\%$ gain). |
| **Classifier Weights as Activation Means Proxies** | Formal Proof & Bounding (Appendix 1.3) | Proves under $L_2$-regularized softmax cross-entropy training that classifier weights act as dual vectors aligning with class centroids. Formally bounds finite-sample directional misalignment on the unit sphere as $\le C_0/\sqrt{N_c} = \epsilon$. |
| **Fisher Information Robustness under ReLU Sparsity** | Formal Derivation (Appendix 1.2) | Formally derives the continuous-discrete Fisher Information of rectified Gaussian coordinates, proving that under extreme noise/sparsity, the Fisher Information is bounded as $F_j \approx 1.13/\sigma_j^2 \propto 1/\sigma_j^2$, validating diagonal inverse-variance coordinate filtering. |
| **End-to-End Physical Viability** | Physical Experiment | Pre-trained ResNet-18 backbone trained on MNIST, FashionMNIST, and SVHN with linear heads and global pre-calibration mean-centering. On 300 test images, FIOSR achieves **59.00%** routing accuracy and **52.00%** joint accuracy, outperforming flat Cosine (56.33% / 50.67%). |
