# 1. Summary of the Paper

## Overview
The paper entitled **"Fisher-Information Optimal Subspace Routing (FIOSR): A Provably Stable Parameter-Free Framework for Test-Time Model Merging"** addresses critical, open challenges in the field of test-time model merging (dynamic ensembling of specialized expert models/adapters without retraining).

Existing test-time ensembling methods typically suffer from:
1. **The Dynamic Routing Paradox:** Trainable parametric routers easily overfit on microscopic calibration support sets (e.g., $N=64$ samples), resulting in saturated, inaccurate routing weights on test sets.
2. **Vectorization Collapse:** Trainable routers rely heavily on the implicit variance-smoothing effect of large batch averages, causing predicted coefficients to fluctuate wildly on single-sample sequential streams ($B=1$).
3. **Flat Euclidean Geometrical Misspecification:** Parameter-free subspace routing (PFSR) methods project representation spaces using standard unweighted cosine similarity, which implicitly assumes a flat, isotropic representation and parameter space. Fine-tuning on specialized tasks warps local representation manifolds, making coordinates exhibit asymmetric sensitivities to noise.

## Proposed Solution: FIOSR
To resolve these limitations, the authors introduce **Fisher-Information Optimal Subspace Routing (FIOSR)**, a training-free and parameter-free dynamic ensembling framework that treats the parameter space as a Riemannian manifold:
- **Diagonal Fisher Information Matrix (dFIM):** Derived over a tiny calibration split ($N_c = 16$ per task), the representation-space dFIM measures coordinate-wise parameter sensitivity. Under a conditional Gaussian assumption, dFIM corresponds exactly to the inverse coordinate noise variance ($1/\sigma^2$). This serves as an analytical coordinate-warping metric.
- **Smoothed and Power-Scaled Regularizer:** To stabilize variance estimation on microscopic calibration splits, a smoothed, power-scaled dFIM regularizer is formulated as $\tilde{F}_{k, c} = (F_{k, c} + \beta)^\gamma / \sum_m (F_{k, c, m} + \beta)^\gamma$, gently warping the local Riemannian metric to highlight discriminative dimensions while suppressing noise.
- **Fisher-Weighted Cosine Similarity:** Replaces standard Euclidean/Cosine similarity with a local Riemannian projection metric scaled by the smoothed dFIM sensitivity tensor.
- **Class-Size Scaling Calibration (CSC):** Adjusts similarity coordinates using analytical expected maximums of random variables on a sphere ($\sqrt{2\log C_k / d}$) to eliminate maximum-selection statistical bias under highly asymmetric expert vocabulary dimensions (e.g., $C_{\text{tasks}} = [10, 10, 10, 4]$).
- **Micro-Batch Homogenization (MBH):** Unsupervised dynamic batch partitioning at the stream level that groups samples by their dominant task. It averages routing coefficients exclusively within single-task groups to prevent **heterogeneity collapse** and shield the merged model from vectorization collapse at batch size $B=1$.

## Primary Quantitative Findings
The framework is evaluated within a 192-dimensional synthetic **Analytical Coordinate Sandbox** across 10 random seeds (seeds 42 to 51) and on a simulated 64-dimensional LoRA activation space:
1. **Homogeneous Setting ($B=256$):** Trainable parametric routers (Linear Router, QWS-Merge, and L3-Softmax) catastrophically overfit and collapse to near-uniform ensembling accuracy ($\sim 36-39\%$). In contrast, parameter-free methods achieve outstanding generalizability. Our **FIOSR** achieves **76.86%** joint accuracy, significantly outperforming the flat Cosine baseline (**68.30%**) by **+8.56%** absolute accuracy.
2. **Individual Expert Recovery:** Under FIOSR, MNIST recovers **100.00%** routing accuracy, FashionMNIST recovers **99.88%** accuracy, CIFAR-10 recovers **56.20%** (theoretical ceiling is 55%), and SVHN recovers **51.36%** (exceeding the baseline coordinate ceiling by over 27% due to Fisher noise-suppression).
3. **Heterogeneous Streams ($B=1$ to $512$):** FIOSR combined with MBH maintains absolute "flat-line" stability across all stream batch sizes (achieving **76.83%** accuracy at $B=1$), demonstrating complete immunity to Vectorization Collapse.
4. **LoRA Activation Space Validation:** On highly anisotropic, correlated activation space statistics from physical LoRA adapters, FIOSR improves routing accuracy from **78.33%** (flat Cosine) to **95.00%** (+16.67% absolute gain) and joint ensembling accuracy to **77.00%** (recovering 98.30% of the absolute oracle ceiling).
