# Intermediate Evaluation 1: Summary of the Paper

This document provides a comprehensive summary of the paper's main topic, approach, key findings, and explicitly claimed contributions based on a thorough read of the submission's LaTeX source files.

## 1. Main Topic
The paper addresses the domain of **adaptive parameter-space model merging during online test-time adaptation (TTA)**. Traditional model-merging methods (such as Task Arithmetic or TIES-Merging) combine fine-tuned expert models using static, uniform coefficients. Recent methods like AdaMerging introduce learnable, layer-wise merging coefficients optimized online during test-time adaptation using gradient-based minimization of predicted output entropy on local, unlabeled data streams. 

The paper identifies a critical, previously unaddressed failure mode in this paradigm, which they term the **Overfitting-Optimizer Paradox**: unsupervised local optimization of merging coefficients easily overfits to transductive noise and local data stream bias, resulting in high-frequency spatial oscillations of coefficients across adjacent layers and causing a catastrophic collapse of internal model representations.

---

## 2. Proposed Approach: RCR-Merge
To resolve this paradox, the authors introduce **Riemannian Curvature-Regularized Test-Time Model Merging (RCR-Merge)**. The core philosophy is to model the deep neural network parameter space as a Riemannian manifold where distance is locally scaled by the curvature of the pre-trained base model. The framework consists of three key steps:

1. **Empirical Base Curvature Estimation (Offline):** The diagonal trace of the Fisher Information Matrix (FIM) of the pre-trained base model is estimated once on a tiny, unlabeled calibration set ($|D_{\text{cal}}| = 64$). This trace serves as a highly localized metric of parameter sensitivity ($c_l$) for each layer block $l \in \{1, \dots, L\}$.
2. **Test-Time Adaptation (Online):** The layer-wise merging coefficients $\boldsymbol{\lambda} \in \mathbb{R}^{K \times L}$ are optimized to minimize the Shannon entropy of predicted probability distributions on the local test stream.
3. **Riemannian Curvature-Weighted Total Variation (RCR-TV) and Absolute Anchoring:**
   - **RCR-TV ($\mathcal{R}_{\text{curv}}$):** A second-order spatial regularizer that penalizes the squared difference of merging coefficients between adjacent layers, scaled by the geometric mean of their pre-trained base curvatures ($\sqrt{c_l c_{l-1}}$). This imposes high penalties (coordinate barriers) in sensitive bottlenecks while allowing adaptation in robust layers.
   - **Absolute Anchoring ($\mathcal{R}_{\text{anchor}}$):** An $L_2$ penalty anchoring the coefficients to their initial uniform configuration to prevent global joint-drift under correlated noise.
   - **Gradient Norm Balancing (GNB):** An unsupervised, scale-invariant coordinate re-parameterization technique. It dynamically scales the regularization weights $\beta$ and $\gamma$ at step 0 by evaluating gradient norms under a worst-case spectral perturbation (highest-frequency spatial oscillation eigenvector).

---

## 3. Key Findings
The paper evaluates the proposed method against multiple baselines (Uniform, unconstrained AdaMerging, PolyMerge, and flat TV-Regularized AdaMerging) across 30 independent seeds using a non-convex **Coupled Model II Landscape** simulator representing task sensitivity profiles of MNIST, FashionMNIST, CIFAR-10, and SVHN. 

Key empirical findings include:
- **Confirmation of the Paradox:** Unconstrained AdaMerging collapses representation and drops performance below the static uniform baseline (by up to -10.31% absolute). This collapse occurs even under a completely decoupled, unbiased evaluation metric (Decoupled Isotropic Euclidean metric), confirming it is a physical and robust phenomenon.
- **RCR-Merge Performance:** RCR-Merge successfully resolves the overfitting collapse, outperforming the static Uniform Baseline by **+3.06% absolute** and unconstrained AdaMerging by **+10.33% absolute** on the Coupled metric, while achieving **+3.05% absolute** over Uniform on the Decoupled metric.
- **Superiority of Conformal Soft Barriers over Rigid Subspaces:** On a modular transition landscape (Stage-wise Modular Transition Landscape), RCR-Merge completely outperforms the quadratic trajectory method PolyMerge by **+2.32% absolute** (Coupled) and **+2.44% absolute** (Decoupled), proving that local soft barriers are mathematically superior to rigid global polynomial assumptions which suffer from Runge's phenomenon.
- **Real-World Transferability:** Real-world pilot studies on full-scale architectures (**BERT-Base** with 110M parameters and **ViT-B/16** with 86M parameters) successfully demonstrate actual representation collapse under unconstrained optimization and perfect stabilization under RCR-Merge.

---

## 4. Explicitly Claimed Contributions and Supporting Evidence
The authors explicitly outline several contributions, with corresponding evidence presented throughout the paper:

1. **Formalization of the Overfitting-Optimizer Paradox:** Supported by mathematical modeling of transductive noise propagation and empirical evidence showing unconstrained AdaMerging collapsing across both synthetic landscapes and real-world transformer backbones.
2. **Introduction of RCR-Merge:** Grounded in a block-diagonal scalar FIM approximation and a local Riemannian coordinate Taylor expansion. Detailed pseudo-code and implementation roadmaps are provided.
3. **Formal Theoretical Guarantees:**
   - **Lemma 1 (Coordinate-Level Spatial Barrier):** Proves that adjacent coefficient differences are bounded inversely by the product of layer curvatures.
   - **Theorem 1 (Representation Drift Bounding):** Establishes a mathematical chain-rule proof linking coefficient variations and local FIM curvature to physical intermediate representation drift.
   - **Spectral Analysis:** Formally models RCR-TV as a curvature-guided Laplacian smoothing filter.
4. **Rigorous Empirical Verification:** Demonstrated through a 30-seed simulation study across multiple landscape specifications and evaluation metrics, combined with two real-world multimodal pilot studies (BERT-Base for NLP, ViT-B/16 for Vision).
