# Intermediate Evaluation 4: Experimental Evaluation and Results Check

This document provides a critical evaluation of the experimental setup, datasets, baselines, metrics, and whether the results presented in the submission support the claimed findings.

## 1. Evaluation of the Experimental Setup
The empirical evaluation in this paper is highly rigorous and designed with exceptional scientific transparency:

* **30-Seed Statistical Significance:** Unlike many deep learning studies that report results on a single seed or a handful of runs, the authors execute their simulations across **30 independent random seeds**, providing precise mean accuracies and standard deviations. This makes their statistical claims highly robust.
* **Simulator Design and Causal Isolation:** While the primary quantitative results are evaluated on a synthetic **Coupled Model II Landscape** simulator, the authors justify this choice extremely well. The simulator allows them to mathematically isolate and control key physical variables (curating early/late layer sensitivities, dialing the spatial coupling factor $\rho$, and introducing discrete functional stage-boundaries), which would be impossible to control on full-scale real-world networks due to confounding variables.
* **Modality and Architecture Transferability:** To address any skepticism regarding the "simulation-only" constraint, the authors conduct two full-scale, real-world pilot studies using standard pre-trained architectures: **BERT-Base** (110M parameters, NLP modality) and **ViT-B/16** (86M parameters, Vision modality). These pilot studies successfully demonstrate actual representation collapse and stabilization under functional autograd, proving the direct transferability of their findings to real-world architectures.

---

## 2. Baselines and Evaluation Metrics
The choice of baselines and evaluation metrics is comprehensive and designed to prevent experimental bias:

* **Baselines:** The paper compares RCR-Merge against representative and competitive baselines, including static uniform merging (Uniform Baseline), unconstrained test-time adaptation (AdaMerging), rigid structural trajectory projection (PolyMerge), and flat spatial smoothing (TV-Regularized AdaMerging).
* **Breaking Circularity (The Decoupled Metric):** A major strength of the experimental design is the dual-metric evaluation. The standard "Smoothness-Biased Spatially Coupled Covariance" metric incorporates the inverse covariance $\boldsymbol{\Sigma}^{-1}$, which mathematically acts as a 1D graph Laplacian and inherently favors any spatial smoothing method (including RCR-Merge). To break this potential circularity, the authors introduce a **Decoupled Isotropic Euclidean Metric** ($\boldsymbol{\Sigma} = \mathbf{I}$) with zero spatial coupling. 
  The fact that RCR-Merge maintains its outstanding, statistically significant performance margins on this decoupled metric (**90.50%** average accuracy compared to **84.82%** for unconstrained AdaMerging and **87.45%** for Uniform) provides definitive proof of its generalizable stabilization properties.

---

## 3. Support for Central Claims
The presented empirical results fully support each of the paper's central claims:

1. **Verification of the Overfitting-Optimizer Paradox:** Standard AdaMerging collapses catastrophically below the Uniform Baseline on both metrics and across both synthetic and real-world architectures (e.g., dropping to **84.82%** on the decoupled metric, and dropping Task 2 accuracy to random guess on BERT and ViT). This strongly supports the claim that unconstrained optimization fits transductive noise and destroys representation flow.
2. **Superiority of Conformal Soft Barriers over Rigid Subspaces:** On the realistic **Stage-wise Modular Transition Landscape** (representing discrete stage block boundaries of deep modular networks), PolyMerge's global quadratic constraint suffers from Runge's phenomenon, collapsing its accuracy to **91.41%** on the decoupled metric. In contrast, RCR-Merge's local conformal barriers permit sharp, localized transitions, achieving **93.85%** accuracy (an absolute performance advantage of **+2.44%** for RCR-Merge). This provides powerful support for the authors' core architectural arguments.
3. **Robustness of GNB:** Sensitivity sweeps over the scale factor $\alpha$ show that RCR-Merge is exceptionally robust across an entire order of magnitude ($\alpha \in [0.1, 1.0]$), replacing a scale-sensitive, parameter-wise search over $\beta$ with a single, highly stable, scale-invariant multiplier.
