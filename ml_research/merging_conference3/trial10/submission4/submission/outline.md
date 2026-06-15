# Paper Outline: QA-Merge: Quantization-Robust Centroid Routing for Low-Precision Edge Serving

## 1. Title & Metadata
* **Title:** QA-Merge: Quantization-Robust Centroid Routing for Low-Precision Edge Serving
* **Author:** Elena Rostova
* **Affiliation:** Department of Computer Science, ETH Zürich, Switzerland
* **Email:** elena.rostova@inf.ethz.ch
* **ICML Package Option:** `[accepted]` (using fictitious author/affiliation for camera-ready style)

## 2. Abstract
* **Context:** Dynamic model ensembling/merging in Coordinate Sandbox (ICS) has emerged as a lightweight, training-free mechanism for continuous task adaptation.
* **Problem:** Real-world edge serving mandates extreme quantization (INT8 activations, INT4 weights). However, standard ensembling assumes high-precision float32 representations. Naive quantization collapses the representation manifold, leading to overlapping centroids, dead gradients, and severe classification accuracy degradation (dropping back to static Uniform Merging).
* **Proposed Solution (QA-Merge):**
  1. *Quantized Centroid Calibration (QCC):* Calibrates task centroids directly in target integer spaces.
  2. *Straight-Through Estimator (STE) Gating:* Facilitates offline gradient optimization under quantized activations.
  3. *Error-Feedback Trajectory Stabilization (EF-Smooth):* Injects discounted layer-to-layer coefficient rounding errors back into the representation flow as high-pass corrections.
* **Results:** Recovers full-precision ensembling gains under extreme INT8/INT4 limits. Preserves a low jitter footprint (0.00 to 0.05) and scales robustly with few calibration samples.

## 3. Section 1: Introduction
* **The Edge Deployment Imperative:** ML models must be deployed on low-power, cheap edge hardware, requiring quantization (INT8/INT4).
* **The Quantization Collapse Bottleneck:** Dynamic routing depends on activation distances. In integer representation space, rounding ruins subtle representational manifolds. Baselines collapse.
* **Contributions:**
  * Empirical deconstruction of the representation collapse in quantized ensembling.
  * Formalization of QCC, STE Gating, and EF-Smooth.
  * Demonstration of robust, stable, and low-latency dynamic ensembling under INT8/INT4 constraints with zero downstream accuracy loss.

## 4. Section 2: Related Work
* **Dynamic Model Merging & Ensembling:** SABLE, ChemMerge, Momentum-Merge, PAC-Kinetics. Highlighting how they assume high-precision Float32.
* **Neural Network Quantization:** Post-Training Quantization (PTQ), Quantization-Aware Training (QAT), and the Straight-Through Estimator (STE).
* **Pragmatic Edge Serving:** Focus on inference latency, memory bandwidth, and compute-efficient routing solutions.

## 5. Section 3: Methodology
* **Preliminaries:** Symmetric uniform quantization operator:
  $$Q(x, s, b) = \text{clip}\left( \left\lfloor \frac{x}{s} \right\rceil, -2^{b-1}, 2^{b-1}-1 \right)$$
* **Quantized Centroid Calibration (QCC):** Directly computing centroids $c'_k$ on quantized early features (Layer 3) to prevent centroid overlap.
* **Straight-Through Estimator (STE) Gating Optimization:** Gating logits $z_{k, b} = \mathbf{w}_k^T Q(h_b^{(3)}, s_{act}, 8) + b_{g, k}$ optimized via standard autograd using $\frac{\partial \tilde{h}}{\partial h} \approx 1$.
* **Error-Feedback Trajectory Stabilization (EF-Smooth):** Tracking layer-wise coefficient rounding error:
  $$\mathbf{e}_b^{(l)} = \boldsymbol{\alpha}_{b, \text{corrected}}^{(l)} - \tilde{\boldsymbol{\alpha}}_b^{(l)}$$
  and feeding it back recursively with decay $\beta$:
  $$\boldsymbol{\alpha}_{b, \text{corrected}}^{(l+1)} = \boldsymbol{\alpha}_b^{(l+1)} + \beta \mathbf{e}_b^{(l)}$$
  $$\tilde{\boldsymbol{\alpha}}_b^{(l+1)} = \text{ProjectToSimplex}\left( Q(\boldsymbol{\alpha}_{b, \text{corrected}}^{(l+1)}, s_{\alpha}, 4) \right)$$
* **Coordinate Sandbox Integration:** Propagating and blending representations using low-precision integer-arithmetic operations.

## 6. Section 4: Experiments & Analysis
* **Experimental Setup:** High-fidelity Coordinate Sandbox ($D=192$, $L=14$). Simulating MNIST, Fashion-MNIST, CIFAR-10, SVHN signatures with Toeplitz covariance (entanglement $\rho \in [0.0, 0.5]$).
* **Main Results (Table 1 & Table 2):**
  * Small-Sample ($N_{cal} = 64$) and Large-Sample ($N_{cal} = 4000$) classification accuracies across all baselines.
  * Clear demonstration of baseline accuracy collapse to Uniform Merging under naive quantization.
  * Showcasing how QA-Merge recovers full-precision ensembling accuracy.
* **Ablations and Diagnostics:**
  * Sample Complexity curves (Figure 2).
  * Trajectory Jitter comparison (Figure 3) comparing Float32, Quantized-Naive, and QA-Merge versions of SABLE and ChemMerge.
* **The Pragmatist Discussion:** Compute efficiency, edge memory savings, and near-zero latency overhead of the proposed additions.

## 7. Section 5: Conclusion & Future Work
* Summary of the practical significance of QA-Merge.
* Future directions on low-level INT8/INT4 assembly kernel implementations for ARM/NEON and RISC-V microcontrollers.

## 8. Appendix
* Comprehensive derivation of the mathematical bounds for EF-Smooth error accumulation.
* Complete hyperparameter configuration table (scaling factors, temperatures, learning rates, etc.).
