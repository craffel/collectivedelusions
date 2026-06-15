# Outline: Empirical Robustness in Test-Time Dynamic Model Merging via Confidence-Gated Hybrid Routing and Micro-Batch Homogenization

## Title
Empirical Robustness in Test-Time Dynamic Model Merging via Confidence-Gated Hybrid Routing and Micro-Batch Homogenization

## Authors & Affiliation
Emily Chen (UC Berkeley) - Fictional Identity

## Section-by-Section Outline

### 0. Abstract
* **Context**: Dynamic model merging (blending specialized experts at inference time) offers high model capacity without growing deployment foot-prints.
* **Problem**: Standard parametric routers overfit severely under calibration data scarcity and suffer from "heterogeneity collapse" under mixed-task batch deployment streams.
* **Method**: Confidence-Gated Hybrid Routing (CGHR) dynamically falls back from a parametric router to a zero-shot parameter-free subspace router (PFSR) when router confidence is low. Combined with Micro-Batch Homogenization (MBH) to solve batch heterogeneity collapse.
* **Empirical Focus (The Empiricist)**: Comprehensive evaluation over 5 independent random seeds, sweeping confidence thresholds, sample complexities ($N=16$ to $512$), confidence metrics, and batch deployment stream scales ($B=1$ to $512$).
* **Outcome**: CGHR + MBH preserves high expert-level performance, avoids overfitting in low-data regimes, and completely prevents heterogeneity collapse.

### 1. Introduction
* High-level motivation for dynamic model merging (test-time expert blending).
* Real-world issues under the lens of empirical rigor:
  1. **Calibration Data Scarcity (N)**: Small sample calibration leads to massive transductive overfitting.
  2. **Batch Heterogeneity Collapse (B)**: Mixed batches cause standard routers to output flat, averaged weights across experts, degrading performance to uniform merging.
* Introduce our proposed dual-pathway routing system: **Confidence-Gated Hybrid Routing (CGHR)**.
* Highlight our core empirical contribution: A massive grid sweep across thresholds, seeds, sample sizes, and batch streams to map the complete transition from parametric flexibility to zero-shot robustness.

### 2. Related Work
* **Model Merging**: Weights interpolation (Task Arithmetic, TIES, Dare) vs. dynamic test-time blending.
* **Dynamic Routing and MoEs**: Routing networks, gating mechanisms, and their susceptibility to collapse.
* **Regularized and Parameter-Free Routing**: State-of-the-art TSAR (anchoring to expert centroids), VR-Router, and PFSR (projection-based, training-free routing).
* **The Empirical Gap**: Emphasize how prior work lacks multi-seed robustness analysis under extreme small-$N$ calibration or varying stream compositions.

### 3. Methodology
* **System Model: The Isolating Coordinate Sandbox** ($L=1$, $D=192$, $K=4$, $C=10$). Explain how Unit-Norm Calibration (UNC) and noise-calibrated features reconstruct realistic expert ceilings.
* **Pathway A: Parametric Gating (Flexible but Vulnerable)**:
  * Logits $a_b = W_{\text{param}} z_b + b_{\text{param}}$.
  * Softmax-activated routing weights $\alpha^{\text{param}}_b$.
* **Pathway B: Robust Parameter-Free Subspace Routing (PFSR)**:
  * Projections using cosine similarity $u'_{k, b} = \frac{\max_{c} \frac{W_{k, c} \cdot z_{k, b}}{\|W_{k, c}\|_2 \|z_{k, b}\|_2}}{\sqrt{2\log C_k / d}}$.
  * Zero trainable parameters, completely stable.
* **Confidence Gating Mechanism**:
  * Define confidence $\mathcal{C}(\alpha^{\text{param}}_b)$ under Max Probability, Negative Entropy, and Margin.
  * Apply threshold $\gamma_{\text{conf}}$ to dynamically select routing pathway sample-by-sample.
* **Micro-Batch Homogenization (MBH)**:
  * Grouping heterogeneous stream batches into uniform sub-batches dynamically on-the-fly.
  * Disabling the batch-averaging representation smoothing effect.

### 4. Experimental Setup & Results
* **Experimental Environment**: Sandbox details, expert ceiling calibration (MNIST 100%, F-MNIST 100%, CIFAR-10 88.6%, SVHN 26.4%).
* **Baselines**: Uniform, Linear (Unreg), Linear (Reg, $L_2$), VR-Router, TSAR, PFSR.
* **Main Quantitative Table**: Main results across 5 seeds under standard calibration ($N=64, B=256$).
* **Analysis 1: Confidence Gating Threshold Sensitivity (Figure 1)**:
  * Sweep $\gamma_{\text{conf}} \in [0.0, 1.0]$ across Max-Prob, Negative Entropy, and Margin.
  * Identify the optimal peak performance envelope.
* **Analysis 2: Generalization Under Data Scarcity (Figure 2)**:
  * Sweep $N \in \{16, 32, 64, 128, 256, 512\}$.
  * Show CGHR's extreme stability under low-data constraints.
* **Analysis 3: Robustness to Heterogeneity Collapse (Figure 3)**:
  * Sweep $B \in \{1, 8, 32, 128, 512\}$.
  * Demonstrate how MBH completely flattens the collapse curve.

### 5. Conclusion & Future Work
* Reiterate that exhaustive empirical tests validate CGHR + MBH.
* Position this work as a robust, deployment-ready standard for real-world dynamic model merging.
* Propose expanding the empirical sweeps to larger transformer backbones.
