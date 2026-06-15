# Paper Outline: Scale-Aligned Quantized Activation Blending (SA-QAB)

## Title:
**Scale-Aligned Quantized Activation Blending: Edge-Robust Multi-Task Model Fusion under Decoupled Quantization**

## Authors & Affiliation (Fictional):
- **Author:** Arthur Vance
- **Affiliation:** Department of Electrical Engineering and Computer Sciences, University of California, Berkeley
- **Email:** arthur.vance@berkeley.edu

## Abstract
- **Context:** Model merging allows dynamic execution of multiple tasks by blending parameter weights or activations. However, deploying merged models on resource-constrained edge platforms necessitates low-bit quantization.
- **Problem:** Conventional parameter-space model merging followed by quantization crashes multi-task representation spaces because of severe weight-space scale imbalances and quantization noise when sequential layers are separated by non-linear activations. Meanwhile, existing quantization-aware model-merging techniques (such as Q-Merge) overfit to specific quantization operators and require costly, non-portable backpropagation.
- **Method:** We propose **Scale-Aligned Quantized Activation Blending (SA-QAB)**, a training-free, forward-only framework designed for edge-robust model fusion. 
- **Core Ideas:** 
  1. **Decoupled Heterogeneous Quantization (DHQ):** Aggressively compresses the heavy base model to INT4 while preserving task experts in INT8.
  2. **Quantization Scale Recovery (QSR):** A lightweight calibration step that computes scale-alignment factors offline to neutralize scale drift between INT4 and INT8 representations.
  3. **Quantized Zero-Shot Centroid Alignment (Q-ZCA):** Realizes routing entirely within the INT8 manifold at an early layer to prevent circular late-routing dependencies.
- **Results:** Evaluated on a 192D synthetic Coordinate Sandbox with non-linear GELU activation layers, standard post-merge quantization (PMQ) collapses catastrophically to **18.60%** accuracy. In contrast, SA-QAB achieves **77.50%** joint mean accuracy, representing a spectacular **+58.90% absolute accuracy improvement** over PMQ, eliminates batch-size heterogeneity collapse entirely, maintains absolute stability across cross-schema hardware shifts, and reduces base backbone memory overhead by ~4x.

## 1. Introduction
- The real-world deployment challenges of multi-task systems on edge and IoT devices (memory and latency constraints).
- The limitations of parameter-space merging under low-bit quantization and non-linearities (weight scale corruption, outliers, and schema shift).
- Introducing SA-QAB: a decoupled, scale-aligned, forward-only activation blending approach.
- Listing four key practical contributions (DHQ, QSR, Q-ZCA, Empirical Validation).

## 2. Related Work
- Model Merging and Dynamic Adaptation.
- Quantization in Deep Learning & Quantization-Aware Merging (Q-Merge).
- Dynamic Routing & Single-Pass Adaptation (SPS-ZCA, SABLE).

## 3. Methodology
- **3.1 Decoupled Heterogeneous Quantization (DHQ):** Symmetric per-channel INT4 for base model $W_{\text{base}}$, per-tensor INT8 for adapters $A_k, B_k$.
- **3.2 Quantized Zero-Shot Centroid Alignment (Q-ZCA):** Integer-space cosine similarity at Layer 3 against INT8-quantized centroids, temperature softmax for dynamic coefficients $\alpha_{k, b}$.
- **3.3 Quantization Scale Recovery (QSR):** Scale calibration factors $\beta_k^{(l)}$ to align INT8 adapter activations with INT4 base activations based on a tiny 64-sample calibration set.
- **3.4 Single-Pass Quantized Activation Blending:** Complete forward-pass blending formula with scale recovery.

## 4. Experiments and Results
- **4.1 Sandbox Environment Setup:** Vision Transformer simulation suite (Coordinate Sandbox), tasks (MNIST, F-MNIST, CIFAR-10, SVHN).
- **4.2 Accuracy Sweep under Diverse Streams:** Compare against Expert Ceiling, PMQ, Q-Merge, and SPS-ZCA. Highlight catastrophic collapse of static baselines and +58.90% improvement of SA-QAB.
- **4.3 Robustness to Cross-Schema Shifts:** Evaluating uncalibrated/calibrated shift from INT4 to INT8.
- **4.4 Out-of-Distribution (OOD) Task Rejection:** 192D diagonal GMM density estimator sweep, showing optimal threshold of $\eta=-255.0$ with 99.2% TPR and 2.4% FRR, and balanced discussion of its limitations.
- **4.5 Discussion of Visualized Artifacts:** Direct references to figures (`results/fig1.png`, `results/batch_size_heterogeneity.png`, `results/rejection_roc_curve.png`).

## 5. Conclusion and Discussion
- Summary of benefits (Pragmatic, training-free, stable, highly memory-efficient).
- Future real-world hardware integration and limits.
