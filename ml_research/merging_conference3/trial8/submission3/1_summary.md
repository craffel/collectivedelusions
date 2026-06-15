# 1. Summary of the Paper

**Title:** Scale-Aligned Quantized Activation Blending: Edge-Robust Multi-Task Model Fusion under Decoupled Quantization

## Core Problem and Motivation
Maintaining separate task-specific expert models (e.g., fine-tuned LoRA adapters) on resource-constrained edge platforms (e.g., microcontrollers, wearable devices, and IoT hardware) is impractical due to severe memory, storage, and computing limits. While model merging (parameter fusion) provides a training-free solution to combine multiple task-specific adapters into a single shared backbone, deploying these merged models on edge devices requires low-bit quantization (e.g., INT4/INT8) to avoid prohibitive memory latency and battery drain. 

Two critical bottlenecks emerge under standard deployment regimes:
1. **Linear Scaling of Ensembling ($O(K)$):** Parallel execution of $K$ experts in activation space scales memory and compute linearly, which is prohibitively expensive for microcontrollers with small active SRAM footprints (often $<1$ MB).
2. **Static vs. Dynamic Modularity:** Weight-space merging (Post-Merge Quantization or PMQ) runs in constant $O(1)$ compute, but collapses catastrophically under realistic non-linear activations (such as GELU) due to severe weight-space scale imbalances and representation misalignment. Furthermore, static merging lacks modularity; registering a new task requires complete offline re-merging and re-quantization.

Existing quantization-aware merging methods (such as Q-Merge) require costly backpropagation-based optimization on calibration datasets and overfit to specific quantization schemas, failing when deployed across different hardware architectures (cross-schema shift).

## Proposed Solution: Scale-Aligned Quantized Activation Blending (SA-QAB)
To resolve these pressing challenges, this paper proposes **Scale-Aligned Quantized Activation Blending (SA-QAB)**, a training-free, forward-pass-only framework designed for robust multi-task edge execution. SA-QAB keeps the heavy shared base model and specialized task adapters separated during quantization, executing them in their native integer formats, and dynamically blends their activations on-the-fly.

SA-QAB is composed of four primary technical components:
1. **Decoupled Heterogeneous Quantization (DHQ):** Compresses the heavy base model weights ($W_{\text{base}}$) to per-channel symmetric 4-bit integer (INT4) format, while keeping lightweight LoRA expert weights in 8-bit integer (INT8) format to preserve task representation boundaries.
2. **Quantized Zero-Shot Centroid Alignment (Q-ZCA):** Performs sample-wise dynamic routing entirely on the integer manifold at an early layer (Layer 3) to prevent circular late-routing dependencies. It computes INT8 cosine similarity between quantized input activations and pre-computed task centroids, followed by a temperature softmax to yield continuous blending coefficients. At compile-time, a hard-argmax pruning is applied to execute exactly one active expert (true $O(1)$ expert compute footprint).
3. **Quantization Scale Recovery (QSR):** A lightweight, training-free calibration step that pre-computes scale recovery factors $\beta_k^{(l)}$ over a small 64-sample dataset to correct for low-bit scale contraction on-the-fly and restore representation fidelity.
4. **Out-of-Distribution (OOD) GMM Rejection Gate:** A pre-whitened diagonal Gaussian Mixture Model (GMM) trained on Layer 3 features to filter OOD noise with $O(D)$ linear-time complexity. Fusing the Zero-phase Component Analysis (ZCA) whitening matrix into the preceding static weights completely eliminates runtime whitening overhead.

## Key Empirical Findings and Results
- **Catastrophic Collapse of Static Merging:** Under non-linear GELU representations, standard post-merge quantization (PMQ) collapses catastrophically to **18.60%** accuracy, and Q-Merge collapses to **22.20%**. 
- **Outstanding SA-QAB Accuracy:** With near-sparse routing, SA-QAB achieves **77.50%** joint mean accuracy, representing a spectacular **+58.90% absolute accuracy improvement** over PMQ.
- **Microcontroller Feasibility:** Emulated on an STM32H7 microcontroller, SA-QAB requires only 360.75 KB of active SRAM ($>60\%$ headroom) and achieves a **2.3x speedup** and **57% energy savings** over parallel ensembling with a negligible **3.7% latency overhead** over the collapsed static model.
- **Physical Pixel Feasibility:** Evaluated on real image manifolds using a pre-trained ViT-Tiny backbone on MNIST, Fashion-MNIST, CIFAR-10, and SVHN, SA-QAB recovers a robust joint multi-task classification accuracy of **84.80%**, confirming that the synthetic sandbox serves as a highly precise proxy.
- **Dynamic Modularity and Storage Scaling:** SA-QAB enables on-the-fly task loading, supporting up to **66 concurrent task experts** under a standard 2MB Flash limit.
