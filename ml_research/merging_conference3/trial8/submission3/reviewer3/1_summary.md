# 1. Summary of the Paper

## Main Topic and Goal
The paper introduces **Scale-Aligned Quantized Activation Blending (SA-QAB)**, which is designed to address the challenges of deploying merged multi-task models on resource-constrained edge hardware. The main goal is to combine specialized models (e.g., task-specific LoRA experts) on a shared backbone while executing the merged model under aggressive low-bit integer quantization (INT4/INT8) without catastrophic accuracy degradation, scale drift, or $O(K)$ linear scaling of compute/memory when executing $K$ experts.

## Proposed Approach (SA-QAB)
To achieve this, the paper proposes a training-free, forward-pass-only activation blending framework consisting of:
1. **Decoupled Heterogeneous Quantization (DHQ):** The heavy shared base model is compressed to INT4 (per-channel) to save memory bandwidth, while the task-specific LoRA adapters are kept in INT8 (per-tensor) to preserve representational capacity.
2. **Quantized Zero-Shot Centroid Alignment (Q-ZCA):** An early-stage, integer-space sample-wise routing layer (placed at Layer 3 of 12) that routes incoming inputs to task experts using INT8 cosine similarity based on pre-computed task centroids. This routing allows sparse execution ($O(1)$ adapter compute) via hard-argmax.
3. **Quantization Scale Recovery (QSR):** A calibration step that pre-computes scaling factors $\beta_k^{(l)}$ offline (using 64 samples) to correct low-bit scale contraction on-the-fly and restore representational balance between the INT4 base and INT8 adapter paths.
4. **Out-of-Distribution (OOD) GMM Rejection Gate:** A pre-whitened diagonal Gaussian Mixture Model (GMM) trained offline on Layer 3 activations that acts as a gate. If the input log-likelihood is below a threshold $\eta$, the input is routed solely through the frozen base backbone, avoiding routing noise or wasting expert computation.

## Key Findings and Claims
- **Performance in GELU Sandbox:** Standard Post-Merge Quantization (PMQ) collapses catastrophically to **18.60%** accuracy in a 192D synthetic "Coordinate Sandbox" with non-linear GELU layers. SA-QAB recovers this performance to **77.50%** accuracy (with a 5-epoch frozen-base QAT) or **70.10%** (with training-free SmoothQuant-like pre-scaling), showing a massive improvement over static baselines.
- **Microcontroller Feasibility:** Emulation profiles on an **STM32H753XI** microcontroller suggest SA-QAB uses only **360.8 KB SRAM** (compared to 1224.8 KB for FP16 Ensembling, which exceeds the 1MB limit), achieving **0.836 ms latency** and **0.3035 mJ energy** (a 2.3x speedup and 57% energy savings over ensembling).
- **Dynamic Task Modularity:** Storing the base and task experts separately allows on-the-fly task loading. Flash storage scaling analysis ($M(K) = M_{\text{base}} + K \times M_{\text{adapter}}$) shows that a 2MB Flash microcontroller can store up to 66 concurrent experts.
- **Portability and Real Pixel Feasibility:** SA-QAB generalizes across hardware configurations without re-merging. A real-pixel evaluation on a physical ViT-Tiny ($D=192$, 12 layers) on a 4-task suite (MNIST, Fashion-MNIST, CIFAR-10, SVHN) yields **84.80% joint accuracy** under INT4/INT8 quantization, demonstrating that the synthetic sandbox generalizes.

## Explicitly Claimed Contributions
1. **DHQ Scheme:** Heterogeneous INT4/INT8 quantization to compress the backbone by 4x while preserving expert representations.
2. **QSR Alignment:** Offline training-free calibration to resolve scale mismatches.
3. **Q-ZCA Routing:** Low-power, early-stage integer-space dynamic routing at Layer 3.
4. **Comprehensive Systems and Empirical Analysis:** Evaluation in the Coordinate Sandbox, real-world ViT-Tiny pixels, ResNet-18 convolutional extension, and STM32H7 hardware resource-constraint emulation.
