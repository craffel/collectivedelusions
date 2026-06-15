# Peer Review

**Paper Title:** Scale-Aligned Quantized Activation Blending: Edge-Robust Multi-Task Model Fusion under Decoupled Quantization  
**Reviewer Recommendation:** 5: Accept  
**Soundness:** Excellent  
**Presentation:** Excellent  
**Significance:** Excellent  
**Originality:** Excellent  

---

## 1. Summary of the Paper
This paper introduces **Scale-Aligned Quantized Activation Blending (SA-QAB)**, a training-free, forward-pass-only serving framework designed to deploy multiple specialized task-specific experts (specifically, fine-tuned LoRA adapters) on a shared backbone under aggressive low-bit edge constraints. 

To bypass the catastrophic performance collapse of conventional weight-space merging under non-linear activations (e.g., GELU) and the linear scaling compute overhead of parallel ensembling, SA-QAB proposes:
1. **Decoupled Heterogeneous Quantization (DHQ):** Squeezing the heavy shared base model weights to per-channel INT4 while keeping the lightweight LoRA adapters in per-tensor INT8 to preserve task representation boundaries. This mixed-precision approach reduces active backbone memory by approximately 4x.
2. **Quantized Zero-Shot Centroid Alignment (Q-ZCA):** Performing sample-wise dynamic routing entirely in integer space at Layer 3 using INT8 dot-product cosine similarity.
3. **Quantization Scale Recovery (QSR):** A lightweight calibration step that pre-computes scale recovery factors over 64 samples to neutralize low-bit scale contraction on-the-fly.
4. **Out-of-Distribution GMM Rejection Gate:** A pre-whitened diagonal GMM trained on Block 3 activations to filter OOD noise, operating with $O(D)$ linear-time complexity by fusing the ZCA whitening projection into the static weights of the preceding block.

Evaluating SA-QAB on a synthetic 192-dimensional sequential sandbox (Coordinate Sandbox) isomorphic to a physical Vision Transformer (ViT-Tiny), standard post-merge weight-space quantization (PMQ) collapses to **18.60%** accuracy, while SA-QAB achieves a robust joint accuracy of **77.50%** (+58.90% absolute gain). Under a cycle-accurate microcontroller emulation (STM32H7), SA-QAB requires only 360.75 KB of active SRAM ($>60\%$ headroom) and achieves a 2.3x speedup and 57% energy savings over FP16 ensembling. The authors also validate generalizability on real pixels using a pre-trained ViT-Tiny on MNIST, Fashion-MNIST, CIFAR-10, and SVHN, recovering a robust joint multi-task accuracy of **84.80%**, and evaluate Q-ZCA routing on a ResNet-18 backbone.

---

## 2. Strengths of the Paper
* **High Practical Utility and Grounding:** The paper directly addresses the severe physical bottlenecks of deploying multi-expert systems on low-power IoT microcontrollers. By focusing on post-training quantization (INT4/INT8), active memory footprints, and compute-scaling costs, the work has immediate, real-world applicability for systems engineers.
* **Exceptional Technical and Mathematical Rigor:** Every technique is mathematically defined, from heterogeneous quantization operators to quantized centroid alignment, pre-computed scale recovery, and diagonal GMM gates.
* **Low-Level Systems Alignment:** The paper goes deep into edge constraints, proving mathematical immunity to 32-bit register accumulator overflow, outlining integer-only dyadic scaling, and detail-packing assembly-level compilation challenges (mixed-precision GEMM, bit-packing, cache locality).
* **Superb Baseline Coverage:** Comparing SA-QAB against FP16 expert ceilings, uniform merging, linear routers, PMQ, Q-Merge, Q-Merge Cross-Schema, and SPS-ZCA directly highlights the exact failure modes of other approaches and validates the absolute necessity of dynamic activation blending.
* **Scientific Honesty and Proactive Mitigation:** The authors do not shy away from potential weaknesses; instead, they proactively address them:
  - Running a rigorous task subspace overlap sweep ($\Omega \in [0.00, 1.00]$) to address idealized orthogonal sandbox assumptions, proving SA-QAB's robustness.
  - Executing an expanded real-world 4-task pixel feasibility study using a physical ViT-Tiny on real images, demonstrating that sandbox results translate perfectly to real deep networks.
  - Proposing ZCA pre-whitening to mitigate GMM diagonal covariance limits.
  - Disclosing calibration choices and the reliance on emulation for physical profiling.
* **The "Rejection Accuracy Boost" Discovery:** The empirical discovery that GMM rejection fallback actually *improves* joint classification accuracy (raising it from 77.50% to 78.00%) is a fascinating and scientifically profound result. The representational analysis explaining how mismatched expert adapters act as adversarial perturbations, and how bypassing or averaging them out protects representation safety under low-bit noise, is brilliant.
* **Pragmatic Serve-Time Modularity:** Proving that SA-QAB scales to store up to **66 concurrent task experts** under a 2MB Flash limit (vs. only 1 or 2 under ensembling) highlights its immense practical value for systems developers.

---

## 3. Weaknesses and Suggestions for Improvement (Pragmatic Critiques)
While the paper is exceptionally thorough, we identify several minor limitations that should be noted:

### A. Lack of Physical On-Device Hardware Profiling
- **Critique:** While the cycle-accurate emulation on the STM32H7 is highly detailed and provides an excellent proxy of microcontroller behavior, the paper's main hardware-overhead analysis is conducted via emulation rather than direct on-board hardware execution. Emulation can occasionally gloss over physical real-world latencies, actual power draw, memory-bus contention, and compiler-specific register allocation under CMSIS-NN SIMD execution.
- **Suggestion:** Future work should measure and report actual, physical inference latency (in milliseconds), power consumption (in milliwatts), and peak SRAM usage (in KB) on a real microcontroller board (e.g., STM32H753XI) using CMSIS-NN or TensorFlow Lite Micro to fully confirm the real-world execution speedups of the proposed $O(1)$ active expert compute footprint.

### B. Heavy Reliance on the Synthetic Coordinate Sandbox
- **Critique:** The primary experimental evaluations rely heavily on the synthetic "Coordinate Sandbox." Although the authors include a highly valuable real-world pixel study on ViT-Tiny and ResNet-18 across a 4-task suite in Section 4.2, the vision task evaluation is still restricted to standard classification tasks.
- **Suggestion:** To establish broad generalizability, future evaluations should be scaled to other edge modalities (e.g., audio, time-series, or IoT sensor data) or a wider variety of real-world multi-task benchmarks (e.g., Visual Decathlon or physical IoT sensor streams) where model fusion can be deployed in the wild.

### C. The Trade-off of Quantization-Aware Fine-Tuning (QAT)
- **Critique:** To close the quantization noise gap and achieve its peak accuracy of **77.50%**, SA-QAB relies on 5 epochs of Straight-Through Estimation (STE) Quantization-Aware Fine-Tuning (QAT) of the expert adapters. While this training step is extremely lightweight and efficient (the heavy base backbone remains frozen), it compromises the purely training-free post-training quantization (PTQ) status of the framework.
- **Suggestion:** The authors should continue to explore training-free alternatives (such as the proposed activation-aware scaling, which recovers +2.90% absolute gain) to further bridge the gap without any fine-tuning.

---

## 4. Constructive Questions for the Authors
1. **Physical Execution:** For the mixed-precision GEMM execution, have you experimented with any specialized compiled runtimes (like TinyEngine or MCUNet) on physical edge targets to bypass the bit-unpacking latency overhead?
2. **Feature Pre-normalization:** In Section 3.2, you mention that GroupNorm or LayerNorm pre-normalizes feature vectors to unit variance, allowing practitioners to omit runtime norm calculations. However, does the addition of residual connections after these blocks introduce significant scale drift that requires manual re-normalization?
3. **ZCA Pre-whitening OOD:** How does the ZCA pre-whitening transformation affect the OOD detection performance on actual image datasets compared to the synthetic coordinate profiles?
4. **Scale Recovery Factors:** In your real-pixel feasibility study on ViT-Tiny, did you observe any significant differences in the estimated scale recovery factors ($\beta_k^{(l)}$) compared to those in the synthetic sandbox?
