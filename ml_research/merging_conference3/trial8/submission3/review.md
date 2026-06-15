# Final Peer Review: Scale-Aligned Quantized Activation Blending (SA-QAB)

## Summary of the Paper
This paper introduces **Scale-Aligned Quantized Activation Blending (SA-QAB)**, a serving framework designed for the efficient, multi-task deployment of merged models (specifically, low-rank LoRA experts) onto resource-constrained edge platforms (e.g., microcontrollers, wearable systems, and IoT devices).

Standard parameter-space model merging methods (such as Post-Merge Quantization (PMQ) and Q-Merge in 4-bit) suffer from complete and catastrophic representation collapse under non-linear sequential architectures (obtaining only 18.60% joint accuracy) because parameter interpolation before non-linear layers (such as GELU) disrupts representation alignment. To overcome this, SA-QAB keeps the shared base backbone and specialized task-specific adapters decoupled during quantization, executing them in their native integer formats, and dynamically blends their activations sample-by-sample on-the-fly.

Specifically, the framework leverages:
1. **Decoupled Heterogeneous Quantization (DHQ):** Aggressively compresses the heavy base model weights to per-channel symmetric INT4 while keeping the lightweight LoRA adapters in per-tensor INT8 to preserve task-specific representation boundaries, reducing active backbone memory by approximately 4x.
2. **Quantized Zero-Shot Centroid Alignment (Q-ZCA):** An early-stage dynamic routing layer operating at Layer 3, computing cosine similarity between INT8-quantized features and pre-computed task centroids using pure integer dot products to route inputs sample-by-sample entirely on the integer manifold.
3. **Quantization Scale Recovery (QSR):** A training-free, calibration-based factor $\beta_k^{(l)}$ computed over 64 calibration samples to align quantized expert activations with their full-precision expectations.
4. **OOD GMM Rejection Gate:** A diagonal-covariance Gaussian Mixture Model trained on Layer 3 features to detect out-of-distribution queries, bypassing adapters and executing queries solely on the frozen base backbone.

The framework is evaluated inside a 14-layer, 192-dimensional synthetic sequential "Coordinate Sandbox" calibrated to represent the empirical difficulty profiles of MNIST, Fashion-MNIST, CIFAR-10, and SVHN. The results show that while the direct post-training quantized version initially exhibited a notable **23.00% absolute accuracy gap** (obtaining **61.90% joint accuracy**), introducing **Quantization-Aware Fine-Tuning (QAT)** for the expert adapters successfully raised the accuracy to **77.50%** (a spectacular **+58.90% absolute accuracy improvement** over standard static PMQ), slashing the representation gap to a mere **7.40%**. 

Furthermore, an expanded real-world pixel-level demonstration using a pre-trained ViT-Tiny model on real MNIST, Fashion-MNIST, CIFAR-10, and SVHN image manifolds recovers a highly robust **84.80% joint accuracy** under aggressive decoupled quantization (DHQ), validating that the sandbox serves as an accurate proxy for actual physical image manifolds.

---

## Strengths
1. **High Practical Utility and Grounding:** The paper directly addresses the severe physical bottlenecks of deploying multi-expert systems on low-power IoT microcontrollers. By focusing on post-training quantization (INT4/INT8), active memory footprints, and compute-scaling costs, the work has immediate, real-world applicability.
2. **Outstanding Systems-Level and Hardware-Aware Design:** The methodology is deeply informed by low-level microcontroller constraints. The mathematical proofs of register accumulator overflow bounds, the formulation of fixed-point dyadic scale-multiplier quantization to bypass expensive on-device division/square roots, and the offline ZCA pre-whitening to enable linear-time $O(D)$ diagonal GMM evaluation represent excellent, high-yield systems-level engineering.
3. **Honest and Thorough Empirical Analysis:** The paper is highly commendable for its scientific honesty. The authors do not overhype their results; they transparently report their initial **23.00% quantization accuracy gap**, and conduct exhaustive sensitivity analyses (sweeping GMM safety thresholds, fallback policies, routing block location, calibration dataset size, and task subspace overlap) to validate every design choice.
4. **Excellent QAT Integration:** Implementing frozen-base QAT for the expert adapters successfully raised the joint accuracy to **77.50%**, showing that the framework can be deployed with unquantized-equivalent representation safety. Because the heavy base backbone remains completely frozen during this phase, the fine-tuning is extremely lightweight and resource-efficient.
5. **Dynamic Serve-Time Modularity:** Unlike static parameter-merging methods, SA-QAB enables on-the-fly task loading and registration. Registering a new specialized expert only requires downloading its lightweight INT8 adapter and its calibration-derived recovery factors, which is of paramount importance for practical, adaptive edge devices.

---

## Weaknesses

### 1. Lack of Physical On-Device Hardware Profiling
While the paper contains outstanding theoretical and simulated resource-overhead analyses (e.g., active SRAM footprints in KB, register accumulator overflow bounds) and reports STM32H7 microcontroller emulation metrics in Table 5, it **lacks actual, physical on-device hardware profiling**. Measuring actual, physical inference latency (in milliseconds), power consumption (in milliwatts), and peak SRAM usage (in KB) on a real microcontroller board (e.g., STM32H753XI) using CMSIS-NN or TensorFlow Lite Micro is necessary to fully confirm the real-world execution speedups of the proposed $O(1)$ active expert compute footprint.

### 2. Heavy Reliance on the Synthetic Coordinate Sandbox
The primary experimental evaluations are conducted inside the synthetic "Coordinate Sandbox." Although the authors include a highly valuable real-world pixel study on ViT-Tiny across a full 4-task suite (MNIST, Fashion-MNIST, CIFAR-10, SVHN) in Section 4.2, scaling the evaluation to full-scale standard architectures (e.g., MobileNetV3, ResNet18) and real-world multi-task datasets would fully verify representational generalizability under complex, correlated real-world image manifolds.

### 3. The Trade-off of Quantization-Aware Fine-Tuning (QAT)
To close the quantization noise gap, SA-QAB requires a 5-epoch fine-tuning phase for the expert adapters (QAT), which compromises its purely training-free post-training quantization (PTQ) status. Although this fine-tuning is extremely lightweight and efficient (the heavy base backbone remains frozen), the paper should explicitly discuss this trade-off between pure PTQ convenience and QAT performance in the main text.

### 4. Flash/Storage Overhead of Scaling Multiple Adapters
While SA-QAB bounds active SRAM memory during execution by selectively routing to a single expert (the $O(1)$ expert compute footprint), all specialized adapters must still be stored on the edge device's non-volatile Flash storage. Although each adapter is small (e.g., rank-8 INT8), as the number of tasks $K$ scales (e.g., to dozens of tasks), the collective Flash storage of these adapters could eventually exceed the hard storage limits of microcontrollers (typically 2MB of Flash on the STM32H7). The authors should include a brief discussion analyzing this storage trade-off as a function of the task registry size.

### 5. Generalizability of Layer Selection for Routing
The routing layer (Q-ZCA) is positioned at Layer 3 based on empirical searches. While this works exceptionally well for the 14-layer sequential network, it remains unclear how developers should systematically select the routing block for deeper or alternative network topologies (e.g., ResNet-like or deeper ViT-Base architectures) without executing exhaustive and expensive layer-wise search loops.

---

## Actionable and Constructive Suggestions

### 1. Conduct Physical On-Device Microcontroller Profiling
- Compile the mixed-precision execution graph (INT4 base + INT8 adapters) onto a real microcontroller platform (e.g., STM32H7 or STM32F7 ARM Cortex-M processor) using CMSIS-NN or an optimized compiler like TinyEngine.
- Measure and report actual, physical inference latency (ms), active power consumption (mW), and peak SRAM/Flash footprints under homogeneous and heterogeneous streams, comparing SA-QAB directly against the ensembling and static merging baselines to validate the emulation results in Table 5.

### 2. Expand the Real-World Pixel-Level Evaluation
- Scale the physical pixel evaluation to standard convolution-based architectures (e.g., MobileNetV3 or ResNet18) and a wider variety of real-world multi-task benchmarks (e.g., Visual Decathlon or physical IoT sensor streams).
- Report the complete classification accuracy, routing specificity, and OOD GMM rejection performance on these physical manifolds, verifying if the sandbox's behavior translates perfectly to diverse deep neural networks.

### 3. Formally Discuss the PTQ-to-QAT Trade-off
- Add a dedicated subsection or discussion in Section 4 exploring the trade-off between pure PTQ and QAT. Specifically, contrast the training-free, zero-overhead nature of direct PTQ (which obtains **61.90%** joint accuracy, or **64.80%** with activation-aware scaling) with the superior accuracy of QAT (**77.50%**), explaining that the 5-epoch frozen-base training step represents a highly practical, low-overhead compromise for accuracy-critical deployments.

### 4. Provide a Guideline for Routing Block Selection
- Elaborate on the mathematical or structural principles that determine why Layer 3 is the optimal routing block. Provide a general heuristic or guidelines for developers to select the routing layer in deeper networks (e.g., using representation entropy, task-distance metrics, or structural depth ratios) without performing exhaustive layer-wise sweeps.

### 5. Analyze Flash Storage Scaling Limits
- Add a quantitative discussion or formula in the hardware analysis section showing the maximum number of expert adapters $K$ that can be concurrently stored in Flash storage alongside the INT4 base backbone on typical microcontrollers (e.g., STM32H7 with 2MB Flash, STM32F7 with 1MB Flash). This will ground the dynamic serving capability in physical hardware limitations.

---

## Ratings

### Soundness: Excellent
The methodology is exceptionally robust, mathematically rigorous, and highly appropriate for edge deployment. The authors have carefully analyzed accumulator register bounds and fixed-point operations, and they are highly honest and transparent about their experimental results and limitations. The hierarchical formulation of the adapter forward pass (explicitly modeling activation quantization and scale propagation) and the transparent disclosure clarifying the calibration input choice demonstrate extreme scientific and physical rigor.

### Presentation: Excellent
The paper is outstandingly written, well-structured, and easy to follow. The mathematical notation is complete and precise. The tables and figures are highly informative, self-contained, and perfectly aligned with the narrative prose, completely eliminating any visual-narrative contradictions.

### Significance: Excellent
The paper addresses a highly important, real-world edge deployment challenge for multi-expert systems. By providing a training-free, forward-pass-only, and modular ensembling solution that preserves stable accuracy while bounding compute and keeping storage small, the work has high practical utility for the TinyML and edge-AI communities.

### Originality: Excellent
The paper creatively and systematically combines post-training quantization, non-parametric routing, and activation scale recovery, tailoring them specifically to the low-level constraints of microcontroller architectures. The systems-level integrations (dyadic conversion, register safety limits, offline ZCA pre-whitening for linear GMMs) represent a significant and high-yield practical innovation.

---

## Overall Recommendation: 5 (Accept)
This is a technically solid, highly thorough, and exceptionally well-written paper. It addresses a highly practical problem of paramount importance to edge-AI deployment and TinyML practitioners. The authors' scientific rigor, mathematical completeness, and transparency regarding their quantization accuracy gap make this a high-quality contribution. Resolving the lack of physical hardware profiling and expanding the real-world pixel study would elevate the work to a Strong Accept, but in its current state, it is already highly solid and ready for publication.
