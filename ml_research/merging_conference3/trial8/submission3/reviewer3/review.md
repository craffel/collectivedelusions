# Peer Review Report

---

## 1. Summary of the Paper

This paper introduces **Scale-Aligned Quantized Activation Blending (SA-QAB)**, a training-free, forward-pass-only framework designed to enable the deployment of merged multi-task models on resource-constrained edge hardware. Standard model merging methods (like Post-Merge Quantization or PMQ) collapse catastrophically when compressed to low-bit formats in the presence of non-linear activations (e.g., GELU). Meanwhile, parallel expert execution (ensembling) scales compute and SRAM linearly ($O(K)$), which exceeds the strict limits of microcontrollers (typically $<1$MB SRAM).

To resolve these pressing bottlenecks, the paper proposes:
1. **Decoupled Heterogeneous Quantization (DHQ):** Squeezing the heavy, shared base model to INT4 while preserving task-specific LoRA adapters in INT8, reducing active backbone memory by approximately 4x.
2. **Quantized Zero-Shot Centroid Alignment (Q-ZCA):** An early-stage, integer-only sample-wise routing layer placed at Layer 3 of 12. It computes cosine similarities against pre-computed task centroids entirely within the INT8 manifold, allowing sparse execution via hard-argmax ($O(1)$ expert compute footprint).
3. **Quantization Scale Recovery (QSR):** A calibration step that pre-computes scaling factors $\beta_k^{(l)}$ offline on a tiny 64-sample set to correct low-bit scale contraction on-the-fly and restore representational balance between the INT4 base and INT8 adapter paths.
4. **Out-of-Distribution (OOD) GMM Rejection Gate:** A pre-whitened diagonal Gaussian Mixture Model (GMM) trained offline on Layer 3 activations that routes out-of-distribution inputs solely through the robust base backbone, avoiding routing noise or wasting expert computation.

**Key Findings:** Evaluated in a 192-dimensional synthetic "Coordinate Sandbox" calibrated to represent image classification profiles (MNIST, F-MNIST, CIFAR-10, SVHN) under non-linear GELU layers, standard post-merge quantization collapses to **18.60%** joint accuracy. In contrast, SA-QAB achieves **77.50%** accuracy (with a 5-epoch frozen-base QAT) or **70.10%** (with training-free SmoothQuant-like pre-scaling). It also eliminates batch heterogeneity collapse entirely. High-fidelity emulation on an STM32H7 microcontroller shows SA-QAB runs within **360.8 KB active SRAM** (compared to 1224.8 KB for FP16 Ensembling), with **0.836 ms latency** and **0.3035 mJ energy** per inference. Finally, a real-pixel feasibility study using a physical ViT-Tiny and a convolutional extension on ResNet-18 confirm that these representational principles scale to real image manifolds.

---

## 2. Strengths and Weaknesses

### Soundness (Rating: Good)
* **Strengths:** 
  * The mathematical formulation of the quantization operators, scale propagation, dynamic routing, and scale recovery factors is highly rigorous and complete.
  * The integer-only approximation of cosine similarity (pre-normalizing centroids offline and using fixed-point bit-shifting to bypass expensive divisions and square roots) is an excellent engineering adaptation for physical microcontrollers.
  * The authors are highly transparent and scientifically honest about their design decisions, explicitly disclosing that they extract QSR calibration activations from clean FP16 streams to avoid compounding noise (despite introducing a minor distribution mismatch at test-time).
* **Weaknesses:**
  * There is a slight tension between the "training-free, forward-only" pitch and the reality that achieving peak performance (**77.50%** joint accuracy) requires a 5-epoch Quantization-Aware Fine-Tuning (QAT) phase on the experts. While the training-free pre-scaling alternative (**70.10%**) is strong, a practitioner must still accept a significant accuracy gap if they cannot run a GPU-bound training loop.

### Presentation (Rating: Excellent)
* **Strengths:**
  * The paper is exceptionally well-written, clear, and structured. 
  * The figures and tables are informative and highly readable. 
  * The "Transparent Disclosure" blocks regarding synthetic data, dimensional isomorphism, and calibration input choices are commendable and provide a model for scientific writing.
* **Weaknesses:**
  * None identified. The presentation quality is exemplary.

### Significance (Rating: Excellent)
* **Strengths:**
  * This work has outstanding practical utility. Bridging the gap between model merging and post-training quantization is a major open bottleneck for TinyML practitioners.
  * The dynamic task registry capability (allowing new INT8 adapters and QSR scales to be loaded and registered on-the-fly without touching the frozen INT4 base or other experts) is a huge operational advantage for modular on-device software engineering.
  * The detailed systems-level profiling—including active SRAM footprint, non-volatile Flash limits, MAC operations, and energy consumption—is exactly what hardware developers need to validate real-world feasibility.
* **Weaknesses:**
  * **The Flash-to-SRAM Paging Overhead:** The authors point out that up to 66 concurrent experts can be stored in the 2MB Flash memory of an STM32H7 microcontroller. However, because the 1MB SRAM cannot hold all 66 adapters simultaneously alongside the base model, sample-wise dynamic routing must copy the selected 27.2 KB adapter weights from Flash to SRAM on-the-fly when the routing target switches. The paper completely omits any analysis or discussion of this **Flash-to-SRAM weight-paging latency**, which on a typical microcontroller can take several milliseconds and potentially obliterate the 0.03 ms emulated routing latency.

### Originality (Rating: Good)
* **Strengths:**
  * While the individual mathematical components (mixed-precision quantization, routing via cosine similarity, GMM-based outlier detection, scale recovery ratios) are established concepts, their **creative combination and optimization for low-power microcontrollers is highly original and clever**.
  * The adaptation of SABLE/SPS-ZCA concepts into an integer-only, CMSIS-NN compatible pipeline is a strong technical contribution.
* **Weaknesses:**
  * The conceptual "delta" on the theoretical side is somewhat incremental, as it primarily ports existing activation-blending and scale-alignment methods into a lower-precision integer framework.

---

## 3. Detailed Ratings and Justifications

### Soundness: Good
The theoretical basis is robust, the equations are complete, and the baseline comparisons are thorough. The minor limitation is that peak accuracy remains reliant on a lightweight GPU-based training phase (QAT) rather than being completely post-training.

### Presentation: Excellent
The writing is clear, logical, and highly descriptive. The explicit disclosures of limitations and methodological trade-offs are incredibly refreshing and make the paper highly trustworthy.

### Significance: Excellent
The paper tackles a very real, high-impact problem in edge computing and TinyML. The systems trade-off analysis, hardware-aligned memory-scaling formulas, and dynamic modularity insights are of paramount importance to practitioners.

### Originality: Good
The work presents a highly successful engineering convergence of model merging, post-training quantization, and low-power routing. The hardware-efficient implementation details show highly original systems-level design.

---

## 4. Overall Recommendation

**Rating: 5 (Accept)**

**Justification:**
This is a technically solid, highly practical paper that addresses a crucial bottleneck in deploying multi-task merged models to resource-constrained edge hardware. The authors provide a compelling, hardware-aligned solution (SA-QAB) that achieves near-static execution efficiency ($O(1)$ expert compute and low SRAM) while successfully recovering from the catastrophic representation collapse that plagues static post-merge quantization on non-linear layers. 

What makes this paper stand out is its deep commitment to physical feasibility: the cycle-accurate STM32H7 microcontroller emulation, the PyTorch host CPU profiling, the task-overlap stress tests, and the real-pixel ViT-Tiny and convolutional ResNet-18 evaluations provide a level of validation that is rare and highly valuable for practitioners. While there are minor systems-level omissions (namely, the paging copy latency from Flash to SRAM during sample-wise routing shifts), they do not overshadow the substantial contributions, systems insights, and overall excellence of the work.

---

## 5. Constructive Feedback and Questions for the Authors

1. **Flash-to-SRAM Weight-Copy Latency:** 
   In Section 4.2, you show that up to 66 concurrent experts can be stored in the 2MB Flash memory of an STM32H753XI. However, these 66 adapters ($66 \times 27.2\text{ KB} \approx 1.8\text{ MB}$) cannot all reside in the 1MB SRAM simultaneously alongside the base model. When Q-ZCA performs dynamic sample-wise routing and switches active experts on-the-fly, the system must page/copy the chosen 27.2 KB adapter weights from non-volatile Flash into active SRAM. Could you discuss or quantify this Flash-to-SRAM memory transfer latency? In standard embedded microcontrollers, copying 27.2 KB of weights over an internal bus can take several milliseconds, which might dominate the 0.836 ms inference time and limit sample-wise routing to block-wise or stream-wise batching in practice.
   
2. **On-Device Local Adaptation Constraints:**
   Since peak joint accuracy (**77.50%**) depends on a 5-epoch GPU-based QAT phase, local on-device adaptation is difficult if the edge device lacks backpropagation capability. If a practitioner needs to load a new expert on-the-fly entirely "offline" (without GPU training), they must rely on the **70.10%** training-free activation-aware scaling option. Could you provide a clear set of guidelines or trade-offs for developers deciding between the training-free PTQ + pre-scaling flow and the QAT flow when deploying new, uncalibrated experts?

3. **Sensitivity and Stability of the GMM Threshold:**
   In Section 4.5, you mention that the pre-whitened diagonal GMM gate achieves optimal OOD discrimination at a safety threshold of $\eta = -255.0$. How stable is this threshold when transitioning across different hardware targets, or when the inputs are subjected to varying levels of physical noise (e.g., camera sensor noise or low-light artifacts) in real-world environments? Is there a recommended online calibration technique for adjusting $\eta$ dynamically on-device?
