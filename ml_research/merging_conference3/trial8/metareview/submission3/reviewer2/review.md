# Peer Review

## Summary of the Paper
The paper addresses the challenge of deploying multi-task merged models onto resource-constrained edge platforms (e.g., low-power microcontrollers) under tight memory and compute constraints. Standard model merging techniques average weights in parameter space to combine multiple specialized adapters (such as LoRA) into a single backbone, but this approach collapses when combined with post-training quantization (PTQ) under non-linear activation functions (e.g., GELU). 

To resolve this, the paper proposes **Scale-Aligned Quantized Activation Blending (SA-QAB)**. Rather than fusing weights, SA-QAB keeps the base model and task-specific adapters separate, executing them in their native integer formats on the device and dynamically blending their activations on-the-fly. The framework includes:
1. **Decoupled Heterogeneous Quantization (DHQ)**: 4-bit integer (INT4) base backbone and 8-bit integer (INT8) expert adapters.
2. **Quantized Zero-Shot Centroid Alignment (Q-ZCA)**: An early-stage (Layer 3) dynamic routing layer that computes cosine similarity between quantized features and task centroids in integer space. Using a low temperature, it performs sparse routing to execute only the single active expert pathway.
3. **Quantization Scale Recovery (QSR)**: Pre-computed scaling factors matching unquantized and quantized activations over a small calibration set to correct for scale contraction.
4. **OOD GMM Rejection Gate**: A diagonal Gaussian Mixture Model in Layer 3 representation space to bypass routing and fallback to the base model for out-of-distribution inputs.

The paper evaluates SA-QAB on a synthetic 192-dimensional Coordinate Sandbox (calibrated to represent MNIST, Fashion-MNIST, CIFAR-10, and SVHN noise profiles) and presents a brief real-pixel feasibility study. It also reports hardware-simulation profiling on an STM32H7 microcontroller.

---

## Major Strengths

1. **Practical Systems Motivation (TinyML Relevance)**: 
   The paper addresses a highly relevant problem for edge-computing and TinyML: how to deploy multiple specialized tasks on low-power, memory-restricted hardware.
2. **Actionable Systems and Hardware-level Details**:
   The paper provides useful optimizations designed specifically for microcontroller constraints, such as employing hard-argmax for sparse $O(1)$ expert execution, pre-normalizing centroids offline and using fixed-point bit-shifting to avoid expensive online divisions/square roots, and restricting GMM covariances to diagonals to reduce computational complexity to $O(D)$ linear-time arithmetic.
3. **Thorough Systems Trade-off and Memory Analysis**:
   The authors provide an explicit analysis of flash memory and active SRAM constraints (Section 4.2), calculating how many adapters can fit on typical microcontroller targets (e.g., up to 66 experts on a 2MB STM32H753XI board), making the systems trade-offs concrete and useful for practitioners.
4. **Commendable Disclosures**:
   The authors are open about several of their limitations, including the lack of physical on-board profiling (relying on cycle-accurate emulation), the reliance on a synthetic coordinate sandbox, and the use of unquantized streams for calibration.

---

## Major Weaknesses and Critical Concerns

### 1. Fundamental Contradiction: "Training-Free" Claim vs. Active Fine-Tuning (QAT)
The paper repeatedly advertises **SA-QAB** as a "**training-free, forward-only**" framework in the Abstract, Section 1, and Section 3. However, Section 4.3 (page 7) reveals a severe contradiction:
- Pure post-training quantized (PTQ) SA-QAB only achieves **50.00% joint accuracy**, which is a massive **34.90% accuracy drop** compared to the unquantized blending baseline (84.90%).
- To obtain the highlighted headline result of **77.50% joint accuracy** (marketed as a "+58.90% absolute improvement over post-merge quantization"), the authors must perform **Quantization-Aware Fine-Tuning (QAT)** for 5 epochs using Straight-Through Estimation (STE) over the adapters and classification heads.
- This is a major scientific and presentation flaw. A method cannot be marketed as "training-free" if its training-free variant is uncompetitive (50.00% accuracy) and requires active backpropagation and gradient-based fine-tuning to achieve its marketed performance. The authors attempt to frame this as a "highly practical compromise," but this fails to resolve the fundamental contradiction in the paper's core claims.

### 2. Methodological Mismatch in Quantization Scale Recovery (QSR)
The proposed QSR scaling factors $\beta_k^{(l)}$ (Equation 4) are pre-computed using intermediate features $h_s^{(l-1)}$ extracted from the **clean, full-precision (FP16)** network stream. At test-time, however, these factors are applied to activations coming from the **noisy, INT4-quantized base backbone**.
- The authors admit that calibrating over quantized features "corrupts the reference expectation with compounding noise, destabilizing the scale factors."
- This indicates that the proposed QSR calibration is mathematically fragile and highly sensitive to noise.
- Applying scaling factors computed on clean distributions to a noisy, quantized stream at test-time introduces a fundamental distribution mismatch, with no theoretical guarantees of stability or correctness.

### 3. Artificial Subspace Orthogonality in the Coordinate Sandbox
The primary quantitative evaluations (Table 2) are conducted in the synthetic **Coordinate Sandbox**. 
- In this environment, task-specific features are generated in **completely disjoint (orthogonal) coordinate subspaces** (e.g., Task 0 in channels [0:48], Task 1 in [48:96]).
- This design makes the dynamic routing task (Q-ZCA) trivial. Because the features are isolated in disjoint channels, Layer 3 features will naturally cluster into perfectly orthogonal vectors.
- Real-world multi-task learning involves features that share and overlap extensively in the same embedding channels. The reliance on an orthogonal coordinate-based simulation suite as the primary quantitative benchmark severely limits the scientific credibility and generalizability of the reported results.

### 4. Selective Omission of Real-Pixel and Baseline Metrics
In Section 4.3 (Expanded Real-World 4-Task Pixel Feasibility Study), the authors briefly present real-pixel evaluations but:
- **Completely omit comparative baseline results**: There is no table showing what PMQ, Uniform Merging, or Q-Merge achieve under the exact same real-pixel setup, making it impossible to evaluate if SA-QAB's benefits are as pronounced on real images.
- **Omit ResNet-18 Classification Accuracy**: For the convolutional backbone, the authors only report the "routing accuracy" (87.00% average routing specificity), but completely omit the final classification accuracy. This selective reporting raises concerns about the actual end-to-end performance.
- **Omit SOTA Model Merging Baselines**: The paper only compares against simple uniform averaging and Q-Merge, completely ignoring widely used, state-of-the-art weight-space model merging methods (such as TIES-Merging, DARES, ZipIt, or Task Arithmetic).

### 5. Massive Host CPU Latency Overhead
In Section 4.2, the authors disclose that on the host CPU in PyTorch, SA-QAB incurs a massive **139.9% latency overhead** compared to the Static 4-bit model (1.136 ms vs. 0.474 ms).
- For edge deployment scenarios running ONNX Runtime, PyTorch Mobile, or TensorFlow Lite with Python bindings, SA-QAB will be **more than 2x slower** than static post-merge quantization, completely undermining its efficiency claims.
- The claim that this overhead disappears on bare-metal CMSIS-NN is highly speculative, as no physical on-board validation was performed.

### 6. Conceptual Mislabeling: SA-QAB is Not Model Merging
Traditional model merging *fuses* multiple weights into a single joint weight tensor, maintaining a strictly **constant $O(1)$ flash storage cost** as the number of tasks scales. SA-QAB, by contrast, keeps all $K$ adapters separate in memory, resulting in a **linear $O(K)$ storage cost**.
- By keeping the adapters separate, SA-QAB completely avoids the fundamental scientific challenge of parameter-space weight interference.
- Thus, SA-QAB is a **multi-adapter mixture-of-experts (MoE) routing system**, not a model merging method. Comparing its SRAM/Flash scaling directly against weight-space merging (Table 3) is misleading because SA-QAB incurs an additional linear storage footprint that true model merging does not.

---

## Detailed Dimension Ratings

### Soundness: Fair
The mathematical formulation is clear, but the soundness of the paper is compromised by:
- The logical contradiction of advertising the framework as "training-free" while relying on QAT for competitive results.
- The distribution mismatch in the QSR calibration step.
- The artificial orthogonality of the primary evaluation environment.

### Presentation: Fair
The paper is well-structured and logical. However:
- The writing is heavily promotional and relies on sensationalist, unscientific language ("spectacular," "catastrophic collapse").
- The paper introduces redundant, convoluted jargon (DHQ, QSR, Q-ZCA) to describe standard adaptations of existing concepts.
- The paper conflates multi-adapter dynamic routing with model merging, misrepresenting the core trade-offs.

### Significance: Fair
- **TinyML / Edge Computing**: The systems-level details and optimizations are highly practical and could be useful for deploying multi-adapter models on microcontrollers under tight memory limits.
- **Broader ML / Model Merging**: The significance is low. Because SA-QAB does not actually merge weights, it does not advance our understanding or capabilities in parameter-space alignment or weight-space fusion.

### Originality: Fair
The framework represents an engineering assembly of pre-existing concepts rather than a fundamental scientific advancement:
- Dynamic routing via task centroids is directly adapted from SABLE and SPS-ZCA.
- Heterogeneous bit-widths are a standard concept from QLoRA and PTQ literature.
- QSR is a basic gain-correction calibration step.
- Diagonal GMMs are a classic, standard technique.

---

## Overall Recommendation

**Rating: 3 (Weak reject)**

**Justification**:
The paper addresses a highly practical and relevant systems-level problem in TinyML and provides extensive hardware-aligned optimizations (such as fixed-point cosine similarity and diagonal GMMs). However, the core merits of the paper are outweighed by significant scientific and methodological weaknesses:
1. A severe contradiction between the advertised "training-free" claims and the actual reliance on a 5-epoch QAT fine-tuning phase to achieve competitive results.
2. The reliance on a highly artificial, synthetic "Coordinate Sandbox" (with disjoint orthogonal coordinate subspaces) as the primary quantitative benchmark.
3. The lack of comparative baseline results or tables for the real-pixel feasibility study, and selective omission of classification metrics for ResNet-18.
4. The conceptual conflation of multi-adapter dynamic routing with weight-space model merging, which masks the linear storage scaling ($O(K)$ flash memory) bottleneck of the proposed approach.

If the authors can resolve the contradictions in their claims, move the real-pixel evaluations to the primary position with complete baseline tables, and evaluate against state-of-the-art model merging baselines, the paper would be a strong candidate for acceptance. In its current form, however, the weaknesses outweigh the merits.

---

## Constructive Feedback and Questions for the Authors

1. **Resolve the "Training-free" Contradiction**: Either remove the "training-free, forward-only" claims from the Abstract, Introduction, and Methodology, or report the actual pure PTQ results (50.00% or 70.10% with pre-scaling) as the primary headline results throughout the text.
2. **Move Real-Pixel Results to Table 2**: Replace the synthetic Coordinate Sandbox results in Table 2 with actual, real-pixel classification accuracy on MNIST, Fashion-MNIST, CIFAR-10, and SVHN using a standard ViT-Tiny backbone. Please include all baselines (PMQ, Uniform Merging, Q-Merge, SOTA weight merging like TIES-Merging and DARES, and unquantized ceilings) under the exact same real-pixel setup.
3. **Report End-to-End ResNet-18 Accuracy**: For the ResNet-18 evaluation, please report the final classification accuracy of the quantized pipeline in addition to the intermediate routing specificity.
4. **Clarify the Storage Scaling Bottleneck**: Explicitly discuss the linear flash storage scaling bottleneck ($M(K) = M_{\text{base}} + K \times M_{\text{adapter}}$) in the methodology and compare it directly against the constant storage footprint of static weight merging. Refrain from labeling the framework as "model merging" since it keeps all adapters separate in memory.
5. **Address the QSR Mismatch**: Please explain why scaling factors $\beta_k^{(l)}$ computed on clean FP16 streams remain mathematically valid when applied to noisy INT4-quantized activations at test-time. Have you tried calibrating over the quantized base features with a small learning-rate correction or a robust statistics method (e.g., median absolute deviation) to handle the noise?
6. **Provide Physical Microcontroller Profiling**: Compile your cmsis-nn kernel implementation and deploy it to a physical STM32H753XI board to measure actual silicon latency, SRAM, and power consumption, verifying if the emulated 3.7% latency overhead holds on real hardware.
7. **Adopt a More Objective, Scientific Tone**: Please revise the manuscript to remove sensationalist adjectives ("spectacular," "catastrophic," "collapse completely") and focus on objective, rigorous scientific descriptions of the empirical observations.
