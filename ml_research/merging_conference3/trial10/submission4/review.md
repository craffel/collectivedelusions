# Peer Review: QA-Merge: Quantization-Robust Centroid Routing for Low-Precision Edge Serving

## Overall Recommendation
* **Rating:** `3: Weak Reject` (A paper with clear merits, but also some weaknesses, which overall outweigh the merits. Papers in this category require revisions before they can be meaningfully built upon by others.)
* **Soundness:** Fair
* **Presentation:** Good
* **Significance:** Fair
* **Originality:** Good

---

## 1. Summary of the Paper
The paper addresses the deployment of dynamic coordinate-space model ensembling architectures under low-precision constraints (8-bit integer [INT8] activations and 4-bit integer [INT4] ensembling weights) for resource-constrained edge hardware. 

The author identifies **Quantization Collapse** as a critical deployment bottleneck: standard ensembling methods (like SABLE, ChemMerge, and Momentum-Merge) rely on high-precision coordinates, and extreme quantization rounding noise collapses representation boundaries, leading to overlapping task centroids, vanishing gradients, and frozen routing trajectories that drop to static uniform merging.

To resolve this, the paper proposes **QA-Merge** (Quantization-Aware Merge), which introduces four techniques:
1. **Quantized Centroid Calibration (QCC):** Computes task-specific centroids offline in high-precision and quantizes the final averaged centroid in target integer spaces, utilizing scale-invariant cosine similarity gating.
2. **Straight-Through Estimator (STE) Gating:** Employs STE during few-shot routing optimization to bypass non-differentiable rounding operators.
3. **Error-Feedback Trajectory Stabilization (EF-Smooth):** Tracks rounding errors of ensembling coefficients and diffuses them downstream, using a **Discrete Simplex Projection** (adapted from Hamilton's method of apportionment) to map weights onto strict 4-bit discrete grids.
4. **Activation Error Feedback (AEF):** Tracks and accumulates sub-grid representation rounding errors layer-by-layer to overcome the "Small-Step Quantization Bottleneck."

The framework is evaluated within a 14-layer Coordinate Sandbox (ICS) simulation, and its ensembling kernels are micro-benchmarked on an ARM Cortex-M7 (STM32H753XI) microcontroller.

---

## 2. Key Strengths of the Paper

* **Elegant Mathematical Design:** The paper features solid mathematical proofs for its proposed techniques. It models EF-Smooth as a first-order FIR noise-shaping filter, proving bounded cumulative blending error independent of network depth. It also formalizes Theorem 3.1, proving a telescoping, bounded representational error for AEF.
* **Algorithmic Creativity:** The adaptation of **Hamilton's method of apportionment** (traditionally used in political census allocations) to map ensembling weights deterministically and non-operatively onto a strict discrete 4-bit simplex is highly creative and original.
* **Excellent Structural Writing:** The manuscript is exceptionally well-written, clear, professional, and easy to follow. It has a highly comprehensive Appendix containing detailed cycle-count and register calculations.
* **Scientific Integrity:** The author is highly honest and transparent in **Remark 3.2**, detailing the distinction between numerical error bounding and trajectory divergence, and empirically tracking this divergence in Section 4.

---

## 3. Weaknesses of the Paper (Critical Flaws)

Despite the mathematical elegance and clear writing, the paper suffers from three critical flaws that severely limit its real-world utility and practical significance:

### Critical Flaw 1: Exclusively Synthetic and Simulated Evaluation (No Real-World Models or Datasets)
The paper claims that QA-Merge successfully recovers full-precision ensembling gains under INT8/INT4 constraints with zero downstream accuracy loss and minimal computational overhead. However, **this claim is validated exclusively inside a synthetic simulation environment (the Analytical Coordinate Sandbox) with synthetic Gaussian data**. 
* There are **no actual deep neural network models** (such as standard CNNs, Vision Transformers, or Large Language Models) evaluated in this paper.
* There are **no actual real-world datasets** evaluated. The "MNIST, Fashion-MNIST, CIFAR-10, SVHN" tasks are not real images; they are generated as synthetic 192-dimensional Gaussian vectors centered around 4 hardcoded task coordinates.
* Consequently, the "accuracies" reported in Table 1 and Table 2 (e.g., CIFAR-10: 92.40% or SVHN: 22.80%) are simulated by adjusting noise levels ($\sigma_k$) and coordinate biases in the sandbox, not by running an image classifier. Real-world deep learning manifolds are highly non-linear, high-dimensional, and far more complex than a toy sandbox. The utility of QA-Merge on actual fine-tuned neural networks remains completely unproven.

### Critical Flaw 2: Misleading Physical Hardware Latency Claims
The paper reports a massive **"5.2x latency speedup"** on an STM32H753XI microcontroller compiled with CMSIS-DSP.
* However, this hardware benchmark is conducted **only on the "compiled integer coordinate propagation kernels"** (the ensembling vector loop consisting of a few lines of vector MAC operations for $D=192$).
* In any actual edge application, the dynamic ensembling/routing loop constitutes a microscopic fraction of the overall inference footprint. The backbone neural network layers (such as standard convolution or attention blocks) consume the vast majority of memory bandwidth and computational cycles.
* Accelerating a tiny vector ensembling loop by 5.2x while ignoring the heavy backbone layers results in a **completely negligible end-to-end latency reduction** (as dictated by Amdahl's Law). Framing this as a practical edge serving solution without reporting end-to-end inference latency of a complete, real model is highly misleading to systems practitioners.

### Critical Flaw 3: Microarchitectural Bottlenecks on Resource-Constrained Hardware
The paper claims the proposed techniques are highly compatible with edge-level execution, but ignores several low-level physical bottlenecks:
* **The Vector Sorting Bottleneck:** The Discrete Simplex Projection (Algorithm 2) requires sorting the fractional remainders of the tasks (`Step 4`) to allocate the remaining shortfall. On specialized edge hardware, modern vector SIMD units, and NPUs, dynamic vector sorting is highly inefficient. NPUs lack parallel sort instructions, and executing sorting in software introduces branch prediction failures and pipeline stalls.
* **Stateful Accumulation Overhead:** Activation Error Feedback (AEF) requires tracking and storing an error-feedback buffer per sample across layers. While 384 bytes is negligible for $D=192$ in a 1-D simulation, scaling this to real-world models (e.g., $D=4096$ in Transformers, large batch sizes, or multi-task streaming) will require megabytes of high-speed SRAM, causing severe memory footprint and register pressure bottlenecks on low-cost edge chips.

---

## 4. Constructive Suggestions for Improvement

To transition this work from an elegant academic prototype into a credible, publication-ready systems paper, the authors must address the following suggestions in a major revision:

1. **Incorporate Real-World Evaluations:** Validate QA-Merge on actual fine-tuned neural network weights (e.g., merging fine-tuned vision adapters like LoRAs for ResNet/ViT, or text adapters for RoBERTa/LLaMA) on real-world datasets (e.g., GLUE, ImageNet, or multi-task classification).
2. **Provide End-to-End Latency Benchmarks:** Measure and report the end-to-end execution times on physical edge hardware (STM32 or mobile processors) for the *entire model* (backbone layers + routing ensembling). This will clarify if the 5.2x ensembling kernel speedup translates to any measurable physical latency savings in a production pipeline.
3. **Introduce Sorting-Free Simplex Apportionment:** Replace the dynamic sorting step in Hamilton's method of apportionment with a parallelizable, branchless approximation (e.g., a randomized rounding or fixed-threshold approach) and quantitatively analyze if this sorting-free variant degrades ensembling performance.
4. **Analyze AEF Register and Memory Pressure at Scale:** Include a scaling study detailing the SRAM footprint and memory bandwidth of the AEF feedback state as the model dimension $D$ scales from 192 to standard real-world sizes (e.g., 768, 1024, or 4096), particularly under multi-token batched generation.
