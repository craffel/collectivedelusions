# Summary of the Paper

## Basic Information
* **Title:** QA-Merge: Quantization-Robust Centroid Routing for Low-Precision Edge Serving
* **Author:** Elena Rostova
* **Affiliation:** Department of Computer Science, ETH Zürich, Switzerland
* **Email:** elena.rostova@inf.ethz.ch

---

## 1. Core Problem Statement
Dynamic model ensembling and weight-merging in latent coordinate spaces have emerged as lightweight, training-free mechanisms for continuous task adaptation during inference. However, deploying these architectures on resource-constrained edge hardware requires low-precision quantization—specifically **8-bit integer (INT8) activations** and **4-bit integer (INT4) ensembling weights**—to meet strict latency, power, and memory bandwidth budgets.

When standard ensembling methods (such as SABLE, ChemMerge, or Momentum-Merge) are subjected to these low-precision constraints, they experience **Quantization Collapse**. The rounding operators act as aggressive step filters that project coordinates onto a coarse, discrete grid. This destroys subtle representation boundaries, causing task centroids to overlap, routing weights to freeze, and gradients to vanish. Consequently, the performance of dynamic ensembling collapses to that of simple, static Uniform Merging, neutralizing all benefits of dynamic adaptation.

---

## 2. Key Proposed Methodology (QA-Merge)
To overcome the deployment bottleneck of representation collapse, the paper proposes **QA-Merge** (Quantization-Aware Merge), which introduces four hardware-compatible, computationally light techniques:

1. **Quantized Centroid Calibration (QCC):** Computes task-specific centroids offline in a quantized space and uses scale-invariant cosine similarity in the integer coordinate space. This prevents centroid overlap, maximizes directional task separation, and avoids range mismatches.
2. **Straight-Through Estimator (STE) Gating Optimization:** Employs the STE during few-shot routing optimization to bypass the non-differentiable rounding operators, enabling gradient-based training through discrete rounding boundaries.
3. **Error-Feedback Trajectory Stabilization (EF-Smooth):** Tracks blending coefficient rounding errors layer-by-layer and diffuses them downstream to stabilize trajectories. It utilizes **Permutation-Invariant Single-Pass Apportionment (PI-SPA)** to map continuous weights onto strict 4-bit integer grid values summing to 1.0 (15 discrete levels) without the $O(K \log K)$ sorting bottleneck of Hamilton's method. PI-SPA uses static unique expert IDs to break ties and an $O(K)$ selection threshold ($\theta$) to select the top $S$ elements, strictly preserving **permutation invariance** and **remainder-magnitude sensitivity** in a branchless, hardware-friendly scan.
4. **Activation Error Feedback (AEF):** Overcomes the *Small-Step Quantization Bottleneck*, where tiny representational updates (the adapter "pull" vectors) are smaller than the quantization step size and round to zero. AEF tracks sub-grid quantization errors layer-by-layer and accumulates them residually, adding them back to the next layer's updates.

---

## 3. Core Empirical Results and Findings
The authors evaluate QA-Merge inside the 14-layer Coordinate Sandbox (ICS) across small-sample ($N_{\text{cal}} = 64$) and large-sample ($N_{\text{cal}} = 4000$) regimes:

* **Simulated Performance Recovery:** Under naive INT8/INT4 quantization, standard baselines collapse to static Uniform Merging (around **65.8%** joint accuracy at $\rho=0.0$). QA-Merge successfully recovers almost **100% of the continuous Float32 ensembling ceilings** in this simulation.
* **Resilience to Weak Distractors:** Tested with a simulated weak SVHN expert (calibrated to **22.80%**). QA-Merge successfully bypasses this distractor, keeping the routing weight allocated to it negligible ($\le 0.02$) for other queries.
* **SmoothQuant $\alpha$-Sweep:** Validates the proposal for Dynamic Outlier-Aware Activation Scaling under heavy-tailed outlier conditions. Sweeping $\alpha \in [0.0, 1.0]$ demonstrates that the optimal balance occurs at $\alpha \in [0.1, 0.3]$, achieving a peak Gating Decision Match Rate of **97.80%** and minimizing Logit MSE.
* **Physical Hardware Latency and Power Benchmarking:** Compiled utilizing CMSIS-DSP integer coordinate propagation kernels on a physical ARM Cortex-M7 microcontroller (STM32H753XI running at 480 MHz). The quantized integer ensembling loop runs in exactly **0.18 ms** per forward pass compared to **0.95 ms** for the FP32 FPU loop, yielding a **5.2x latency speedup** and reducing power consumption by **42%** (to 18 mW).
* **AEF SRAM Scaling Analysis:** Demonstrates that the AEF state memory footprint scales extremely well, requiring only **8 KB** of SRAM per layer at standard LLM scale ($D=4096$, batch size 1).
* **Trajectory Divergence Analysis:** Empirically demonstrates that the quantized representation trajectory $\tilde{h}^{(l)}$ remains extremely close to the true continuous Float32 path (mean $\ell_2$ distance of **0.0413** at Layer 14), verifying that the feedback-driven trajectory divergence is benign.
* **Generalizability:** The paper validates generalizability via a standalone PyTorch implementation (`toy_qamerge_lora.py`) of a multi-expert dynamic LoRA-mixture layer operating under low precision.

---

## 4. Overall Strengths and Limitations

### Key Strengths:
* **Addressing Physical Constraints:** Directly targets the physical realities of low-precision edge chips (memory bandwidth, register pressure, SRAM footprint, FPU availability).
* **Elegant Mathematical Design:** Provides formal proofs of error bounds for both EF-Smooth and AEF (using a telescoping property showing bounded representational error).
* **Strict Permutation Invariance:** The proposed PI-SPA apportionment method guarantees permutation invariance and remainder-magnitude sensitivity while maintaining a branchless $O(K)$ threshold complexity.
* **Physical Validation:** Grounded with a real-world physical microcontroller benchmark (STM32H7) and a PyTorch LoRA simulation script, establishing high credibility.

### Key Limitations:
* **Coordinate Sandbox Setup:** The primary performance recovery curves are evaluated inside the Coordinate Sandbox with Gaussian-generated task data rather than on actual massive deep neural networks (e.g., LLaMA or ViT) and real image/text datasets. However, this is heavily mitigated by the control and representation-space isolation of the sandbox, and the mathematical proofs of generalization.
