# Summary: Scale-Aligned Quantized Activation Blending (SA-QAB)

## 1. Main Topic
The paper addresses the challenge of deploying multi-task merged models onto resource-constrained edge platforms (such as microcontrollers and wearables) that require low-bit integer quantization (INT4/INT8). Standard model merging methods compress or average weights in parameter space (Post-Merge Quantization or PMQ), which suffers from severe representation collapse under low-bit quantization and non-linear activations. Activation-space ensembling, on the other hand, scales computationally and memory-wise as $O(K)$ for $K$ experts, which is too expensive for edge hardware.

## 2. Proposed Approach
The authors propose **Scale-Aligned Quantized Activation Blending (SA-QAB)**, a framework designed to preserve multi-task modularity and performance under low-bit quantization. The key components of SA-QAB are:
- **Decoupled Heterogeneous Quantization (DHQ):** Squeezes the heavy, shared base backbone to per-channel 4-bit integer (INT4) representation while keeping the lightweight, task-specific LoRA adapters/experts in 8-bit integer (INT8) representation.
- **Quantized Zero-Shot Centroid Alignment (Q-ZCA):** An early-stage dynamic routing layer at Layer 3 that computes the cosine similarity between quantized input activations and pre-computed task centroids entirely on the integer manifold. It utilizes a low temperature to achieve near-sparse routing (executing exactly one active expert).
- **Quantization Scale Recovery (QSR):** Pre-computes scale recovery factors over a small offline calibration set (64 samples per task) to correct for scale contraction in low-bit adapters on-the-fly without requiring backpropagation.
- **Out-of-Distribution (OOD) GMM Rejection Gate:** A low-power diagonal-covariance Gaussian Mixture Model (GMM) trained on Layer 3 features to filter OOD queries, bypassing the adapters and falling back to the robust base backbone.

## 3. Key Findings
- **Catastrophic Collapse of Static Merging:** Static, weight-space model merging (PMQ 4-bit and Q-Merge 4-bit) collapses to near-random performance (~18.60% and ~22.20% joint accuracy respectively) when evaluated under non-linear activations like GELU.
- **Activation-Space Resilience:** SA-QAB avoids weight-space interference by executing base and expert paths in parallel and dynamically blending activations. It recovers joint accuracy to **77.50%** (when using a 5-epoch Quantization-Aware Fine-Tuning phase for adapters) or **50.00%** (using pure post-training quantization), showing a major improvement over static PMQ.
- **Hardware Feasibility:** Physical profiling emulation on an STM32H7 microcontroller demonstrates that SA-QAB fits comfortably within 360.8 KB of active SRAM (with $>60\%$ headroom on a 1MB device) and achieves a **2.3x speedup** and **57% energy savings** compared to full-precision ensembling. It adds only a minor 3.7% latency overhead over the collapsed static 4-bit baseline.

## 4. Explicitly Claimed Contributions and Evidence
1. **Decoupled Heterogeneous Quantization (DHQ):** Combining INT4 base backbone with INT8 adapters. *Evidence:* Table 2 (reducing active SRAM footprint from 1224.8 KB to 360.8 KB) and Section 3.1.
2. **Quantized Zero-Shot Centroid Alignment (Q-ZCA):** Operating routing entirely on the integer manifold. *Evidence:* Table 3 (retaining routing accuracy of ~80.40% under INT4 backbone noise) and Appendix B.3.
3. **Quantization Scale Recovery (QSR):** Restoring scale contraction in low-bit adapters. *Evidence:* Section 3.3 and Appendix B.6 (+0.70% absolute accuracy gain under ultra-low INT4/INT4 setups).
4. **OOD GMM Rejection Gate:** Low-overhead filtering. *Evidence:* Section 3.5, Table 2, and Appendix B.1 (achieving 98.0% OOD TPR and 2.4% FRR at optimal threshold).
5. **Experimental Verification:** Evaluation in a 192D synthetic Coordinate Sandbox and a real-pixel ViT-Tiny across MNIST, Fashion-MNIST, CIFAR-10, and SVHN. *Evidence:* Section 4.1, Table 3.
