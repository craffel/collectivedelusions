# 1. Summary of the Paper

## Main Topic and Problem Statement
The paper addresses the challenge of deploying dynamic activation-space model ensembling systems on resource-constrained edge hardware. Specifically, while dynamic weight-merging and ensembling in latent coordinate spaces (such as those evaluated in the Interactive Coordinate Sandbox (ICS) environment, e.g., SABLE, ChemMerge, Momentum-Merge) achieve remarkable continuous-task adaptation in high-precision (Float32) floating-point environments, they experience catastrophic failure—termed "Quantization Collapse"—when compiled and deployed under extreme low-precision limits, such as 8-bit integer (INT8) activations and 4-bit integer (INT4) ensembling weights. 

The authors identify that standard uniform quantization rounding acts as an aggressive, non-differentiable step filter over intermediate activation coordinates. This rounding noise destroys representation boundaries, leading to overlapping task centroids, vanishing gradients, and highly biased routing trajectories, which degrades dynamic ensembling performance down to that of simple static Uniform Merging.

## Proposed Approach: QA-Merge
To bridge this gap and stabilize representation trajectories in low-precision spaces, the authors propose **QA-Merge (Quantization-Aware Merge)**, a framework comprising four lightweight, mutually reinforcing techniques:
1. **Quantized Centroid Calibration (QCC):** Calibrates task-specific centroids directly in the target quantized integer representation space using an offline few-shot calibration dataset. It uses scale-invariant cosine similarity in the integer coordinate space to prevent quantization range mismatches.
2. **Straight-Through Estimator (STE) Gating Optimization:** Optimizes parametric routing weights and biases during offline optimization by using the Straight-Through Estimator (STE) to bypass the non-differentiable rounding operator in the backward pass, allowing gradients to flow back to the gating heads.
3. **Error-Feedback Trajectory Stabilization (EF-Smooth):** Recursively tracks the rounding errors of ensembling coefficients (which are projected onto a discrete 4-bit simplex grid using a Permutation-Invariant Single-Pass Apportionment, PI-SPA) and injects them back into the routing weights of the next layer as a high-pass feedback correction.
4. **Activation Error Feedback (AEF):** Tracks sub-grid activation quantization errors layer-by-layer and adds them back residually to the next layer's update, overcoming the "Small-Step Quantization Bottleneck" where tiny representation updates are smaller than the quantization step size and round to zero.

## Key Findings
- **Naive Quantization Causes Collapse:** Standard ensembling baselines (SABLE, ChemMerge, Momentum-Merge) drop directly to static Uniform Merging levels (65.80% joint accuracy) under naive INT8/INT4 quantization.
- **QA-Merge Fully Recovers Performance:** Across all tested environments and entanglement levels ($\rho \in [0.0, 0.5]$), QA-Merge successfully recovers full-precision ensembling gains, tracking the Float32 performance ceilings within 0.1%–0.3% absolute accuracy.
- **Robustness and System Efficiency:** The method is highly sample-efficient and keeps trajectory jitter low. On physical hardware (STM32H753XI), it achieves a 5.2x latency speedup and a 42% power reduction compared to Float32 execution, running in 0.18 ms with 18 mW consumption.

## Explicitly Claimed Contributions (with Evidence)
1. **Identification of Quantization Collapse:** Deconstructed the catastrophic failure mode of dynamic ensembling under naive low-precision rounding (backed by empirical curves in Figure 1 and baseline results in Tables 1 & 2).
2. **Formalization of QA-Merge:** Introduced QCC, STE Gating, EF-Smooth, and AEF to stabilize low-precision representations (backed by mathematical formulations in Section 3 and proofs in Appendix A).
3. **Telescoping Bounded Representational Error of AEF:** Proved mathematically that AEF acts as a closed-loop error integrator that bounds the cumulative activation quantization error relative to the local quantized-state trajectory (Theorem 3.1, proven in Appendix A.2).
4. **Comprehensive Evaluation in ICS:** Verified complete recovery of Float32 performance on standard visual task signatures across varying entanglement levels and sample calibration sizes (Tables 1 & 2).
5. **Physical Hardware Deployment Blueprint:** Benchmarked and profiled the integer pipeline on ARM Cortex-M7 and NVIDIA Jetson Nano edge platforms (Appendix B.1 and B.2).
