# 4. Experiment Check: LoRA Subspace Projection Routing (LSPR)

## 4.1 Evaluation Methodology and Experimental Design
The experimental section of the paper is exceptionally thorough, combining rigorous physical evaluations inside a fully-trained PyTorch multi-task environment (the **Isolating Coordinate Sandbox (ICS)**) with precise hardware-level CPU latency benchmarks.

### A. The Isolating Coordinate Sandbox (ICS)
- **Design:** The ICS features a frozen backbone mapping from $D_{\text{in}} = 64$ to a hidden dimension of $D = 192$, and $K=3$ task-specific adapted networks fine-tuned via LoRA (rank $r=8$) using standard backpropagation under PyTorch. A fourth task serves as the out-of-distribution (OOD) testbed.
- **Strengths:** By evaluating on physically trained PyTorch weights and emergent activations, the experiments avoid the circularity of custom data generators, ensuring scientific rigor. The tasks feature continuous representational overlap, meaning there are overlapping features and leakage, which simulates realistic, challenging deployment scenarios.
- **Limitations (Critical Analysis):** As a simplified sandbox, it does not capture the multi-layer, multi-head attention dynamics, layer normalization, residual connections, or token-level routing of full-scale modern Transformers (e.g., Llama-3-8B, ViT-H). The absolute accuracy numbers (85.81%) and OOD AUROC (0.9755) are specific to this synthetic proof-of-concept and would likely degrade on large-scale real-world datasets (e.g., GLUE, ImageNet-1K). The authors transparently acknowledge this limitation and use high-dimensional random projection theory to analyze how the geometric properties would scale to massive models (Section 4.1).

### B. Stream Configurations and Baselines
- **Streams:** The paper evaluates under two realistic serving scenarios: a **Homogeneous Stream** and a highly mixed **Heterogeneous Stream** (simulating multi-tenant concurrent requests with a batch size of $B=256$).
- **Baselines:** The paper compares LSPR against a highly comprehensive set of baselines representing the entire spectrum of dynamic merging:
  1. *Expert Ceiling:* Absolute upper bound of perfect routing.
  2. *Uniform Merging:* Static weight average.
  3. *Linear Router (Reg) & QWS-Merge:* Trainable parametric dynamic routers.
  4. *PFSR + MBH SOTA:* Head-centroid projection with sequential micro-batch partitioning.
  5. *SPS-ZCA SOTA:* Multi-stage calibration with EM-fitted coordinate GMMs.

---

## 4.2 Main Empirical Findings and Validation

### A. Classification Performance and Immunity to Collapse
- **Findings:** Under both streams, LSPR achieves a **Joint Mean Accuracy of 85.81%**, matching the Expert Ceiling and coming within 0.13% of the highly complex SPS-ZCA SOTA (85.94%) while requiring zero trainable parameters and zero calibration datasets.
- **Heterogeneity Collapse:** Classic parametric routers (Linear, QWS-Merge) achieve 23.96% accuracy under mixed streams, collapsing to the Uniform baseline. This is because they average merging coefficients over the batch to construct a single merged weight vector, which averages to $[1/3, 1/3, 1/3]$ under heterogeneous batches. LSPR is immune to this because it performs sample-specific blending on-the-fly inside a single parallel pass, maintaining its 85.81% accuracy across all batch sizes.

### B. Out-of-Distribution (OOD) Rejection
- **Findings:** LSPR's zero-shot scale-invariant projection energy achieves an outstanding Area Under the ROC curve (AUROC) of **0.9755** under domain shifts.
- **Significance:** This far surpasses SABLE and matches or exceeds SPS-ZCA, which relies on fitting multi-dimensional GMM density models via EM on offline calibration splits. This physically demonstrates that the closed-form QR projection serves as a highly precise density estimator without any parametric fitting.

### C. Systems Serving Latency
- **Findings:** Profiled on a host CPU using high-precision timers, under mixed streams ($B=256$):
  - *PFSR + MBH SOTA* (Sequential Serving) takes **139.02 ms** due to micro-batch partitioning and sequential forward passes.
  - *LSPR* (Ours) takes only **49.46 ms** (a **2.81$\times$ physical speedup**).
- **GPU Deployment Complementarity:** The authors acknowledge that while they benchmark on a CPU (representing edge serving where custom CUDA kernels are unsupported), LSPR's mathematical routing is highly complementary to production GPU serving frameworks like S-LoRA and Punica. They map out a clear 2-step systems integration showing how LSPR can be layered as a high-speed routing step on top of batch GEMM kernels (like `bgmv` or `sgmv`) for massive scale-up servers.

---

## 4.3 Detailed Ablation Analysis
The experiments include extensive and highly informative ablation studies that empirically validate every claim in the paper:
1. **Routing Temperature ($\tau$):** Confirms that a sharp temperature ($\tau \le 0.01$) is critical to enforce precise routing and prevent activation dilution.
2. **Necessity of Joint Loss:** Shows that training standard LoRA without reconstruction loss results in random down-projection matrices, collapsing LSPR accuracy to 19.79%.
3. **Proof of Warm Alignment:** Shows that performing warm-alignment on unaligned LoRA adapters for just 60 steps rotates the column space into alignment (boosting on-task alignment by 4.1$\times$ to 0.4076) and restores ensembling accuracy to 66.02% with 0% downstream degradation.
4. **Proof of Sparse-LSPR Gating:** Evaluates Sparse-LSPR Top-2 gating, showing it achieves exactly 85.81% Joint Mean Accuracy (fully recovering full LSPR ensembling accuracy) while decoupling execution latency from registry size $K$ (yielding a flat serving latency curve).
5. **Downstream Capacity Trade-offs & Split-Rank LoRA:** Validates that adding reconstruction loss ($\lambda = 1.5$) does not degrade downstream expert accuracy, with individual accuracies remaining at 85.81%. It also validates a **split-rank strategy** (Split-Rank LoRA) to decouple task capacity from autoencoding under extreme rank constraints, recovering 84.11% accuracy with robust routing alignment ($0.5447$).
6. **Layer-Wise Freezing:** Empirically proves that layer-wise freezing (74.09% Joint Mean Accuracy) perfectly recovers the Expert Ceiling and beats layer-wise recomputation (51.43%) by 22.66%, as downstream layer weights are unaligned and recomputing coefficients on them introduces noise.

## 4.4 Experimental Summary
The empirical validation of LSPR is of very high quality. While the scale of the evaluation is limited by the synthetic nature of the sandbox (which is transparently acknowledged), the inclusion of deep, physically trained PyTorch ablations (proving Warm Alignment, Split-Rank LoRA, Sparse-LSPR, and Layer-Wise Freezing) provides an exceptional level of scientific rigor and empirical completeness.
