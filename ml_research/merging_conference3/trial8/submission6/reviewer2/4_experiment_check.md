# Evaluation Phase 4: Critical Experimental Evaluation

## Experimental Setup and Dataset Validity
The primary weakness of the paper's experimental validation lies in its **reliance on a synthetic sandbox environment**:
1. **The Isolating Coordinate Sandbox (ICS):** All physical accuracy, OOD AUROC, and latency scaling evaluations are conducted within a custom synthetic sandbox. While the authors describe this sandbox as "high-fidelity," it models a highly simplified frozen linear projection mapping from $D_{\text{in}} = 64$ to $D = 192$, with $K = 3$ task-specific adapters. 
2. **Artificial Task-Specific Activations:** The task-specific activations are simulated using simple coordinate-based shifts:
   $$h_b = h_{b,\text{base}} \odot s_k + t_k$$
   In real-world deep neural networks, activations are governed by complex, highly non-linear, high-dimensional distributions with semantic and contextual dependencies. The coordinate shifts in the sandbox are highly structured and likely make subspace separation artificially easy, which explains the extremely high performance reported (85.81% accuracy and 0.9755 OOD AUROC).
3. **Absence of Real-World Scale-Up:** There are **zero experiments** conducted on standard large-scale benchmarks (e.g., GLUE or SuperGLUE for NLP, ImageNet-1K or VTAB for vision) using actual pre-trained Transformers (e.g., Llama-3-8B, ViT-B, or BERT). This represents a significant empirical-to-theoretical gap, leaving the practical viability of LSPR on complex real-world workloads unproven.

---

## Baselines and Comparative Rigor
The paper compares LSPR against an appropriate set of baselines:
1. **Uniform Merging:** Static averaging of LoRA weights, which serves as a baseline representing capacity dilution.
2. **Linear Router and QWS-Merge:** Classic parametric routers that optimize ensembling coefficients. The authors show that these routers suffer from "heterogeneity collapse" in mixed batches (collapsing to Uniform Merging's 23.96% accuracy), as they average coefficients over the entire batch. This is a highly valid systems critique, and the comparison rigorously demonstrates LSPR's immunity to this collapse.
3. **PFSR and SABLE:** Classification-head-based SOTA training-free routers. The authors compare LSPR's early-layer routing against these models, demonstrating LSPR's structural advantages (avoiding the Early-Layer Routing Paradox and operating head-free).
4. **SPS-ZCA:** The current training-free SOTA.
   - **Critical Observation:** SPS-ZCA and LSPR achieve **identical Joint Mean Accuracy (85.81%)** in the homogeneous and heterogeneous streams in the sandbox. This means that LSPR does not actually outperform the current SOTA in terms of classification accuracy in the sandbox.
   - **The Delta:** LSPR's advantage over SPS-ZCA is not accuracy, but rather data-efficiency and systems simplicity: it requires **zero parameters and zero task-specific calibration data**, whereas SPS-ZCA requires high-dimensional centroids pre-computed from a 64-sample calibration split per task, Unit-Norm Calibration, Dispersion Calibration, and EM-fitted GMMs. This trade-off is fair and well-argued.

---

## Physical Latency Benchmarking
The wall-clock CPU execution latency benchmarking provides valuable systems insights:
1. **Flat Latency Scaling:** The physical CPU latency profiles verify that LSPR scales flat with batch size $B$, outperforming Micro-Batch Homogenization (PFSR+MBH) which scales linearly due to partitioning mixed batches and sequentially reloading weights.
2. **The Crossover Point ($K_{\text{crossover}} \approx 20$):** In a commendable act of scientific honesty, the authors identify a clear crossover point at $K \approx 20$, where executing all $K$ experts in parallel becomes compute-bound ($\mathcal{O}(B \cdot K \cdot r \cdot D)$) and slower than sequential serving.
3. **Sparse-LSPR Validation:** The authors physically evaluate Sparse-LSPR (Top-2 gating) on a heterogeneous stream. It achieves 85.81% accuracy while decoupling latency from registry size $K$, verifying that the proposed scaling strategy is physically viable.
4. **Hardware Scope Limitation:** The latency profiling is restricted to host CPUs. While the authors argue that CPUs represent resource-constrained edge systems (where sequential DRAM weight reloads bottle execution), most production multi-tenant serving occurs on GPUs where execution layouts are highly multi-threaded and PyTorch scheduler loop overheads are often compiled away. It is unclear if the crossover point and the speedups of LSPR would remain identical on highly parallel GPU architectures.

---

## Empirical Support of Mathematical Claims (Ablation Rigor)
The paper excels in its ablation studies, which physically validate almost every mathematical claim:
1. **The Failure of Standard LoRA:** The authors train standard LoRA adapters (cross-entropy alone) and show that the subspace alignment is virtually random (0.0975), dropping LSPR accuracy to 19.79%. This empirically supports their theoretical claim that weight-activation alignment is not automatic and requires a co-designed reconstruction loss or post-hoc warm alignment.
2. **Verification of Post-Hoc Warm Alignment:** The authors implement the warm alignment phase (tuning $A_k$ for 60 steps on 64 queries) and show that it successfully rotates $A_k$ into alignment, increasing the alignment score from 0.0975 to 0.4076 and recovering accuracy to 66.02% without degrading the learned features. This is a crucial validation of their public-adapter compatibility claim.
3. **Split-Rank LoRA Validation:** The physical evaluation of Split-Rank LoRA demonstrates that it achieves a highly competitive 84.11% accuracy while preserving high-fidelity subspace alignment (0.5447) on its dedicated routing columns. This validates that the optimization-capacity trade-off can be successfully decoupled in practice.
4. **Validation of Layer-Wise Freezing:** In a multi-layer setup, Layer-Wise Freezing recovers 100% of the Expert Ceiling (74.09%), outperforming Layer-Wise Recomputation (51.43%) by a wide margin. This physically supports the authors' claim that computing coefficients on unaligned downstream layers is mathematically noisy and destructive.
