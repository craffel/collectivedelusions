# 1. Summary of the Paper

## Main Topic and Approach
The paper addresses a core bottleneck in systems-ML for resource-constrained edge devices: concurrently serving multiple task-specific expert adapters (e.g., LoRA) from a shared pre-trained backbone. 
- **The Problem:** 
  - Static weight merging (e.g., Task Arithmetic, TIES-Merging) averages weights before deployment, which is computationally fast but suffers from **"heterogeneity collapse"** when executing mixed-task inputs simultaneously, destroying task specialization.
  - State-of-the-art dynamic routing frameworks like PFSR solve this using Micro-Batch Homogenization (MBH) to split mixed streams on-the-fly, but this introduces up to $K$ sequential backbone passes. On sequential edge CPUs, this linear $O(K)$ latency scaling is a fatal flaw.
  - Traditional dynamic routers also suffer from a **"routing paradox"** because they require late-stage penultimate features to route, forcing the system to run the backbone twice.
- **The Solution (SPS-ZCA):**
  - The authors propose **SPS-ZCA** (Single-Pass Sample-Wise Routing with Zero-Shot Centroid Alignment), a completely training-free and parameter-free framework.
  - **Single-Pass Activation-Space Dynamic Blending (SPS):** Instead of splitting the batch and running the base model sequentially, SPS executes the backbone once and dynamically blends the expert adapter activations on-the-fly sample-wise. This preserves a constant $O(1)$ backbone latency.
  - **Zero-Shot Centroid Alignment (ZCA):** Bypasses noisy classification heads and late penultimate features. Instead, it pre-computes task centroids using a tiny, 64-sample calibration split in the shared early representation space (Layer 3) of the pre-trained backbone. ZCA then routes inputs based on cosine similarity to these centroids.
  - **Resolving the Paradox:** To avoid mismatch, the authors freeze and share the first 3 layers (no LoRAs are trained or placed there). During inference, these early layers run task-agnostically, representations are extracted at Layer 3 to compute routing weights, and the remaining layers (4 to $L$) execute with parallel activation-space blending.
  - **UNC (Unit-Norm Calibration):** Resolves cross-expert representation norm imbalances (implemented via cosine similarity).
  - **IDC (Intra-Task Dispersion Calibration):** Normalizes differences in task-manifold spatial spreads by dividing raw coordinates by the expected in-distribution similarity scale.
  - **OOD Rejection via Coordinate GMM:** Fits a diagonal GMM on the 4D routing coordinates to reject out-of-distribution queries with a modality-specific fallback flow.

## Key Findings
- **High Accuracy:** In both the Isolating Coordinate Sandbox (ICS) simulation and end-to-end physical PyTorch/GPT-2 testing, SPS-ZCA recovers **100.0% of the Expert Ceiling** (Joint Mean of 79.80% in simulation, 76.14% in physical PyTorch).
- **Outperforming Baselines:** Outperforms prior SOTA PFSR+MBH by **+3.66%** absolute joint accuracy with zero trainable parameters.
- **Latency & Speedup:** 
  - Under massive batch sizes ($B=256$), standard uncompiled PyTorch experiences framework overheads (slowdowns). However, at small batch scales ($B=16$), our vectorized scatter-gather implementation (SPS-VSG) achieves a verified physical **1.17$\times$ wall-clock speedup** out of the box on a sequential CPU (16.63 ms vs. MBH's 19.42 ms).
  - Analytical modeling projects a **3.90$\times$ speedup** (199.0 ms vs. 776.4 ms) under a compiler-fused memory loop layout.
- **Manifold Calibration:** IDC successfully restores routing balance under asymmetric task manifolds (recovering from 95.40% misrouting down to a balanced 47.00%).
- **OOD Performance:** The diagonal GMM coordinate density estimator rejects OOD queries with a **95.2% true positive rate** at a low 4.3% false positive rate.

## Explicitly Claimed Contributions (with Evidence)
1. **SPS-ZCA Formulation:** Introduces activation blending inside a single forward pass, converting sequential $O(K)$ latency back to $O(1)$ constant backbone pass scaling.
   *Evidence:* Flat latency profiles in Figure 1, throughput $>1000$ samples/sec, and physical $1.17\times$ speedup at low batch scales in Section 5.5.
2. **Zero-Shot Centroid Alignment (ZCA):** Solves the routing paradox and classification head dependency using early representation-space task centroids.
   *Evidence:* Achieving 100.0% routing accuracy on physical ViT-Tiny and 98.50% on GPT-2, outperforming PFSR by $+3.66\%$ joint accuracy.
3. **UNC, IDC, and GMM Coordinate Density Estimation:** Handles representation imbalances, manifold spread asymmetries, and OOD queries without joint training or parameter overhead.
   *Evidence:* Ablations C, D, and E show UNC restores accuracy under scale drift, IDC balances routing across compact vs. dispersed manifolds, and GMM rejects 95.2% of OOD inputs.
4. **Systems-ML Co-design and Generalizability Analysis:** Outlines hardware-aware memory-bandwidth models, compiled loop layout specifications, text-modality (GPT-2) validation with KV cache sharing, and boundary conditions.
   *Evidence:* Hardware cost modeling in Section 5.3, GPT-2 sequence classification/generation results in Section 5.6, and Appendix C/Appendix A.
