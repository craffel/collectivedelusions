# 1. Summary of the Paper

## Main Topic and Problem Addressed
This paper addresses the deployment-critical bottleneck of serving multiple specialized Parameter-Efficient Fine-Tuning (PEFT) experts simultaneously on resource-constrained edge hardware (such as mobile phones, smartwatches, IoT nodes, and low-power microcontrollers). 
Standard LoRA adapters are efficient in parameter size, but serving dozens of concurrent experts in floating-point precision (FP32/FP16) exceeds the tiny on-chip SRAM capacity of edge devices. This forces constant reloading of expert weights from main DRAM to cache/SRAM, destroying memory-bandwidth efficiency and causing severe execution latency.

Furthermore, standard routing approaches suffer from a latency-heterogeneity trade-off:
- Statically merging expert weights in parameter space (e.g., Task Arithmetic, TIES-Merging) collapses under heterogeneous, task-interleaved input streams (heterogeneity collapse).
- Sequential batch partitioning and dispatching (such as Micro-Batch Homogenization / MBH) requires up to $K$ sequential forward passes of the massive base model backbone, multiplying latency and violating strict real-time response constraints.

## Proposed Approach: Q-SPS and CG-Q-SPS
To resolve these bottlenecks, the authors propose **Q-SPS** (Quantized Single-Pass Activation-Space Dynamic Blending) and its execution-gated variant **CG-Q-SPS** (Conditional Gated Q-SPS), a training-free systems-ML co-designed framework featuring:
1. **Low-Rank Weight-Quantized LoRA Experts:** Quantizes task-specific LoRA adapters (rank $r=8$) to low-bitwidth symmetric integer formats (INT8/INT4).
2. **Integer-Precision Activation-Space Blending:** Performs dynamic sample-wise expert ensembling entirely in high-throughput integer precision natively accelerated on edge CPUs and NPUs inside a single parallel forward pass ($O(1)$ constant backbone latency), bypassing the sequential micro-batch partitioning of SOTA baselines.
3. **Quantization-Aware Scale Calibration (QASC):** A post-hoc, training-free calibration protocol that sequentially decouples down-projection and up-projection scale optimization (reducing search complexity from $O(N^2)$ to $O(N)$) to minimize discretization rounding noise and activation outlier clipping.
4. **Conditional Expert Gating (CG-Q-SPS):** Losslessly bypasses executing expert adapter pathways whose routing coefficients fall below a threshold ($\theta=0.01$). This resolves the parallel ensembling execution contradiction, reducing DRAM load volume and active compute.
5. **Zero-Shot Centroid Alignment (ZCA) with IDC:** Routes inputs task-agnostically in Layer 3 representation space, combined with a lightweight diagonal Coordinate Gaussian Mixture Model (GMM) safety shield for robust Out-of-Distribution (OOD) task rejection.

## Key Findings and Quantified Claims
* **Absolute Immunity to Collapse:** Under heterogeneous streams, CG-Q-SPS (INT4 + QASC) remains completely immune to both "heterogeneity collapse" and "quantization collapse," preserving an outstanding simulated **79.40% Joint Mean** and recovering **99.5%** of the unquantized FP32 ensembling ceiling (79.80%).
* **Massive Memory Footprint Savings:** Quantizing LoRA adapters to 4-bit precision slashes the combined expert memory footprint from 2.76 MB to **0.345 MB** (a massive **87.5% savings**), allowing dozens of active experts to fit natively inside microcontroller SRAM or shared L1/L2 caches ($<512$ KB).
* **High-Throughput Acceleration:** By bypassing the sequential sub-batch dispatching of MBH (which takes 749.8 ms cumulative latency over 1,024 heterogeneous samples), CG-Q-SPS processes mixed batches in a single parallel pass with fast integer arithmetic, consuming only **189.1 ms** cumulative latency—a projected **3.97$\times$ physical speedup**.
* **Robust OOD Rejection:** The Coordinate GMM safety shield filters out high-dimensional visual noise early, achieving an outstanding **AUC of 0.98** (95.2% TPR at 4.3% FPR) with a lightweight $O(K)$ computational complexity suitable for edge processors.
* **Empirical Multi-Block Validation:** On physical pre-trained ViT-Tiny weights across Blocks 5, 9, and 12, QASC Dynamic and Static scaling both reduce relative reconstruction MSE from 6.68% (RTN baseline) to **2.80%**, recovering output cosine similarity to **0.9861**. Under end-to-end multi-layer compounding simulation over all 12 blocks, QASC Static Scaling successfully prevents error propagation, achieving **84.38%** top-1 prediction agreement and slashing relative logit MSE to **1.20%**.
