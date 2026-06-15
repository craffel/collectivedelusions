# 2. Novelty and Delta Analysis

## Conceptual Delta from Prior Art
A thorough analysis of prior art confirms that **Q-SPS** / **CG-Q-SPS** represents a significant and highly pragmatic advancement in edge serving of specialized experts, combining dynamic routing, quantization, and systems-level execution-gating in a training-free framework.

### 1. Contrast with Static Model Merging (e.g., Task Arithmetic, TIES-Merging, DARE)
* **Prior Art:** Static parameter-space merging methods average or prune model weights. While they require zero runtime memory or compute overhead, they suffer from severe **"heterogeneity collapse"** under highly mixed, interleaved input streams because they cannot adapt dynamically on a per-sample basis.
* **Proposed Delta:** Q-SPS operates in the *activation space*, performing sample-wise dynamic ensembling inside a single parallel forward pass. It achieves absolute immunity to heterogeneity collapse while maintaining the performance of independent experts.

### 2. Contrast with Dynamic PEFT Serving and Systems Pipelines (e.g., S-LoRA, Punica, PFSR + MBH SOTA)
* **Prior Art (GPU-centric):** S-LoRA and Punica manage multi-LoRA serving on high-end GPU clusters via custom memory paging kernels, which do not translate to low-power edge CPUs or microcontrollers where scheduling and memory paging overheads are severe.
* **Prior Art (Edge-centric):** PFSR with Micro-Batch Homogenization (MBH) avoids dynamic interference by partitioning heterogeneous batches into homogeneous sub-batches. However, this sequential dispatching requires executing the massive base model backbone up to $K$ times, multiplying execution latency and violating strict real-time mandates.
* **Proposed Delta:** Q-SPS performs ensembling inside a single parallel pass ($O(1)$ constant backbone latency), bypassing the sequential micro-batch bottleneck and delivering a projected **3.97$\times$ physical speedup** under highly interleaved streams.

### 3. Contrast with Existing Activation Blending (e.g., SABLE, SPS-ZCA)
* **Prior Art:** SABLE and SPS-ZCA perform sample-wise activation ensembling, but are strictly limited to unquantized, high-precision floating-point execution (FP32/FP16). This fails to exploit native edge integer accelerators (NPUs, ARM Neon) and incurs substantial SRAM/cache footprint and DRAM transfer bandwidth overheads.
* **Proposed Delta:** Q-SPS is the first to execute activation-space dynamic blending in **pure low-precision integer precision (INT8/INT4)**, enabling native on-device acceleration.

### 4. Technical Novelty of Specific Components
* **Quantization-Aware Scale Calibration (QASC):** Quantizing independent experts introduces severe representation noise because different experts develop disparate dynamic ranges. Standard uniform RTN or MinMax quantization degrades low-bitwidth representation alignment. QASC is a training-free post-hoc calibration that decouples the joint down- and up-projection scale optimization sequentially, reducing complexity from $O(N^2)$ to $O(N)$ (0.05s on CPU) and restoring **99.5%** of the unquantized float accuracy.
* **Conditional Expert Gating (CG-Q-SPS):** Under sharp, low-temperature routing regimes, ensembling all experts in parallel is highly wasteful because most expert activations are ultimately scaled by near-zero coefficients. CG-Q-SPS resolves this architectural contradiction by introducing a conditional bypass ($\theta = 0.01$) that dynamically skips executing inactive expert pathways on a per-sample basis. This losslessly scales adapter execution compute as $1/K$ and prevents cache pollution and DRAM transfer overheads.
* **Coordinate GMM Safety Shield:** Rather than fitting high-dimensional density estimators (which are computationally heavy and vulnerable to feature drift), CG-Q-SPS projects Layer 3 features into a low-dimensional similarity coordinate space ($K=4$) and fits a lightweight diagonal GMM. This achieves an outstanding AUC of **0.98** (95.2% TPR at 4.3% FPR) with negligible $O(K)$ computational complexity.

## Characterization of Novelty
The novelty of this paper is **highly significant and exceptionally practical**. It represents a **systems-machine learning co-design** explicitly engineered for the physical constraints of low-power microprocessors. By resolving the parallel-expert execution contradiction, analyzing cache locality via local batch re-ordering, characterizing the HLC Pareto frontier, and optimizing post-training calibration, the authors provide a comprehensive, actionable, and theoretically robust contribution.
