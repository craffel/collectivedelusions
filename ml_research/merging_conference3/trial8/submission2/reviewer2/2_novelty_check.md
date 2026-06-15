# Intermediate Evaluation: 2_novelty_check.md

## Key Novel Aspects and "Delta" from Prior Work

The paper presents several major advancements and key differences compared to established work in Parameter-Efficient Fine-Tuning (PEFT) serving and dynamic model merging:

1. **Pure Integer Precision Activation-Space Blending:** 
   Prior ensembling pipelines like S-LoRA, Punica, and PFSR either target massive server-class GPUs with specialized CUDA kernels or utilize sequential batch partitioning (MBH) which multiplies latency by running the backbone model multiple times. While unquantized activation blending (SPS-ZCA) resolves the latency-heterogeneity trade-off, it serves unquantized floating-point (FP16/FP32) weights, which fails to exploit edge-native low-power integer accelerators (DSP, ARM Neon, NPUs) and requires heavy DRAM-SRAM transfer bandwidth.
   * **The Delta:** Q-SPS is the first framework to execute dynamic activation-space ensembling natively in low-bit symmetric integer precision (INT8/INT4). This represents a major breakthrough for resource-constrained IoT nodes and edge microcontrollers.

2. **Quantization-Aware Scale Calibration (QASC):**
   Standard post-training quantization (PTQ) techniques (like RTN or independent MinMax calibration) destroy the delicate geometric alignment of low-rank subspaces in low-bitwidths, causing severe representation noise and accuracy degradation.
   * **The Delta:** QASC introduces a training-free, sequentially decoupled scale calibration. It optimizes clipping bounds by sequentially minimizing the Mean Squared Error of the down-projection and up-projection layers. By decoupling the joint optimization, it collapses the search complexity from $O(N^3)$ to $O(N)$, running in under 0.1 seconds per layer while recovering 99.5% of the unquantized float accuracy.

3. **Conditional Expert Gating (CG-Q-SPS) to Resolve Parallel ensembling Contradiction:**
   A fundamental execution contradiction in parallel ensembling is that while parallel execution is desired for a single-pass backbone, ensembling all $K$ expert adapters is highly inefficient since most experts are scaled by near-zero coefficients under sharp routing temperature.
   * **The Delta:** CG-Q-SPS resolves this by applying a gating threshold ($\theta = 0.01$). It dynamically bypasses the execution of inactive expert adapters, achieving $O(1)$ constant latency for the frozen base backbone while dynamically scaling active adapter compute load to be proportional only to the active experts ($O(1)$ under confident routing, scaling gracefully to $O(p)$ at ambiguous boundaries). For high-density registries ($K=24$), this scales adapter compute load as $1/K$ (consuming under 5% active load).

4. **Systems-Level Edge CPU Co-Design Optimizations:**
   Real-world edge deployments have complex physical bottlenecks like cache pollution, instruction stalls from format conversions, dynamic register unpacking (INT4), and thread synchronization barriers. The paper goes far beyond basic algorithmic novelty by proposing highly pragmatic co-design features:
   * **Local Batch Re-Ordering:** Predetermining routing coordinates at early Layer 3 allows sorting the batch samples before entering Layer 4, grouping identical active expert paths to maximize temporal cache residency and eliminate DRAM weight swapping.
   * **Temporal-Aware Routing Hysteresis:** An EWMA coordinate filter for sequential $B=1$ serving that suppresses representation-space noise, limiting routing flicker to under 0.8% and preventing cache-thrashing on microcontrollers.
   * **Adaptive Battery-Aware Gating:** A software-defined knob that dynamically increases the gating threshold as battery level declines, scaling down active expert compute to conserve physical energy under low-power regimes.

5. **Exposing the Redundancy of Explicit Centroid Orthogonalization:**
   The paper explores Gram-Schmidt Cross-Centroid Orthogonalization (GS-CCO) and L&ouml;wdin Symmetric Manifold De-Entangling (SMD) as theoretical methods to de-entangle centroids.
   * **The Delta:** Instead of claiming these complex orthogonalization techniques as necessary performance-boosters, the authors provide an intellectually honest finding that explicit orthogonalization is actually mathematically redundant and even detrimental due to noise spillover in joint projection spaces. They prove that the simpler, raw unorthogonalized ZCA-IDC baseline remains the most robust overall under severe entanglement.

---

## Characterization of Novelty

The novelty of this paper is **highly significant and deeply pragmatic**. It does not merely tweak an existing algorithm; it provides a comprehensive, co-designed system that bridges the gap between deep-learning ensembling algorithms and physical microprocessor hardware constraints (caches, DRAM bandwidth, register unpacking, battery power). 

Rather than focusing on purely academic, high-performance margins on large GPU clusters, the authors tackle the actual, deployment-critical bottlenecks of low-power edge CPUs and microcontrollers. The introduction of sequentially decoupled calibration (QASC), conditional gating (CG-Q-SPS), local batch re-ordering, temporal routing hysteresis, and adaptive battery-aware gating provides a complete systems-ML playbook for edge practitioners. The intellectual honesty of evaluating orthogonalization methods and showing their empirical drawbacks under noise further elevates the quality and maturity of the scientific contribution.
