# Novelty and Originality Check

## 1. Assessment of Novelty
The paper addresses an important and practical challenge: **how to execute dynamic, activation-space model ensembling architectures under low-precision (INT8/INT4) edge constraints.**

The core novelty is the combination and adaptation of established concepts from different fields (such as signal-processing error-diffusion, political census apportionment, and modern hardware-aware quantization) to solve quantization collapse in representation-space model ensembling. While the individual components are inspired by established techniques, their integration and optimization for this specific hardware-constrained bottleneck are highly creative:

* **Identification of the "Quantization Collapse":** The paper systematically identifies that quantization noise projects continuous representation coordinates onto a coarse, discrete grid, which collapses dynamic distance-based routing into static uniform merging.
* **Permutation-Invariant Single-Pass Apportionment (PI-SPA):** Standard ensembling weights must sum to exactly 1.0 (or 15 levels under 4-bit representation). To perform this discrete projection without the non-parallelizable $O(K \log K)$ sorting bottleneck of Hamilton's method, the authors propose **PI-SPA**. By perturbing the fractional remainders $r_k$ with a tiny deterministic tie-breaker based on static unique expert IDs ($r'_k = r_k + \epsilon \cdot \text{ID}_k$), and then finding the $S$-th largest perturbed remainder as a threshold $\theta = \text{Select}(\{r'_j\}_{j=1}^K, S)$ using an $O(K)$ parallel Selection algorithm, PI-SPA strictly guarantees both **permutation invariance** and **remainder-magnitude sensitivity** while being completely branchless and parallelizable on specialized SIMD vector pipelines.
* **EF-Smooth (Error-Feedback Trajectory Stabilization):** Inspired by signal-processing error diffusion (halftoning), EF-Smooth tracks rounding errors of blending coefficients and diffuses them downstream, treating the deep layer cascade as a first-order noise-shaping filter.
* **Activation Error Feedback (AEF):** Solves the "Small-Step Quantization Bottleneck" (where sub-grid activation updates round to zero on coarse grids) by residually accumulating representation rounding errors across layers.
* **Dynamic Outlier-Aware Activation Scaling (SmoothQuant Calibration):** Integrates an offline scaling-migration matrix $S$ with a strength parameter $\alpha \in [0.1, 0.3]$ to migrate dynamic activation outlier difficulty onto static weights and centroids.

---

## 2. Detailed Comparison with Related Work

| Dimension | Standard Quantization (PTQ/QAT) | Standard Model Merging (SABLE/ChemMerge) | Proposed QA-Merge |
| :--- | :--- | :--- | :--- |
| **Precision Assumption** | Compresses Float32 models to INT8/INT4. | Assumes Float32 activations and ensembling weights. | Integrates dynamic ensembling directly into INT8/INT4 spaces. |
| **Routing / Gating** | Static routing (standard layers). | Continuous distance-based routing (Float32). | Scale-invariant cosine similarity in discrete integer space; QCC. |
| **Gating Optimization** | Standard backpropagation. | Training-free, or basic SGD in Float32. | STE-based gradient propagation through discrete rounding boundaries. |
| **Discretization Noise** | Handled via fine-tuning or scaling factors. | Ignored (treated as a non-issue in Float32). | Active noise-shaping via **EF-Smooth** and **PI-SPA Discrete Simplex Projection**. |
| **Under-threshold Updates** | Rounded to zero (small-step loss). | N/A (continuous floats). | **Activation Error Feedback (AEF)** accumulates and restores sub-grid updates. |
| **Outlier Handling** | LLM.int8() mixed-precision or SmoothQuant. | Ignored. | Dynamic Channel Scaling with **SmoothQuant Calibration ($\alpha \in [0.1, 0.3]$)**. |

---

## 3. Practical Impact and Significance (The Pragmatist's View)
The paper's practical novelty and significance are exceptionally high:

1. **Strict Permutation Invariance:** The proposed PI-SPA algorithm elegantly solves the lack of permutation invariance inherent to simpler sequential apportionment schemes. This ensures that the ensembling results are completely stable across different compilations, memory layouts, and expert configurations, which is a major requirement for robust edge deployment.
2. **Unified Integer Pipeline via Amdahl's Law Mitigation:** The paper addresses Amdahl's Law directly. While the coordinate propagation loop is computationally small, running it in Float32 would force expensive dynamic format conversions (INT8 $\leftrightarrow$ FP32) at every single layer because the backbone layers are already compiled in INT8. By keeping the ensembling operations natively in the integer domain, QA-Merge completely eliminates these format-conversion overheads (which consume up to 30% of total latency on microcontrollers), enabling a fully unified, end-to-end integer pipeline.
3. **Physical Hardware Verification:** The 5.2x latency speedup and 42% power reduction measured on a real physical Cortex-M7 board running compiled CMSIS-DSP integer kernels provide concrete empirical proof of the systems-level benefits of this approach.
4. **Frugal SRAM Scaling:** Showing that AEF consumes only 8 KB of SRAM per layer at $D=4096$ scale verifies that stateful tracking does not cause memory-bandwidth bottlenecks or cache thrashing on edge accelerators.

## Conclusion on Novelty
The paper presents an **exceptionally creative, cross-disciplinary, and highly practical combination of algorithmic techniques** (STE, error diffusion, PI-SPA, and SmoothQuant calibration) to address the quantization collapse in dynamic coordinate-space ensembling. By resolving the permutation-invariance issue, validating on actual microcontroller hardware, and proving memory and execution scaling, the paper establishes real-world significance and provides a solid contribution to the low-precision edge intelligence domain.
