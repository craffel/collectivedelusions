# 5. Presentation, Impact, and Significance Evaluation

## Major Strengths

1. **Exemplary Mathematical and Physical Rigor:** The paper is highly rigorous. Instead of relying on heuristic Euclidean shortcuts, the authors model spacecraft trajectory updates strictly on the curved unit hypersphere $\mathbb{S}^{D-1}$ by incorporating local tangent space projections, exact spherical geodesic steps (the Exponential Map), and closed-form Parallel Transport of the velocity vector to prevent numerical drift.
2. **Elegant Control-Theoretic Foundation:** In Section 3.5, the authors establish a rigorous control-theoretic proof demonstrating that first-order smoothers (like EMA and ChemMerge) act as low-pass filters with a transfer function $H_1(s)$ that introduces a severe phase lag and delay in closed feedback loops, triggering overshoots and oscillations. In contrast, GraviMerge's second-order spring-mass-damper formulation acts as a highly active low-pass filter (decaying at $-40$ dB/decade) that proactively pulls the spacecraft probe toward target centroids, mathematically breaking the accuracy-stability barrier.
3. **Outstanding Empirical Gains:** GraviMerge achieves **88.69%** serving accuracy while slashing layer-wise routing jitter to **0.00190 MAD**, representing a spectacular **6.01$\times$** reduction compared to ChemMerge, **5.47$\times$** compared to EMA, and **2.40$\times$** compared to SABLE, completely dominating the Accuracy-Stability Pareto Frontier.
4. **Exhaustive Appendix and Ablations:** Sibling notes and evaluations in the Appendix are incredibly thorough, spanning:
   * **Algorithm 1:** A step-by-step implementation guide for Decoupled and Coupled controllers.
   * **Integration Blueprint & Memory Analysis:** Detailed memory footprints and scaling strategies (Low-Dimensional Spacecraft Projection, Block-Structured Geodesic Integration) to make token-wise routing memory-efficient on consumer GPUs.
   * **OOD Gating (Sentinel Attractor Dynamics):** A confidence-gated fallback mechanism that forces the probe to the geometric barycenter (uniform expert mixture) under OOD streams, validated empirically.
   * **Auto-Tuning Extensions:** AGS (Adaptive Gravitational Scheduling) and Adaptive Viscous Drag schedulers, verifying self-calibration.
   * **Latency Profiling:** Wall-clock latency benchmarks showing negligible overhead (under 4 ms on Llama-3 scale dimensions $D=4096$).

---

## Areas for Improvement

1. **Downstream LLM Benchmark Gap:** The primary evaluation is restricted to the Projected Digit Representation Space (RDS) Proxy. While high-fidelity and mathematically robust, evaluating downstream text generation on standard benchmarks (e.g., MMLU, GSM8k) using a physical LLM (like Llama-3-8B) with active LoRA experts would fully validate the simulation-to-reality transition.
2. **Contextualizing Real-World Test-Time Merging Baselines:** From a scholarly perspective, the literature review and discussion would be stronger if the paper situated its contributions relative to emerging real-world instance-level dynamic LoRA selection/merging frameworks such as **LoRA on the Go (LoGo)** (ACL 2026), **DA-MergeLoRA**, and **TTMM (Test-Time Model Merging)**.
3. **Hardware Latency Constraints of Coupled Mode:** In Coupled GraviMerge, the feedback force introduces a sequential dependency on intermediate layer activations. The authors should more explicitly discuss the hardware parallelization bottlenecks of Coupled Mode compared to Decoupled Mode (which allows complete parallel pre-computation of all layer weights immediately after Layer 3).

---

## Overall Presentation Quality
We rate the overall presentation quality as **Excellent**. 
* The manuscript is structured logically and reads beautifully. 
* The mathematical notation is elegant, clean, and consistent. 
* The visualizations are highly professional: Figure 1a clearly shows the smooth, oscillation-free trajectory of GraviMerge's ensembling weights compared to the volatile drift of ChemMerge and high-frequency oscillations of SABLE, and Figure 1b provides a clean depiction of the Accuracy-Stability Pareto Frontier.

---

## Potential Impact and Significance
The potential impact of this paper is **high**:
* **High Practical Utility:** Test-time ensembling of multiple specialized experts (LoRA) is crucial for low-latency multi-task edge serving. Solving the stability-accuracy dilemma of dynamic model merging enables highly stable, real-time, multi-expert deployment on resource-constrained hardware.
* **Inspirational Paradigm:** Bridging deep learning ensembling with second-order physics and differential geometry can inspire researchers to design robust, stable deep learning controllers grounded in physical, geometric, and control-theoretic principles.
