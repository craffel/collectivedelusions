# Peer Review of "QA-Merge: Quantization-Robust Centroid Routing for Low-Precision Edge Serving"

## 1. Summary of the Paper
The paper addresses a critical bottleneck in deploying dynamic, activation-space model ensembling systems on resource-constrained edge hardware. While dynamic latent coordinate merging methods (such as SABLE, ChemMerge, and Momentum-Merge) achieve remarkable continuous-task adaptation in full-precision (Float32) environments, compiling them under extreme low-precision limits (INT8 activations and INT4 ensembling weights) causes a catastrophic "Quantization Collapse." Rounding noise destroys subtle activation boundaries, causing task centroids to overlap and routing weights to freeze on the discrete grid, which degrades dynamic ensembling performance directly to the level of simple static Uniform Merging.

To resolve this bottleneck, the authors propose **QA-Merge (Quantization-Aware Merge)**, a framework comprising four lightweight, hardware-compatible techniques:
1. **Quantized Centroid Calibration (QCC):** Calibrates task-specific centroids directly in the target quantized integer representation space to maximize task separation, and uses scale-invariant cosine similarity in the integer coordinate space.
2. **Straight-Through Estimator (STE) Gating Optimization:** Employs the Straight-Through Estimator (STE) to bypass the non-differentiable rounding operator during offline optimization, allowing gradients to flow back to the gating heads.
3. **Error-Feedback Trajectory Stabilization (EF-Smooth):** Tracks ensembling coefficient rounding errors layer-by-layer and injects them back as a high-pass feedback correction, stabilizing representation trajectories on a 4-bit discrete simplex grid.
4. **Activation Error Feedback (AEF):** Accumulates representation rounding errors layer-by-layer and feeds them back residually, overcoming the "Small-Step Quantization Bottleneck" on coarse INT8 activation grids.

Evaluated inside the high-fidelity Interactive Coordinate Sandbox (ICS) environment, QA-Merge successfully recovers $\approx 100\%$ of full-precision ensembling gains under extreme low-precision constraints with zero downstream accuracy loss. Microarchitectural profiling and physical deployment benchmarks on an ARM Cortex-M7 microcontroller confirm a massive 5.2x latency speedup (0.18 ms vs 0.95 ms) and a 42% power reduction compared to Float32 execution, confirming real-world deployability.

---

## 2. Strengths and Weaknesses

### Strengths
1. **Rigorously Theoretically Grounded:** Unlike purely empirical papers that rely on heuristics, this work is grounded in solid mathematical analysis. The authors provide formal proofs of error bounds for both EF-Smooth (Appendix A.1) and AEF (Theorem 3.1 in Appendix A.2). Showing that AEF is a closed-loop error integrator that bounds cumulative activation rounding errors to a constant ($\frac{s_{\text{act}} \sqrt{D}}{2}$) independent of network depth provides a deep, mathematically sound explanation of why the proposed feedback loops prevent representational drift.
2. **Intellectual and Mathematical Honesty:** Commendable transparency is shown in Remark 1.1, where the authors clarify that Theorem 3.1 bounds numerical rounding errors relative to a local quantized-pull trajectory rather than the true continuous Float32 path. They then empirically validate that the actual trajectory divergence remains extremely small ($\approx 0.0413$ at Layer 14) and benign for classification, completing the analysis with high scientific rigor.
3. **Pragmatic, Hardware-Aware Systems Design:** The techniques are exceptionally lightweight, requiring no extra parameters and fewer than 50 integer operations per layer. The design of **PI-SPA** (Permutation-Invariant Single-Pass Apportionment) to make Hamilton's apportionment branchless and sorting-free on vector processors is highly creative and elegant.
4. **Exhaustive Quantitative Verification:** The experiments cover small-sample ($N_{\text{cal}}=64$) and large-sample ($N_{\text{cal}}=4000$) regimes, 6 different ensembling baselines, and a full sweep of the representation entanglement parameter $\rho \in [0.0, 0.5]$, proving that QA-Merge consistently recovers $\approx 100\%$ of full-precision ensembling gains.
5. **Physical Hardware Profiling:** Compiling the integer loop using CMSIS-DSP on an ARM Cortex-M7 microcontroller (confirming 0.18 ms latency, 5.2x speedup, and 42% power reduction) demonstrates high real-world deployability and bridges the gap between algorithmic theory and embedded systems.

### Weaknesses
1. **Lack of Large-Scale Transformer Evaluations:** The paper's primary experimental evaluations are conducted inside the Interactive Coordinate Sandbox (ICS). While the sandbox is an excellent, high-fidelity tool for isolating variables and validating ensembling theory, evaluating *only* in a simulator is a limitation. The paper would be significantly stronger if the authors included actual NLP (e.g., LLaMA, Mixtral) or computer vision (e.g., ViT) transformer benchmarks.
2. **Microarchitectural Bottleneck of Cosine Gating:** In Section 3.2, cosine similarity is computed in the integer coordinate space at each layer:
   $$d_{k, b} = \frac{Q(h_b^{(3)}, s_{\text{act}}, 8) \cdot c'_k}{\| Q(h_b^{(3)}, s_{\text{act}}, 8) \|_2 \| c'_k \|_2 + \epsilon}$$
   Computing the $\ell_2$ norm of the activation vector requires a sum of squares and a square root, followed by division. On constrained edge hardware (such as ARM Cortex-M7), square root and division are extremely expensive and slow. Executing this at every layer (layers 4 to 14) is a non-trivial bottleneck. The authors should explicitly discuss how to approximate these operations (e.g., using fast inverse square root or coordinate shift approximations) in an integer-only register pipeline.
3. **Violation of Uniform Rounding Bounds by Clipping:** The proofs of EF-Smooth and AEF assume uniform coordinate-wise rounding bounds. However, if the percentile-based static calibration or hardware clipping operator (Section 3.1) is triggered (e.g., due to severe out-of-distribution outliers), the rounding error can exceed these bounds, violating the theoretical guarantees. Bounding the probability or impact of clipping-induced errors would strengthen the mathematical analysis.
4. **Opportunity for Deeper Mathematical Positioning of Apportionment:** In Section 3.4, the authors utilize Hamilton's method of apportionment to project weights onto a discrete simplex grid. They present it as an algorithm but miss an opportunity to formally state its mathematical properties. It is a known result in discrete mathematics that Hamilton's method solves the exact global minimizer of the $\ell_1$ distance:
   $$\min_{\mathbf{q} \in \mathbb{N}^K, \sum q_k = L} \|\mathbf{q} - \boldsymbol{\alpha} L\|_1$$
   Formally noting this would strengthen the theoretical foundation of their work and show why this specific apportionment method is mathematically justified rather than just being an empirical heuristic.

---

## 3. Detailed Evaluations

### Soundness: Good
The technical claims are well supported, and the methodology is highly sound and appropriate:
- **AEF and EF-Smooth** are highly elegant solutions that directly target the mathematical root causes of Quantization Collapse (accumulating sub-grid updates and diffusing coefficient rounding errors).
- **The proofs** in Appendix A are correct, mathematically rigorous, and provide solid guarantees on numerical error bounds.
- **The physical hardware profiling** on ARM Cortex-M7 confirms that the ensembling loop is highly efficient (0.18 ms execution).
- *Justification for rating (Good instead of Excellent):* The proof assumptions rely on uniform rounding bounds, which are violated when clipping occurs (e.g., due to out-of-distribution outliers). Furthermore, computing cosine similarity (requiring square root and division) at every layer introduces a microarchitectural bottleneck that is not fully addressed or approximated in the text.

### Presentation: Excellent
The paper is exceptionally well-written and structured:
- The narrative is cohesive, tracking the journey from identifying "Quantization Collapse" to formalizing the mathematical correction layers, proving their error bounds, and verifying them empirically.
- Figures and tables are clean, highly informative, and directly support the text.
- The Appendix is highly complete, containing full pseudocode (Algorithms 1 & 2), a detailed hyperparameter table, diagnostic sweeps (SmoothQuant sensitivity, sample complexity, trajectory jitter, and decay factor $\beta$), and a concrete implementation roadmap.

### Significance: Good
The paper addresses a highly important problem (resource-constrained edge serving) and advances our understanding of dynamic low-precision model ensembling:
- Memory bandwidth and latency are critical bottlenecks on the edge. Enabling dynamic activation-space ensembling to run entirely within a low-precision integer pipeline (eliminating expensive format conversions and off-chip weight swapping) is a major systems breakthrough.
- The paper provides highly practical and actionable guidelines for edge deployment (e.g., setting the SmoothQuant parameter $\alpha \in [0.1, 0.3]$ and the EF-Smooth factor $\beta = 0.8$).
- *Justification for rating (Good instead of Excellent):* The evaluation is primarily restricted to the simulated Interactive Coordinate Sandbox (ICS). While the sandbox is high-fidelity, real-world evaluations on actual large-scale transformer benchmarks would elevate the impact and significance to Excellent.

### Originality: Excellent
The work provides highly novel insights and techniques:
- It is the first to identify and deconstruct "Quantization Collapse" in dynamic ensembling, and the first to apply quantization-aware optimization (STE) and error-feedback loops (EF-Smooth and AEF) to stabilize ensembling weights and representations.
- Modeling ensembling weight rounding errors as a first-order noise-shaping FIR high-pass filter is a creative and elegant bridge between digital signal processing and machine learning.
- The design of **PI-SPA** (Permutation-Invariant Single-Pass Apportionment) to make Hamilton's apportionment branchless and sorting-free on vector pipelines is a brilliant, highly original contribution.

---

## 4. Overall Recommendation
**Recommendation: 5 (Accept)**

**Summary of Recommendation:**
This is an incredibly strong, technically sound, and highly rigorous paper. It successfully bridges the gap between elegant algorithmic ensembling and practical edge-serving constraints. The authors do not merely present empirical heuristics; they ground their error-feedback loops in formal digital signal processing principles and provide correct mathematical proofs of error bounds. 

The proposed Permutation-Invariant Single-Pass Apportionment (PI-SPA) is an exceptionally elegant, sorting-free, and branchless solution to the integer apportionment bottleneck on vector pipelines. The complete recovery of Float32 ensembling ceilings under extreme INT8/INT4 constraints, coupled with physical microcontroller benchmarks showing a 5.2x latency speedup and 42% power reduction, makes this a complete and compelling work.

While the primary limitation is that evaluations are restricted to the simulated Coordinate Sandbox, the authors' high scientific rigour, intellectual honesty, and detailed implementation roadmap fully justify acceptance. I highly recommend this paper for publication.

---

## 5. Constructive Questions and Feedback for the Authors
1. **Integer Approximation of Cosine Gating:** Could the authors clarify how the square root and division required for scale-invariant cosine similarity (Eq. 4) are implemented in integer-only registers? Are there lightweight fixed-point approximations or look-up tables that can bypass this microarchitectural bottleneck?
2. **Clipping Probability and Bounds:** In Section 3.1, you discuss using percentile-based static calibration combined with clipping to handle out-of-distribution outliers. Since clipping violates the uniform rounding bounds assumed in the proofs, could you provide an analysis or bound on the probability of clipping, or discuss how clipping-induced errors propagate through AEF and EF-Smooth?
3. **Formalizing Hamilton Apportionment's Optimality:** It would strengthen the mathematical positioning of the paper if you explicitly noted that Hamilton's method of apportionment is the exact global minimizer of the $\ell_1$ distance between discrete allocations and continuous quotas. This adds a rigorous geometric interpretation to the simplex projection.
4. **Deterministic Bias in PI-SPA:** Under exact ties ($r_1 = r_2$), the static ID perturbation always favors the expert with the higher static ID. While ties are rare, did you consider a pseudo-random tie-breaker (e.g., using a fast hash function of the inputs as a seed) to eliminate this minor deterministic bias, and what would be its hardware overhead?
