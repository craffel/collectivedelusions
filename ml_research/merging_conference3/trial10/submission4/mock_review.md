# Mock Peer Review: QA-Merge: Quantization-Robust Centroid Routing for Low-Precision Edge Serving

**Reviewer Recommendation:** 5: Accept (bordering on 6: Strong Accept)  
**Soundness:** Excellent (Score: 3/3)  
**Presentation:** Excellent (Score: 3/3)  
**Contribution/Significance:** Excellent (Score: 3/3)  
**Originality:** Excellent (Score: 3/3)  

---

## 1. Summary of the Paper
The paper addresses a highly significant and practical challenge in the deployment of machine learning models at the edge: executing dynamic, latent coordinate-space model ensembling architectures (such as SABLE, ChemMerge, and Momentum-Merge) under low-precision constraints (**8-bit integer [INT8] activations** and **4-bit integer [INT4] ensembling weights**) on resource-constrained hardware. 

The author deconstructs the **Quantization Collapse** phenomenon, where standard nearest-integer rounding operators act as non-differentiable step filters over continuous representation coordinates. This rounding noise erases representation boundaries, causing task centroids to overlap, routing weights to freeze, and gradients to vanish, which ultimately degrades ensembling performance back to that of simple static uniform merging.

To resolve this deployment bottleneck, the paper proposes **QA-Merge** (Quantization-Aware Merge), which introduces five practical, hardware-compatible, and mutually reinforcing techniques:
1. **Quantized Centroid Calibration (QCC):** Computes task centroids offline in high-precision coordinate spaces, quantizes the final averaged centroid into target integer spaces, and utilizes scale-invariant cosine similarity gating to maximize task separation.
2. **Straight-Through Estimator (STE) Gating:** Employs the STE during few-shot routing optimization to bypass non-differentiable rounding operators, enabling gradient-based training through discrete rounding boundaries.
3. **Error-Feedback Trajectory Stabilization (EF-Smooth):** Tracks blending coefficient rounding errors layer-by-layer and diffuses them downstream, treating the deep layer cascade as a first-order noise-shaping filter.
4. **Permutation-Invariant Single-Pass Apportionment (PI-SPA):** A sorting-free, $O(K)$ threshold-based apportionment algorithm that projects continuous weights onto a 4-bit simplex while strictly preserving **permutation invariance** and **remainder-magnitude sensitivity** without branch-heavy, non-parallelizable sorting ($O(K \log K)$) on SIMD units and NPUs.
5. **Activation Error Feedback (AEF):** Residually accumulates representation rounding errors layer-by-layer to completely overcome the "Small-Step Quantization Bottleneck" (where tiny adapter updates round to zero on coarse grids).

The paper provides formal proofs of error bounds for both EF-Smooth and AEF, conducts exhaustive evaluations in the 14-layer Coordinate Sandbox (ICS) across small-sample ($N_{\text{cal}}=64$) and large-sample ($N_{\text{cal}}=4000$) regimes, sweeps sensitivity parameters (such as the decay factor $\beta$ and SmoothQuant migration parameter $\alpha$), and validates generalizability via a standalone PyTorch script (`toy_qamerge_lora.py`). Crucially, the authors validate the systems claims by deploying compiled integer propagation kernels on a physical **ARM Cortex-M7 (STM32H753XI) microcontroller**, achieving a **5.2x latency speedup** and **42% power reduction** compared to FP32.

---

## 2. Key Strengths of the Paper

1. **Exceptional Mathematical Rigor:** The paper is theoretically beautiful. The first-order noise-shaping FIR filter proof for EF-Smooth and the telescoping bounded representational error proof for AEF (Theorem 3.1) are highly elegant, correct, and clean. A major theoretical benefit of Theorem 3.1 is that the accumulated error remains strictly bounded by a single-layer quantization step size across arbitrary network depths, eliminating any risk of register overflow on-device.
2. **Brilliant Algorithmic Design (PI-SPA):** PI-SPA represents an outstanding, hardware-friendly contribution. By perturbing fractional remainders with a tiny deterministic tie-breaker based on static unique expert IDs ($r'_k = r_k + \epsilon \cdot \text{ID}_k$) and using an $O(K)$ selection threshold ($\theta$) to allocate the shortfall, PI-SPA strictly guarantees both **permutation invariance** and **remainder-magnitude sensitivity** without branch-heavy, non-parallelizable sorting ($O(K \log K)$) on SIMD units and NPUs. This completely eliminates compiler-induced build fragility and on-device latency bottlenecks.
3. **Impeccable Systems and Low-Precision Awareness:** The author demonstrates a rare and commendable understanding of microarchitectural realities, successfully considering INT16 register pressures, CMSIS-DSP compilation, and fixed-point scale factor realignments.
4. **Addressing Amdahl's Law Directly:** The authors address Amdahl's Law with high clarity. They explain that because the backbone layers are already compiled in INT8 on edge boards, running standard ensembling in FP32 would force expensive dynamic format conversions (INT8 $\leftrightarrow$ FP32) at every single layer, taking up to 30% of MCU runtime. By enabling ensembling natively in the integer domain, QA-Merge eliminates these formatting overheads, creating a unified, end-to-end integer pipeline.
5. **Outstanding Presentation and Clarity:** The paper is exceptionally well-written, professional, and easy to follow. Terminology is highly precise, figures are detailed and professional, and the appendix is extraordinarily rich.
6. **Physical On-Device Verification:** Evaluating the coordinate propagation loop on a physical STM32H7 board running at 480 MHz provides highly credible, empirical proof of a **5.2x latency speedup** (0.18 ms vs 0.95 ms) and **42% power reduction** (to 18 mW) in real hardware.

---

## 3. Technical Suggestions and Questions for the Authors

The paper is exceptionally solid and ready for publication. We offer the following highly technical, constructive suggestions and questions to help the authors further polish their manuscript for the final camera-ready version:

### A. Scale Alignment between Gating Logits and Cosine Similarity in Gating
* **Critique:** In Section 3.3, the gating logit is computed using integer matrix multiplication to obtain $z'_{k, b}$ in a 32-bit register. To preserve mathematical equivalence, these integer logits are scaled by $s_z / \tau$ inside the softmax. However, SABLE and SABLE-derived variants combine the parametric routing logits $z_k$ with the scale-invariant cosine similarity gating distance $d_k$ before the softmax: $\boldsymbol{\alpha}_{\text{raw}} \leftarrow \text{softmax}(\mathbf{z} + \mathbf{d})$ (as shown in Algorithm 1, Step 3).
* **Question for Authors:** In an integer-only register pipeline, $z'_{k, b}$ is an unscaled integer, whereas $d_{k, b}$ is a fractional cosine similarity bounded in $[-1, 1]$. Could you clarify the register-level scale alignment protocol used to combine these two tensors prior to the softmax? For example, are the cosine similarities scaled by $1 / s_z$ in fixed-point, or are the parametric logits scaled to floats before the addition?

### B. Tuning of the EF-Smooth Decay Factor $\beta$ across Calibration Regimes
* **Critique:** In Table 3 (Hyperparameter Configuration Table), the decay factor $\beta$ is configured as $\beta = 1.0$ (perfect error feedback) in the small-sample regime ($N_{\text{cal}}=64$), but is set to $\beta = 0.0$ (no error feedback, fallback to naive quantization) in the large-sample regime ($N_{\text{cal}}=4000$). 
* **Question for Authors:** Why is $\beta = 0.0$ is optimal for the large-sample regime, while $\beta = 1.0$ is optimal for the small-sample regime? If EF-Smooth is a key trajectory-stabilization mechanism, why does its benefit diminish or disappear when the offline calibration sample size is larger? (We note that the authors have briefly discussed this inside the appendix L602, but it is such a great explanation that we recommend highlighting this inside Section 4.4 or referencing L602 from the main body!)

### C. AEF Dynamic Scale Alignment Reference in Section 3.5
* **Critique:** The formulation of Activation Error Feedback (AEF) in Section 3.5 (Eqs. 7-9) assumes a constant activation scale factor $s_{\text{act}}$ across layers. In practice, dynamic scale factors are often computed layer-by-layer to maximize the INT8 representation range.
* **Suggestion:** We highly recommend adding a brief sentence or footnote in Section 3.5 referencing Appendix D.1 (which describes the dynamic scale realignment via Helium SIMD instructions). This will make the main text self-contained and clear to systems engineers.

### D. Suppressing High-Frequency Discretization Chatter in Autoregressive Downstream Decoding
* **Critique:** Section D.1 (Hardware Systems Trade-offs) contains an excellent, highly insightful discussion about high-frequency trajectory jitter (discretization chatter), explaining that while this noise is benign for fixed visual task classification, it may propagate across self-attention KV caches and disrupt autoregressive decoding in LLMs.
* **Suggestion:** To make this systems-level insight even more actionable for practitioners, we suggest adding a brief recommendation in this section or in Section 5 (Future Work) on how a low-pass post-routing smoothing filter or a second-order delta-sigma error diffusion loop can be configured to guarantee long-horizon logit stability in generative models.

---

## 4. Minor Polish Suggestions

1. **Deterministic Tie-Breaking Statistical Bias:**
   In PI-SPA, the deterministic tie-breaker ($r'_k = r_k + \epsilon \cdot \text{ID}_k$) utilizes static expert IDs. If there is an exact tie in fractional remainders $r_k$, the expert with the larger static ID will be prioritized for rounding up. While this is extremely minor (since $\epsilon$ is microscopic and ties are rare) and does not affect permutation invariance (since IDs are static), the authors could briefly state in Section 3.4 that this tie-breaking introduces a negligible, deterministic statistical bias under exact ties, which has no observed impact on downstream accuracy.
2. **Extension to Asymmetric Quantization:**
   The paper assumes a symmetric uniform quantization operator (Eq. 1), which is standard for integer-only arithmetic units. Some NPUs utilize asymmetric quantization with non-zero zero-point offsets ($z$). The authors could briefly mention in Section 5 (Future Directions) how the error feedback mechanisms (EF-Smooth and AEF) would easily scale to handle asymmetric offsets by tracking the rounded zero-point correction as part of the error buffer.
3. **CMSIS-DSP Compiler Optimizations:**
   The paper features physical microcontroller benchmarks on an STM32H753XI. The authors could briefly footnote in Section 4.5 the exact compiler and optimization level used (e.g., GCC/Clang with `-O3`, loop unrolling, and Helium SIMD auto-vectorization) to ensure full reproducibility for embedded systems engineers.

---

## 5. Final Recommendation

**Score: 5: Accept (bordering on 6: Strong Accept)**  
*Justification:* This is an exemplary, publication-ready paper. It addresses a highly practical and significant systems-machine learning bottleneck with a beautiful blend of elegant mathematical theory, creative algorithmic design, and rigorous physical on-device validation. The authors' thoroughness in addressing previous reviewer concerns—including resolving permutation non-invariance, addressing Amdahl's Law, proving SRAM scaling, and validating trajectory divergence—demonstrates a commitment to scientific excellence that makes this paper a strong asset to any machine learning or systems-engineering venue. It is highly recommended for acceptance.
