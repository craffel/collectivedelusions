# Peer Review

## Summary of the Paper
This paper addresses the deployment of dynamic model ensembling and weight-merging in latent coordinate spaces under resource-constrained, low-precision edge-hardware settings (e.g., 8-bit integer activations [INT8] and 4-bit integer ensembling weights [INT4]). The authors identify a "Quantization Collapse" phenomenon: standard uniform quantization introduces rounding noise that degrades dynamic ensembling, collapsing its performance to simple static uniform merging. 

To overcome this, the authors propose **QA-Merge** (Quantization-Aware Merge), a parameter-free suite of four lightweight, hardware-compatible techniques:
1. **Quantized Centroid Calibration (QCC):** Computes reference task centroids in a high-precision continuous space using calibration data before quantizing them to INT8, maximizing task separation, and utilizing scale-invariant cosine similarity.
2. **Straight-Through Estimator (STE) Gating:** Bypasses non-differentiable rounding operators during the calibration of parametric routers, allowing gradient-based optimization through rounding boundaries.
3. **Error-Feedback Trajectory Stabilization (EF-Smooth):** Tracks ensembling weight rounding errors layer-by-layer and injects them back to the subsequent layer as a high-pass feedback correction. It utilizes a novel, sorting-free **Permutation-Invariant Single-Pass Apportionment (PI-SPA)** algorithm to project weights onto the discrete 4-bit simplex.
4. **Activation Error Feedback (AEF):** Residually accumulates sub-grid activation rounding errors across layers and feeds them back, overcoming the "Small-Step Quantization Bottleneck" where small updates are rounded to zero.

The authors evaluate QA-Merge inside the Analytical Coordinate Sandbox (ICS) environment, showing complete recovery of full-precision ensembling gains. They also provide physical benchmarks on an STM32H753XI microcontroller (Cortex-M7) demonstrating a 5.2x speedup on the ensembling loop, and provide a PyTorch toy implementation demonstrating a dynamic LoRA-mixture.

---

## Strengths and Weaknesses

### Soundness, Presentation, Significance, and Originality

#### Soundness:
- **Strengths:** Individually, the proposed mechanisms (QCC, STE gating, EF-Smooth, and AEF) are highly appropriate and logically address the specific causes of quantization collapse. The mathematical formulation is sound, and the formal proof of Theorem 3.1 (Telescoping Bounded Representational Error of AEF) is theoretically rigorous, successfully bounding cumulative representation rounding errors relative to the local quantized-state trajectory.
- **Weaknesses:** While mathematically rigorous, the evaluation relies heavily on a synthetic coordinate-space simulator (ICS). The paper lacks validation of the proposed QA-Merge on a real-world deep backbone network (e.g., LLaMA-3, Mistral, ResNet, or Vision Transformer) running on actual dataset streams. The PyTorch script (`toy_qamerge_lora.py`) is merely a toy demonstration on random input tensors. Additionally, the microcontroller latency speedup is evaluated exclusively on the coordinate ensembling loop rather than end-to-end on a physical hardware deployment.

#### Presentation:
- **Strengths:** The presentation quality is exceptional. The manuscript is highly polished, beautifully structured in ICML style, and incredibly comprehensive. Notations are consistent, the figures are professional, and potential technical/systems concerns are proactively and thoroughly addressed across the detailed appendix sections.
- **Weaknesses:** None of significance. The writing and organization are exemplary.

#### Significance:
- **Strengths:** Dynamic model ensembling at the edge is a highly relevant problem. QA-Merge offers a practical, parameter-free, and computationally light framework to maintain a unified integer-only pipeline without dynamic float-to-int format conversions, which are notorious bottlenecks on resource-constrained microprocessors.
- **Weaknesses:** The significance of the empirical findings is currently constrained by the simulated and synthetic nature of the main experiments. Because QA-Merge is not evaluated on real-world backbone models and datasets, it remains unverified whether the proposed techniques generalize to actual non-linear representation manifolds under real dataset distributions.

#### Originality:
- **Strengths:** While individual components like the Straight-Through Estimator or error-diffusion feedback are established concepts, their creative synthesis into a unified, low-precision latent ensembling framework (**QA-Merge**) is highly original. The proposed **PI-SPA** algorithm is a particularly creative, sorting-free, and branchless $O(K)$ alternative to Hamilton's apportionment, showing strong system-level originality.
- **Weaknesses:** None of significance.

---

## Detailed Strengths

1. **Pragmatic, Low-Overhead Engineering:** The suite is parameter-free, requiring fewer than 50 elementary integer operations per layer, making it highly practical for physical edge microprocessors.
2. **Permutation-Invariant Single-Pass Apportionment (PI-SPA):** Resolving the sorting bottleneck ($O(K \log K)$) of standard Hamilton apportionment with a branchless, sorting-free $O(K)$ algorithm that uses static expert ID perturbation and parallel threshold selection is a highlight of the work. This is highly compatible with SIMD and vector architectures.
3. **Hardware & Mathematical Completeness:** Commendably, the paper includes formal mathematical proofs (Theorem 3.1), physical microcontroller deployments (Cortex-M7, demonstrating 0.18 ms vs 0.95 ms FPU latency), detailed hyperparameter tables, and runnable PyTorch artifacts, which provide a complete and reproducible deployment blueprint.
4. **Proactive System Analysis:** The detailed discussions in Appendix E on accumulator requirements, scale realignment, trajectory jitter (chatter) in generative settings, and SRAM scaling are excellent and show deep familiarity with physical system constraints.

---

## Detailed Weaknesses (Empirical Critiques)

1. **Lack of Real-World Backbone and Dataset Validation:** 
   The primary empirical limitation is that all main results (Table 1 and Table 2) are conducted strictly inside the Analytical Coordinate Sandbox (ICS) using synthetic task signatures. While ICS is a useful environment for isolating ensembling dynamics, it remains a synthetic simulator. The authors claim in Section 4.1 that "the coordinate-space dynamics of QA-Merge evaluated here generalize directly to any deep model's latent space." However, without validating QA-Merge on a real backbone (e.g., LLaMA-3 with quantized LoRA expert adapters, or a quantized ViT with expert adapters) on real datasets (e.g., GLUE, ImageNet, CommonsenseQA), this generalization claim is empirically unsupported. The PyTorch script is a toy validation on random input tensors.
2. **Missing Statistical Significance Metrics:**
   Across all accuracy tables in the paper and the appendix, the authors report only single-point accuracies (e.g., "79.20%", "65.80%"). Despite these experiments being run inside a synthetic simulator where representation covariance and task experts are simulated, **no standard deviations, confidence intervals, or error bars** are reported. For a simulation study, evaluating performance across multiple random initializations/seeds and reporting statistical confidence is an essential empirical standard to rule out random variance.
3. **End-to-End Hardware Latency Impact under Amdahl's Law:**
   The physical microcontroller benchmark (5.2x latency speedup and 42% power reduction on STM32H753XI) is evaluated exclusively on the coordinate propagation/ensembling loop itself. As the authors themselves disclose in Appendix E: "because the heavy backbone layers of a deep neural network consume the vast majority of execution time, accelerating a tiny ensembling loop yields a seemingly negligible end-to-end speedup." Without profiling the end-to-end model latency on the hardware, the actual hardware benefit of accelerating this specific loop remains unquantified.

---

## Detailed Feedback and Questions for Authors

1. **On Real Backbone Validation:** Can the authors provide empirical results of QA-Merge applied to a real-world pre-trained backbone model (e.g., a quantized 1B/3B LLM with task-specific LoRA adapters, or a quantized ResNet/ViT model) on actual downstream datasets? This would dramatically strengthen the paper's main claims regarding real-world deployability.
2. **On Statistical Rigor:** Please report standard deviations or confidence intervals across multiple random seeds for Table 1 and Table 2. Since this is a simulated sandbox environment, establishing statistical significance is critical.
3. **On PI-SPA Tie-Breaking:** The PI-SPA algorithm uses a static expert ID perturbation ($r'_k = r_k + \epsilon \cdot \text{ID}_k$) to resolve ties deterministically. While the authors state that exact ties are exceptionally rare, does the deterministic statistical bias of using static IDs have any observable impact on ensembling fairness or accuracy in highly symmetric settings where remainders frequently overlap?
4. **On End-to-End Profiling:** For a standard model (such as a 4-bit quantized MobileNet or a quantized 1B LLM), what percentage of the total end-to-end inference budget on the microcontroller is typically consumed by the ensembling/blending loop versus the backbone? Quantifying this would clarify the real-world impact under Amdahl's Law.

---

## Ratings

### Soundness: Good
The theoretical analysis (Theorem 3.1) and the individual components of the methodology are mathematically sound and well-reasoned. However, because the evaluation is confined to a synthetic simulator and lacks validation on real-world backbones and standard datasets, the empirical soundness is limited.

### Presentation: Excellent
The manuscript is beautifully written, exceptionally clear, and structured with outstanding attention to detail. The authors address potential systems trade-offs with commendable depth in the appendix.

### Significance: Good
The paper addresses a highly practical problem (low-precision deployment of dynamic ensembling) and proposes elegant, lightweight solutions. If validated on real-world models, the potential impact on edge-adaptive intelligence is high, though it is currently limited by the scope of the evaluation.

### Originality: Excellent
The synthesis of error-diffusion noise shaping with dynamic coordinate ensembling is highly creative, and the proposed Permutation-Invariant Single-Pass Apportionment (PI-SPA) is a novel and very practical contribution to the system level.

---

## Overall Recommendation

**Rating: 4 (Weak Accept)**

This is a technically solid, exceptionally well-written, and comprehensive paper that successfully bridges the gap between high-precision dynamic coordinate ensembling and low-precision edge deployments. The mathematical proof of Theorem 3.1 is rigorous, the proposed PI-SPA algorithm is highly creative, and the physical microcontroller benchmarks show impressive local gains. 

However, the contribution's impact is currently limited by its evaluation, which is restricted to a synthetic coordinate-space simulator and a toy PyTorch script. Validating the framework on actual real-world backbone models and datasets, and reporting statistical metrics (error bars/standard deviations) for the accuracy tables, are critical steps to elevate this work to a strong accept. I recommend a Weak Accept, urging the authors to address these empirical gaps.
