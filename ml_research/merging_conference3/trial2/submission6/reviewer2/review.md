# Peer Review: Q-Merge: A Pragmatic Approach to Quantization-Aware Model Merging under Extreme Deployment Constraints

## Summary of the Paper
This paper presents **Quantization-Aware Model Merging (Q-Merge)**, a lightweight, test-time adaptation framework that enables the deployment of multi-task merged neural networks under strict memory and hardware constraints. Weight-space model merging (e.g., Task Arithmetic) is a highly efficient way to fuse specialized expert models, but subsequent low-bit post-training quantization (PTQ) to INT8 or INT4 formats introduces severe rounding noise that degrades performance. Conversely, merging pre-quantized experts breaks linear mode connectivity due to disjoint quantization grids. 

Q-Merge addresses this deployment bottleneck by optimizing continuous, layer-wise merging coefficients ($\Lambda$) directly under the non-differentiable quantization rounding and clipping operators. Adaptation is performed at test-time by minimizing the joint prediction entropy on a tiny, unlabeled calibration stream (64 images in total). The paper formalizes two optimization strategies: zero-order stochastic mutation (1+1 ES) and first-order gradient descent (Adam GD) using the Straight-Through Estimator (STE) to propagate gradients back to the continuous coefficients. 

Through extensive evaluation on a pre-trained ViT-Tiny backbone across a four-task classification benchmark (MNIST, FashionMNIST, CIFAR-10, SVHN), the authors demonstrate that 8-bit Q-Merge with STE achieves **74.30%** average accuracy (surpassing the unquantized uniform FP16 baseline of 71.88%). Under aggressive 4-bit per-channel weight quantization, Q-Merge with STE achieves **63.36%** average accuracy, outperforming the naive post-merge baseline (56.66%) by **6.70%** absolute. Extensive systems analyses in the Appendix confirm Q-Merge's robustness to scale factor discretization, activation quantization (W8A8/W4A4), and non-stationary stream imbalance.

---

## Strengths and Weaknesses

### Strengths
1. **Elegant Conceptual Simplicity:** The proposed method is remarkably simple, clean, and direct. Rather than introducing highly complex custom architectures, dynamic routing networks, or convoluted multi-stage training schedules that would complicate on-device adaptation, Q-Merge parameterizes the search space using only a tiny layer-wise blending tensor ($14 \times 4 = 56$ parameters for ViT-Tiny, or $32 \times K$ for LLaMA-7B). This minimalism makes the adaptation extremely fast, data-efficient, and easy to integrate into existing edge deployment compilers.
2. **Exceptional Scientific Honesty and Transparency:** The authors deserve high praise for their rigorous deconstruction of the optimizer confounding factor in Section 4.4.2. They explicitly isolate the effect of the optimizer by introducing an Adam-optimized unquantized baseline. This honest analysis clarifies that the apparent "regularization" of Q-Merge over standard AdaMerging is primarily due to the transition from a zero-order optimizer (1+1 ES) to a first-order optimizer (Adam GD), rather than an inherent benefit of quantization noise. This level of transparency is rare and represents outstanding scientific integrity.
3. **Zero Inference Latency or Storage Overhead:** From a pragmatic systems engineering perspective, Q-Merge is ideal. Once the blending coefficients are optimized on a small calibration stream, they are locked, and the joint weights are compiled into a static, fully quantized low-bit integer representation (e.g., INT8 or INT4). The blending coefficients are discarded, resulting in absolute zero runtime latency or storage overhead during edge inference.
4. **Mathematical and Explanatory Rigor:** The mathematical formulation of the dual-path gradient flow through the non-differentiable rounding operator and the dynamically computed channel-wise scale factors (Equation 15) is comprehensive and clear. This derivation resolves any potential ambiguity regarding automatic differentiation behavior over dynamic PTQ scales.
5. **Statistical Soundness:** The core quantitative results are evaluated and reported across **three independent random seeds** with standard deviations included, confirming the reliability and significance of the performance gains.
6. **Comprehensiveness of Systems-Level Analyses:** The authors provide a highly thorough set of ablation and sensitivity analyses in the Appendix, directly addressing practical edge implementation details. These include:
   - Defining a highly complemental 2-stage pipeline integrating Q-Merge with advanced local PTQ (AdaRound), achieving state-of-the-art results (64.46% in 4-bit).
   - Simulating scale factor precision discretization ($N_{\text{fraction}} \in \{8, 16, 32\}$-bit) to prove feasibility on integer-only MCUs.
   - Deriving and evaluating a dual-STE joint weight-activation quantization scheme (W8A8 and W4A4 configurations).
   - Proposing and validating low-overhead online stream task-balancing heuristics (Confidence-Based FIFO Stratification) to protect against non-stationary stream skew.

### Weaknesses
1. **Toy-Scale Evaluation:** While understandable for a detailed multi-seed and multi-baseline study, the empirical validation is conducted on a toy-scale backbone (ViT-Tiny, 5.7M parameters) and task experts trained under low-data regimes (512 images per task), leading to low-capacity models with minimal parameter drift. Under this setup, the SVHN unmerged expert achieves a low performance of 41.34% because of few-shot fine-tuning.
   - *Mitigation:* The authors explicitly acknowledge this limitation in Section 5.2. They also provide a thorough theoretical scaling analysis in Appendix B.4 showing that as backbones scale to LLaMA-7B, the search space size remains virtually static ($56 \to 128$ parameters), achieving up to a $5.23 \times 10^7 \times$ compression factor. Evaluating on fully converged experts and multi-billion parameter autoregressive language models (e.g., LLaMA, Mistral) on diverse text generation/reasoning tasks (such as MMLU or GSM8K) remains a crucial future validation step.

---

## Detailed Evaluation of Dimensions

### Soundness: Excellent
The paper is technically flawless and highly rigorous. Every core equation is mathematically justified, and the authors provide an exceptionally complete derivation of the STE-based dual-path gradient flow through both the weight coordinates and the dynamic scale factors. The experimental design is robust, incorporating multiple baselines, independent seeds with standard deviation, and honest optimizer-controlled comparisons. Potential failure modes (such as trivial class collapse under unsupervised entropy, activation memory overhead, and calibration stream skew) are proactively identified and mitigated with elegant, low-overhead solutions (e.g., layer-wise blending bottlenecks, Forward-Mode AD, and FIFO buffer stratification).

### Presentation: Excellent
The manuscript is beautifully written, extremely clear, and highly structured. The logical flow is seamless, starting with the core dilemma (alignment loss vs. quantization noise), introducing the elegant formulation of Q-Merge, proving the advantages of first-order STE over zero-order mutation, and wrapping up with a very complete set of systems-level discussions in the Appendix. The tables are professional, the figures are high-signal and self-explanatory, and the notations are consistent throughout.

### Significance: Good
The work addresses a highly relevant, real-world edge deployment challenge: how to serve multitasking models under tight on-device memory and storage constraints. By bridging weight-space model merging and post-training network quantization, Q-Merge offers a highly practical, zero-inference-overhead design pattern for edge engineers. While the empirical evaluation is currently restricted to vision backbones of modest scale, the theoretical scaling and complexity analyses confirm that Q-Merge's low-dimensional blending formulation is uniquely suited to scale to large-scale foundation models (such as LLaMA-7B) with massive parameter compression.

### Originality: Excellent
Q-Merge is the first framework to formulate and solve model merging directly under the quantization operator. It successfully adapts model parameters to tolerate discretization noise before rounding occurs (global coordinate alignment), which conceptually differs from and complements traditional PTQ algorithms (which optimize local reconstruction errors of a fixed continuous weight). Furthermore, demonstrating that a simple, first-order gradient-based optimization utilizing the Straight-Through Estimator is highly stable and superior to zero-order mutations in quantized, blended weight spaces is a highly original and valuable finding for the network compression community.

---

## Overall Recommendation

**Rating: 5 (Accept)**

### Justification of the Recommendation
This is an outstanding paper that solves a highly complex deployment constraint (low-bit multitasking) through an **exceptionally elegant, simple, and direct formulation**. It leverages standard, clean mathematical primitives (per-channel Round-to-Nearest, Shannon prediction entropy, and standard backpropagation with the Straight-Through Estimator) to achieve a zero-inference-overhead compressed model.

The authors demonstrate top-tier scientific integrity by meticulously isolating and deconstructing the optimizer confounding factor, providing a perfectly fair baseline comparison. The mathematical completeness of the gradient flow derivations, combined with extensive, statistically rigorous evaluation across multiple seeds and baselines, leaves no doubt about the correctness and viability of the approach.

While the current experiments are limited to a toy-scale vision backbone (ViT-Tiny) with minimal parameter drift, the authors explicitly and transparently acknowledge this limitation and provide comprehensive systems-level analyses and theoretical scaling proofs that make the paper's claims completely watertight. The elegance of the proposed approach, its high practical utility, and its exemplary scientific rigor make this a highly valuable and clear accept.
