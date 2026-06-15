# Peer Review: Q-PolyMerge

## 1. Summary of the Paper
The paper introduces **Q-PolyMerge**, a parameter-efficient framework for quantization-aware test-time model merging on resource-constrained edge devices. Multi-task model merging combines task-specific experts fine-tuned from a shared base model directly in parameter space, but subsequent post-training quantization (PTQ) to low-bit integers (e.g., INT8 or INT4) introduces severe representation and alignment noise. While test-time adaptation (TTA) methods like AdaMerging attempt to resolve this on-device by optimizing layer-wise coefficients on small unlabeled calibration streams, the authors identify the **Overfitting-Optimizer Paradox**: unconstrained high-dimensional optimization (e.g., 56 parameters for a ViT-Tiny) easily overfits to transductive noise on tiny calibration streams (16 images), yielding highly jagged, unstable schedules.

To resolve this, Q-PolyMerge restricts the merging coefficients to a low-dimensional continuous polynomial subspace of normalized layer depth. For a quadratic polynomial ($d=2$), this reduces the optimization space by over 78% (from 56 to 12 parameters). The smooth trajectory acts as a low-pass filter, preventing degenerate entropy collapse. Q-PolyMerge enables both first-order optimization via the Straight-Through Estimator (STE) and a zero-order 1+1 Evolution Strategy (1+1 ES) pathway. By bypassing backpropagation, the zero-order ES pathway eliminates activation caching, reducing the peak volatile memory (SRAM) footprint by >95% (e.g., to 4.05 MB under 4-bit PTQ), making dynamic on-device adaptation memory-viable for microcontrollers.

---

## 2. Strengths and Weaknesses

### Strengths:
1. **Insightful Problem Formulation:** The identification and formal definition of the "Overfitting-Optimizer Paradox" under extremely compact test-time streams (16 images) is a highly pragmatic and valuable contribution. It highlights a critical bottleneck in deploying test-time adaptation algorithms in the wild.
2. **Analytical and Systems Rigor:** The paper is mathematically mature and exceptionally thorough. The step-by-step mathematical derivation of the 158.4 MB activation cache memory footprint (Appendix B.2) is exemplary, as is the detailed, hardware-specific fixed-point execution blueprint (Appendix B.3) for non-linear operators.
3. **Dual-Optimization Coherence:** Providing both a first-order STE gradient descent pathway and a zero-order 1+1 ES pathway is a highly logical and complete design. Bypassing gradient caching to achieve a >95% reduction in volatile memory (SRAM) is a highly compelling systems-level motivation.
4. **Outstanding Presentation and Context:** The manuscript is exceptionally well-written, easy to follow, and beautifully positioned relative to concurrent 2025/2026 low-bit merging works (such as TVQ, E-PMQ, and 1bit-Merging).

### Weaknesses:
1. **Toy-Scale Evaluation and Upsampled Datasets:** The empirical evaluation is conducted exclusively on a tiny **ViT-Tiny** (5.7M parameters) backbone across classic toy datasets (**MNIST, FashionMNIST, CIFAR-10, SVHN**). Because ViT-Tiny requires 224x224 inputs, these small 28x28 and 32x32 images are artificially upsampled by a factor of 7x, which does not represent the visual complexity, clutter, or scale of actual edge environments. Modern model merging and TTA literature operates on large-scale foundation models (such as CLIP-ViT-B/16 or LLaMA-7B/70B) on challenging downstream benchmarks (MMLU, GSM8k, ImageNet-1K).
2. **Omission of the Critical "Task-Wise" Baseline:** The paper compares unconstrained layer-wise optimization (56 parameters) against continuous polynomial optimization (12 parameters). However, it completely omits the most obvious and simpler baseline: **Task-Wise Q-Merge** (optimizing a single scale coefficient per task uniform across all layers, requiring only $K=4$ parameters). A task-wise baseline represents a perfectly smooth/flat trajectory across layers, is even more parameter-efficient, and acts as a perfect regularizer. Without this comparison, it is unproven whether the layer-wise polynomial variation provides any performance advantage over a simple uniform scaling baseline.
3. **Underfitting due to Global Smoothness Assumption:** Forcing heterogeneous layer blocks (attention projections vs. FFN MLP expansions) to lie on a smooth, slowly-varying global quadratic curve is a strong inductive bias that can lead to severe underfitting. This is highly visible in Table 1 (8-bit Adam STE): on MNIST, Q-PolyMerge degrades performance by **-12.77%** compared to unconstrained Q-Merge (45.93% vs 58.70%); on FashionMNIST, it drops by **-9.42%** (59.93% vs 69.35%); and on CIFAR-10, it drops by **-8.07%** (66.00% vs 74.07%). Q-PolyMerge only matches the unconstrained average because of a massive SVHN spike (+29%). This indicates that the polynomial prior acts as a severe over-regularizer that restricts the model's capacity to resolve task-specific layer interference.
4. **Massive Statistical Volatility across Seeds:** Under the zero-order ES pathway, individual task standard deviations across the 3 seeds are extraordinarily high (e.g., **18.95%** on CIFAR-10 and **11.75%** on MNIST under 8-bit ES). Such high variance indicates that the random mutations are highly sensitive to the specific 16 images sampled, making the statistical superiority claims weak.
5. **Lack of Physical Hardware-in-the-Loop Validation:** Despite claiming to enable "on-device adaptation on physical edge microcontrollers for the first time," there is **zero physical, on-chip evaluation** on actual edge processors (e.g., ARM Cortex-M7 or RISC-V GAP8). Relying purely on theoretical mathematical derivations of activation caching and modeled latency/energy profiles does not satisfy the empirical standard required to claim physical, on-device viability.

---

## 3. Soundness: Fair
*Justification:* The analytical and mathematical foundations of the paper are highly rigorous, and the systems-level memory derivations are precise and transparent. However, the empirical results reveal that the core global smoothness assumption introduces severe underfitting, causing large performance drops on three of the four tasks under gradient descent. Furthermore, the omission of the critical Task-Wise baseline leaves the core hypothesis unvalidated, and the zero-order pathway exhibits extreme fragility in 4-bit landscapes.

---

## 4. Presentation: Excellent
*Justification:* The submission is exceptionally clear, logical, and structured. The extensive mathematical derivations in the appendices and the detailed comparative discussions of concurrent works are mature, highly readable, and thoroughly documented.

---

## 5. Significance: Good
*Justification:* Consolidating multiple task-specific experts into a single quantized model that can adapt on-the-fly under strict memory budgets is a highly important and relevant problem for edge AI. Bypassing gradient memory to achieve a >95% peak SRAM reduction represents a highly practical and significant contribution. However, the significance of the findings is currently limited by the small-scale, toy-dataset vision setup.

---

## 6. Originality: Good
*Justification:* The combination of a continuous polynomial constraint with quantization-aware test-time model merging is an elegant, creative, and highly practical combination of existing mathematical and optimization tools. It provides a unique software prior designed specifically to tackle the Overfitting-Optimizer Paradox under extreme resource scarcity.

---

## 7. Overall Recommendation: 3 (Weak Reject)
*Justification:* While the paper exhibits excellent presentation, detailed systems-level motivation, and clear analytical rigor, its empirical foundation has several critical gaps that must be addressed. An empirically focused review reveals that the core continuous polynomial trajectory hypothesis is not yet fully validated due to the lack of a simple "Task-Wise" baseline, and its performance suffers from severe underfitting on less sensitive tasks. Furthermore, the claim of physical on-device viability is not supported by physical, hardware-in-the-loop measurements on actual edge silicon. 

To raise this submission to a tier-1 conference standard, the authors are encouraged to:
1. **Incorporate the Task-Wise Baseline:** Evaluate an optimized Task-Wise Q-Merge baseline (4 parameters, uniform across all layers) to demonstrate whether the layer-wise polynomial variation actually provides a statistical performance benefit.
2. **Scale Up the Evaluation:** Move beyond toy 28x28 grayscale upsampled vision datasets to larger, realistic benchmarks (such as CLIP-ViT-B/16 on VTAB or LLaMA-7B on language tasks).
3. **Conduct Hardware-in-the-Loop Measurements:** Replace the "modeled" latency and energy metrics with physical, on-chip measurements taken on actual physical edge silicon (e.g., ARM Cortex-M7 or RISC-V GAP8) to validate their on-device edge viability claims.
4. **Stabilize Zero-Order Search:** Explore and evaluate more stable derivative-free optimization techniques (e.g., CMA-ES or historical momentum filtering) to mitigate the massive variance and fragility of 1+1 ES in low-bit landscapes.
