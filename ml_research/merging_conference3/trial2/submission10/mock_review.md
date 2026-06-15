# Official Peer Review Form

## Summary of the Paper
This paper presents a rigorous and highly insightful deconstructive study of **AdaMerging** (Yang et al., 2024), a state-of-the-art unsupervised test-time model merging framework. Adaptive model merging combines task-specific expert neural networks into a single multi-task model without retraining on original data by learning scaling coefficients. AdaMerging minimizes prediction entropy on a small, unlabeled calibration split to find these scales.

Through a highly systematic, skeptical, and minimalist lens, this work exposes two major optimization anomalies:
1. **The Overfitting-Optimizer Paradox**: Optimizing hundreds of layer-specific scaling coefficients ($\Lambda \in \mathbb{R}^{L \times T}$) on tiny calibration batches leads to transductive test-time overfitting. By introducing two diagnostic control treatments—*Intra-Task Layer Shuffling* and *Spatial Averaging*—the authors demonstrate that while the learned parameters capture a functional representational routing specialized to the network's hierarchy (explaining why shuffling collapses average accuracy from $88.05\%$ to $78.61\%$), they also capture high-frequency overfitting noise. A simple post-hoc *Spatial Averaging* (taking the flat mean per task, reducing parameter count by 99.6%) acts as an elegant low-pass filter, smoothing away the overfitting component and achieving $84.96\%$ accuracy, outperforming static Task Arithmetic ($84.64\%$).
2. **The Spatial Averaging Paradox**: While indirect optimization (optimizing high-dimensional parameters and averaging them post-hoc) succeeds, direct optimization of those same flat task-wise scales (Task-wise AdaMerging) fails spectacularly, degrading accuracy to $81.19\%$ (below uniform baseline). The authors explain this through **multi-task gradient imbalance** on uncalibrated prediction entropy: simple tasks (MNIST, FashionMNIST) have sharp distributions and dominate the joint entropy gradient, driving the optimizer to scale up their coefficients. Under a low-dimensional bottleneck, this joint scaling creates severe parameter interference in early shared projection layers, collapsing representation structures for the harder tasks (CIFAR-10, SVHN). Under high-dimensional optimization, local layer degrees of freedom allow the optimizer to minimize entropy without resorting to global scaling trade-offs, which post-hoc spatial averaging then regularizes.

The authors evaluate their claims on a 4-task vision benchmark across 3 seeds using CLIP ViT-B/32, scaling evaluations to the **full standard test splits (56,032 images in total)**. They also analyze internal representational manifolds using layer-by-layer Linear CKA representational similarity across all 12 blocks, showing that early layers remain aligned while late task-specific layers specialize. Finally, they evaluate a proposed *Calibrated Prediction Entropy* remedy and find that it still fails, demonstrating that the pathology is a fundamental structural weight-space bottleneck limitation.

---

## Strengths and Weaknesses

### Strengths:
1. **Outstanding Scientific Narrative**: The paper is exceptionally well-written, reading like a detective story. It systematically deconstructs a complex, SOTA adaptive model merging framework to isolate its true causal drivers, adhering closely to the principle of Occam's razor.
2. **Empirical Rigor**: Evaluating all experts, merged models, and baselines on the **full test splits (56,032 images in total)** instead of small subsets completely eliminates data selection bias and provides watertight statistical precision. Reporting standard deviations across three seeds further enhances credibility.
3. **Innovative Diagnostic Controls**: The introduction of *Intra-Task Layer Shuffling* and *Spatial Averaging* is highly creative. These simple, elegant control treatments allow the authors to dissect and verify the structural specialization and transductive overfitting of coefficients without introducing convoluted new pipelines.
4. **Comprehensive Baseline Comparisons**: The authors integrate SOTA static baselines (TIES-Merging, DARE-Merging) and perform a complete grid sweep over standard Task Arithmetic scaling factors ($\lambda \in \{0.1, 0.2, 0.3, 0.4, 0.5\}$). This thoroughly contextualizes post-hoc Spatial Averaging as a highly practical, label-free, and self-regularizing scaling estimator.
5. **Excellent Representational Visualizations**: The inclusion of layer-by-layer Linear CKA representational similarity curves across all 12 blocks of the Transformer backbone visually maps and substantiates the hierarchical routing hypothesis (how early layers remain aligned with the target expert while late layers specialize).
6. **Scientific Honesty**: The paper embraces optimization failures (e.g., the failure of direct task-wise optimization and the failure of the calibrated remedy) as central, high-signal findings that build a deeper, more complete explanation of weight-space optimization limits.

### Weaknesses:
The paper is exceptionally polished and theoretically rigorous, meaning that its weaknesses are extremely minor:
1. **Qualitative Context for SVHN Variance**: SVHN exhibits significantly higher standard deviations ($\pm 5.97\%$ to $\pm 7.31\%$) compared to other datasets in Table 1. While the authors mention that this is due to data selection and calibration variance, a brief qualitative note explaining SVHN's domain complexity (varying street-view fonts, illumination, background clutter) would provide better grounding.
2. **Generative LLM Perplexity Scaling roadmap**: The future work section discusses scaling to LLMs using token-level perplexity. A slightly more explicit roadmap regarding how token-level gradient imbalance could cause similar pathologies (e.g., highly repetitive boilerplate language dominating complex reasoning gradients) would strengthen the proposal.
3. **Synergy with Static Weight-Space Sparsification**: While the authors contrast their method with TIES-Merging and DARE-Merging, they do not explicitly highlight the potential synergy of combining post-hoc spatial averaging on top of sign-resolved or pruned base task vectors. This is a promising avenue that should be explicitly proposed.

---

## Soundness
**Rating: Excellent**

**Justification:**
The paper is technically flawless and methodologically sound. The authors are completely transparent about their evaluation assumptions (such as Oracle Routing for visual backbones), and they meticulously verify their hypotheses using rigorous empirical diagnostic controls. The explanation of the Spatial Averaging Paradox through uncalibrated prediction entropy and multi-task gradient imbalance is mathematically elegant and fully supported by individual task accuracy collapses (e.g., CIFAR-10 and SVHN collapsing while MNIST remains high). The fact that the Calibrated Prediction Entropy remedy still fails proves that the authors have successfully identified a fundamental structural weight-space bottleneck issue rather than an optimization or hyperparameter artifact. The entire lifecycle—implementation, multi-seed testing, full-scale test split evaluation, and representation manifold checks via Linear CKA—is executed with maximum academic rigor.

---

## Presentation
**Rating: Excellent**

**Justification:**
The submission is beautifully written, clearly structured, and easy to follow. The mathematical notation is precise and consistent throughout the paper. Section 4.2.1’s cross-referencing is flawless, and all tables and figures are integrated seamlessly. The newly added Figure 4 (layer-by-layer CKA plot) is a masterpiece, clearly illustrating representational dynamics through the network's deep architecture. The authors have optimized equations, tables, and layouts to fit perfectly within the standard ICML template, with zero horizontal or vertical overflows.

---

## Significance
**Rating: Excellent**

**Justification:**
In a field that is currently flooded with increasingly complex, parameter-rich adaptive model merging schemes, this paper stands out as a highly significant course-correction. By deconstructing SOTA AdaMerging and exposing the fundamental limitations of prediction entropy minimization under low-dimensional constraints, it prevents future researchers from falling into pathological optimization traps. It highlights the immense practical value of post-hoc Spatial Averaging as an elegant, parameter-free, and self-regularizing scaling estimator, representing an outstanding contribution that is likely to influence future weight-space model combination research.

---

## Originality
**Rating: Excellent**

**Justification:**
While the paper is a deconstructive study, its originality is exceptional. It provides deep, novel insights into why adaptive weight-space optimization works at a high-dimensional level (layer-wise routing) and why it fails at a low-dimensional level (multi-task gradient imbalance). The discovery of the Spatial Averaging Paradox and the systematic deconstruction of the Overfitting-Optimizer Paradox via layer shuffling are highly creative and novel contributions that clearly distinguish this work from previous workshop-level critiques.

---

## Overall Recommendation

**Rating: 6: Strong Accept**

**Detailed Feedback and Minor Constructive Suggestions:**
This is an outstanding, publication-ready manuscript of the highest caliber. It is technically sound, exceptionally well-written, empirically watertight, and provides a significant contribution to the model-merging literature. I recommend a Strong Accept (6) and offer a few minor, constructive suggestions to elevate the paper to absolute professional perfection:

1. **Elaborate on SVHN's Qualitative Complexity**: In Section 4.1 or 4.2.2, add a brief, explicit sentence detailing the qualitative domain difficulty of SVHN (varying fonts, illumination, background clutter, distracting adjacent digits) to explain why SVHN experiences much higher data selection and calibration variance compared to highly homogeneous datasets like MNIST.
2. **Explicitly Propose Synergy with Weight-Space Sparsification**: In Section 4.2.4 or the Conclusion, explicitly propose combining post-hoc Spatial Averaging on top of pruned or sign-resolved base task vectors (such as TIES-Merging) as an exciting synergy, demonstrating how adaptive scale regularization can build upon static sparsification.
3. **Detail LLM Perplexity Gradient Imbalances**: In Section 5.1 (Future Work), expand the LLM scaling discussion to explicitly note that highly repetitive or easily predictable tokens (such as boilerplate templates or highly frequent function words) could dominate token-level perplexity gradients over rare, complex reasoning tokens, mirroring the easy vs. hard task gradient imbalance observed in vision models.
