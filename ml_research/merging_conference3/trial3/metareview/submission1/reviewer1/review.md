# Peer Review of Conference Submission

## Summary of the Paper
This paper presents a highly rigorous, independent methodological deconstruction and robust evaluation of Quantization-Aware Model Merging (such as Q-Merge). Model merging combines the parameter weights of task-specific expert neural networks starting from a shared pre-trained initialization. However, post-training quantization (PTQ) to low bit-widths (e.g., 4-bit) introduces discretization noise that can catastrophically damage merged performance. To address this, current paradigms optimize layer-wise merging coefficients directly under simulated quantization using prediction entropy minimization on small calibration streams.

This paper approaches these claims with deep scientific skepticism, exposing critical, previously unstudied vulnerabilities in quantization-aware weight-space fusion. The authors design a standardized, highly controlled evaluation setup using a `ViT-Tiny` backbone across four distinct tasks (MNIST, FashionMNIST, CIFAR-10, SVHN). Through a **Multi-Axial Robustness Audit**, they systematically evaluate the paradigm along four axes:
1. **Calibration Size Sweep:** Reveals that direct low-bit optimization under quantization constraints is not actually necessary, as unquantized optimization in FP16 followed by post-hoc target quantization (Quantized AdaMerging) consistently and substantially outperforms Q-Merge ($30.00\%$ vs $26.25\%$).
2. **Cross-Schema Generalization Matrix:** Exposes catastrophic **Quantization-Operator Overfitting** (the "Cross-Schema Generalization Gap"), where continuous coefficients optimized under one schema (e.g., asymmetric per-channel) drop by up to $20.37\%$ absolute accuracy (collapsing to near random guessing) when run under coarsened target hardware-relevant schemas (e.g., symmetric per-tensor).
3. **Regularization and Optimizer Choice:** Compares Straight-Through Estimation (STE) with a derivative-free **1+1 Evolution Strategy (1+1 ES)**. Shows that while 1+1 ES achieves superior optimization on the source schema ($20.75\%$ vs $17.88\%$), its powerful search capabilities overfit the simulated rounding thresholds even more severely, leading to worse cross-schema generalization gaps.
4. **Stream Distortion and Skew Robustness:** Demonstrates that unsupervised entropy minimization is highly fragile under class imbalance, collapsing decision boundaries under Gini skew.

In addition, the authors introduce a **Supervised Calibration Baseline** to decouple data-scarcity from objective fragility, and perform PoC audits of architectural variations (CNN/ResNet-18) and low-rank SVD projections (critically evaluated as a "Low-Capacity Generalization Illusion"). Finally, they formulate four crucial methodological recommendations for robust, deployment-realistic weight-space fusion.

---

## Strengths and Weaknesses

### Strengths
- **Paradigm-Shifting Conceptual Contributions:** This paper does not simply propose another minor performance tweak. Instead, it challenges and systematically deconstructs the foundational assumptions of an entire emerging research paradigm. Exposing **Quantization-Operator Overfitting** (the "Cross-Schema Generalization Gap") is a major conceptual leap that directly addresses the gap between deep learning simulation and heterogeneous physical hardware accelerators (e.g., Edge TPUs mandating symmetric per-tensor arithmetic vs high-performance DSPs supporting asymmetric per-channel representations).
- **Refuting the Necessity of Quantization-Aware Search:** Proving that unquantized FP16 continuous search followed by post-hoc target quantization (Quantized AdaMerging) consistently outperforms direct STE-based low-bit optimization ($30.00\%$ vs $26.25\%$) is an exceptionally strong, surprising, and ambitious finding that fundamentally challenges current research trends.
- **Outstanding Mathematical and Empirical Rigor:** The paper's mathematical formalization is exemplary. The PTQ equations are highly precise, and the authors go to great lengths to detail the asymmetric gradient flow of scale $s$ and zero-point $z$ parameters. The empirical design is highly controlled, featuring informative baselines (FP16 Task Arithmetic, Naive M-then-Q, Quantized AdaMerging, and a Supervised Calibration Baseline), and uses rigorous ablation sweeps (varying learning rates, dynamic initialization, spatial regularizer scaling) to rule out simple tuning issues.
- **Exceptional Scientific Transparency:** The authors exhibit outstanding academic honesty in analyzing potential confounders. For example, they identify the flat generalization gap of the low-rank SVD projection as a **"Low-Capacity Generalization Illusion"** caused by severe representational capacity degradation, rather than a robust alignment of weights.
- **Actionable Algorithmic Roadmap:** The proposed methodological mandates are highly constructive, detailed, and ambitious. Instead of just pointing out failures, the authors outline concrete algorithmic solutions (e.g., confidence-thresholded pseudo-labeling, hybrid STE/ES local search pipelines, and pre-quantization landscape smoothing via TIES/DARE).

### Weaknesses
- **Empirical Scale-Up Limitations:** The empirical evaluation is primarily conducted on a lightweight model (`ViT-Tiny`). While the authors provide a highly compelling scientific defense (and mathematical projections explaining why the Cross-Schema Generalization Gap is expected to *expand* with model scale), physical verification on a larger model (e.g., a 1B/3B language or vision-language model) would further enhance the paper's completeness.
- **Joint Weight-Activation Quantization:** The audit is limited to weight-only quantization (W4). Real-world edge deployment typically mandates joint weight-activation quantization (e.g., W4A8 or W4A4) where dynamic activation outliers add further noise. Jointly evaluating activation quantization would make the audit even more comprehensive.

---

## Soundness
**Rating:** Excellent

The paper is technically flawless and mathematically precise. Every component of the model merging process, quantization operators, and optimization backends is rigorously formalized. The authors are exceptionally careful and honest in evaluating their work, providing thorough baseline comparisons, and executing empirical PoC extensions on alternative architectures (CNNs) and low-rank subspaces. The reporting of mean and standard deviation over multiple random seeds ensures the empirical claims are highly reliable.

---

## Presentation
**Rating:** Excellent

The submission is exceptionally well-written, clearly structured, and easy to follow. The authors position their work beautifully in the context of recent weight-space fusion and post-training quantization literature. Every table and figure is highly informative and directly aligns with the narrative.

---

## Significance
**Rating:** Excellent

This paper addresses an extremely important and relevant problem: the deployment-feasibility of merged models on edge hardware. It reveals a critical deployment-limiting bottleneck (quantization-operator overfitting) that has been completely ignored by prior works. By exposing this gap and providing a concrete mathematical framework and detailed algorithmic roadmap, the paper has the potential to influence future weight-space fusion and edge-computing research, prompting the community to pivot toward cross-operator validation.

---

## Originality
**Rating:** Excellent

The originality of this paper is outstanding. It departs from the standard "state-of-the-art" chasing narrative and instead takes a bold, skeptical, and conceptual step backward. The introduction of the Cross-Schema Generalization Matrix and the deconstruction of the necessity of quantization-aware search represent highly original conceptual leaps that will fundamentally alter how researchers approach low-bit model ensembling.

---

## Overall Recommendation and Justification

**Recommendation:** 6: Strong Accept

### Justification
This is an exceptional, technically flawless, and paradigm-shifting paper. The authors have approached the quantization-aware model merging paradigm with deep methodological skepticism, exposing a critical, previously unstudied vulnerability: **Quantization-Operator Overfitting**. By demonstrating that continuous coefficients overfit intensely to simulated rounding thresholds and collapse under target hardware mismatches—and by proving that direct quantization-aware search is consistently outperformed by full-precision search followed by post-hoc quantization—this paper refutes foundational premises in the field. 

The paper is mathematically rigorous, empirically robust, and written with outstanding scientific transparency and clarity. Its potential impact on both ML researchers and hardware deployment practitioners is exceptionally high. Therefore, it deserves a Strong Accept.
