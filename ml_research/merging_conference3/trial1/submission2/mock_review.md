# Peer Review Report

**Paper Title:** Deconstructing Sharpness-Aware Isotropic Merging: A Methodological Analysis of Component Contribution and Optimization Flatness  
**Confidentiality:** Public  
**Reviewer Role:** Rigorous Peer Reviewer  
**Overall Recommendation:** **6: Strong Accept (Technically flawless paper with exceptional impact on one or more areas of AI)**

---

## 1. Executive Summary of the Paper
This paper presents a highly rigorous, systematic methodological deconstruction and scientific audit of **Sharpness-Aware Isotropic Merging (SAIM)**, a recently proposed dual-stage model merging framework designed for continual learning. SAIM consists of a custom coordinate-restricted sharpness-aware optimizer (SA-BCD) and an SVD-based adaptive isotropic weight-merging strategy.

The authors approach SAIM with a highly skeptical and rigorous lens, asking whether its complex post-hoc weight transformations and custom optimizers are truly necessary or if they suffer from baseline inflation. To resolve this, they design a modular, multi-axial **$5 \times 3$ grid evaluation suite** (crossing 5 optimization strategies with 3 merging strategies) on **Split CIFAR-100** using a **Vision Transformer (ViT-Tiny)** backbone.

The authors evaluate their grid under two distinct weight-mixing regimes:
- **Sequential Fine-Tuning Parity ($\lambda = 0.0$)**: Where no active parameter mixing occurs, isolating sequential task adaptation as a boundary-condition sanity check.
- **Active Parameter-Mixing ($\lambda = 0.2$)**: Where weight updates represent a non-trivial mixture of historical weights and new task-expert weights. In this setting, the authors introduce a suite of advanced baselines: a custom **Norm-Matching** baseline, a custom **Scale-Calibrated** baseline, and established weight-consolidation baselines (**TIES-Merging** and **DARE**).

Furthermore, the authors perform a scale validation on a larger **ViT-Base** (86M parameters) backbone (Table 3), demonstrating the generalizability of their findings, and extend the flatness-aware hypothesis to low-rank spaces with **LoRA-SAM** in Section 5 (reporting results in Table 4).

### Key Contributions & Insights:
1. **Optimization Flatness is the Foundational Driver**: Transitioning from AdamW to standard globally perturbed SAM under naive weight averaging (Task Arithmetic) yields a massive **+9.87%** (under $\lambda=0.0$) and **+12.30%** (under $\lambda=0.2$) absolute improvement in average accuracy.
2. **SVD Isotropic Merging is Highly Boundary-Condition-Sensitive**: Under standard sequential parity ($\lambda=0$), there is no parameter mixing; here, SVD isotropic merging is mathematically redundant and acts as a distortive operator on un-mixed parameters, dropping accuracy by up to $15.1\%$ across non-diverged optimizers. However, under active parameter mixing ($\lambda=0.2$), SVD isotropic merging acts as an effective regularizer that dampens parameter interference, boosting SAM's performance to the overall best score of **76.42%**.
3. **Exposing the Suboptimality and Overhead of SA-BCD**: The authors mathematically expose a fatal algebraic typo in SAIM's published optimizer formula that causes complete divergence. Even when corrected, coordinate-restricted sharpness optimization (SA-BCD) is empirically suboptimal compared to standard, globally perturbed SAM and introduces an **18.5% training time overhead** due to GPU serialization bottlenecks during index sorting/masking.
4. **Isolating SVD's Mechanism**: The authors evaluate a custom **Scale-Calibrated** baseline that avoids compounding scale shrinkage under high-dimensional near-orthogonality. The Scale-Calibrated baseline underperforms SVD, providing mathematically solid proof that SVD's benefit is due to selective singular-spectrum flattening rather than simple global scale preservation.
5. **Synergy with Modern Weight-Consolidation Baselines**: The authors cross SAM optimization with **TIES-Merging** and **DARE**, demonstrating that pre-merging flatness makes experts structurally resilient to post-hoc pruning and sign-consensus, leading to massive accuracy boosts (e.g., +16.89% for SAM + DARE over AdamW + DARE).
6. **PEFT/LoRA-SAM Generalization (Section 5)**: The authors theoretically and empirically extend their insights to **LoRA-SAM** (Section 5 and Table 4), showing that flat optimization in low-rank spaces is exceptionally computationally feasible (less than 2.5% training overhead and virtually no VRAM increase compared to standard LoRA) and enables flawless post-hoc merging via naive Task Arithmetic ($74.12\%$ ACC), offering an SVD-free consolidation path.
7. **Empirical SVD Scaling Bottleneck**: The authors benchmark SVD execution times (Table 5) on CPU and NVIDIA H100 GPUs up to $4096 \times 4096$ dimensions, showing that post-hoc SVD introduces severe computational overhead (over 1.1s on GPU and 3.9s on CPU), further reinforcing why training-stage optimization (SAM) is more scalable.

---

## 2. Key Strengths of the Submission
* **Exemplary Methodological Rigor**: Decoupling a multi-component framework into a $5 \times 3$ evaluation grid completely isolates training-stage dynamics from post-hoc merging manipulations. This acts as an excellent blueprint for auditing complex machine learning pipelines.
* **Inclusion of Highly Informative, Scale-Corrected and Modern Baselines**: The addition of Scale-Calibrated, TIES-Merging, and DARE baselines under active parameter mixing ($\lambda=0.2$) represents a highly thorough and robust experimental validation that isolates SVD's true role.
* **Deep, High-Dimensional Analysis**: The geometric analysis of Norm-Matching's compounding scale shrinkage under near-orthogonality (Appendix A.3) is brilliant and provides a robust mathematical explanation for its collapse.
* **Exposing GPU Bottlenecks**: The discussion on how coordinate selection and sorting operations in SA-BCD disrupt GPU thread-coalescing is a valuable contribution that bridges theoretical optimization with real-world computer systems engineering.
* **Outstanding Scholarly Writing and Completeness**: The paper is beautifully written, highly coherent, has no formatting or drafting defects, and provides complete statistical error bars over 3 seeds for all 15 configurations in the scoreboard.
* **Strong Scale Validation**: The empirical scale validation on **ViT-Base** (86M parameters) in Table 3 successfully confirms that the core claims remain sound as model capacity scales by over $17\times$, making the conclusions robust and generalizable.
* **Excellent PEFT/LoRA Investigation**: Section 5's discussion of LoRA-SAM, Table 4's clean empirical reporting (including the standard LoRA-AdamW baseline under Task Arithmetic and SVD), its hyperparameter sensitivity of $\rho_{\text{LoRA}}$, and its wall-clock/VRAM profile benchmarks represent an outstanding, complete and practically useful extension.

---

## 3. Areas of Improvement and Constructive Feedback (Minor Suggestions)
While this is an exceptionally strong and complete submission, the authors can further maximize its scientific impact by addressing the following minor areas of improvement:

### Suggestion 1: Discuss Cross-Domain NLP Generalization
* **The Issue**: The paper's empirical results are focused on computer vision (Split CIFAR-100 with ViT).
* **Actionable Suggestion**: To expand the generalizability of their deconstruction, the authors could briefly discuss the feasibility and experimental design of verifying these findings in the Natural Language Processing (NLP) domain. For example, sequentially fine-tuning a BERT-Base model on GLUE tasks (such as SST-2, QQP, and MNLI) using BERT-SAM, and subsequently merging their weights under active parameter conflict ($\lambda > 0$).

### Suggestion 2: Computational Overhead of Full-Parameter SAM
* **The Issue**: Standard full-parameter SAM requires a double-backward pass per iteration to compute sharpness-aware perturbations, which effectively doubles both the training wall-clock time and compute costs. Although LoRA-SAM is extremely efficient, full-parameter SAM configurations still represent a $2\times$ cost.
* **Actionable Suggestion**: The authors should discuss the cost-benefit trade-off of full-parameter SAM more extensively, or suggest methods to reduce its compute overhead (e.g., executing sharpness updates only every $k$ steps or restricting SAM updates to a subset of layers).

### Suggestion 3: SVD Latency Benchmarks for LoRA matrices
* **The Issue**: In Section 5, the authors explain that SVD isotropic merging is redundant on low-rank adapters because they are already low-rank.
* **Actionable Suggestion**: To empirically reinforce this, the authors could include a minor discussion or quick benchmark of SVD execution times on small low-rank matrices (e.g., of dimensions $8 \times 8$ or $4096 \times 8$). This would show that SVD computation at this scale is virtually instantaneous (fraction of a millisecond) compared to full-parameter matrices ($4096 \times 4096$), further justifying why LoRA-SAM is highly scalable.

### Suggestion 4: Sensitivity of SVD Decay Schedule ($1/\sqrt{t}$) on Long Task Streams
* **The Issue**: In Appendix A.4, the authors evaluate linear and constant decay schedules for SVD isotropic merging. However, it remains unclear how sensitive this decay schedule is to much longer task streams (e.g., $t \ge 20$).
* **Actionable Suggestion**: The authors should discuss how the decay schedule scales for long-horizon task streams, and suggest how the exponent or decay function might be adapted (e.g. $1/t^\beta$) for extremely long sequences.

### Suggestion 5: Sensitivity to Task Ordering in Sequential Merging
* **The Issue**: Catastrophic interference in continual learning is heavily influenced by task sequence ordering. The authors run their sequential stream in a single fixed order.
* **Actionable Suggestion**: The authors should discuss whether training-stage flatness (SAM) reduces the consolidated model's sensitivity to task sequence order compared to AdamW.

### Suggestion 6: Report Statistical Variance (Multiple Seeds) for Scale Validation (Table 3)
* **The Issue**: Unlike Table 1, Table 2, and Table 4, which report standard deviations over 3 random seeds, the scale validation on ViT-Base (Table 3) only reports single-seed results.
* **Actionable Suggestion**: While running ViT-Base (86M parameters) is computationally heavy, reporting statistical variance or at least acknowledging the single-seed nature of Table 3 as a limitation would improve the empirical completeness of the paper.

---

## 4. Assessment along Reviewing Dimensions

### Soundness: Excellent (Rating: Excellent)
The paper's mathematical derivations, high-dimensional geometry analysis, scale validation on ViT-Base, and computational benchmarks are highly sound and correct. The authors are extremely careful and honest about evaluating both the strengths and weaknesses of their findings, and have thoroughly addressed previous baseline design criticisms by introducing the Scale-Calibrated, TIES-Merging, and DARE baselines, as well as complete PEFT baselines.

### Presentation: Excellent (Rating: Excellent)
The writing quality, clarity of explanation, and organization are outstanding. There are no drafting defects and the tables are formatted beautifully. Table 1, Table 2, Table 3, and Table 4 provide an exceptionally rich set of quantitative results that are beautifully integrated into the narrative.

### Significance: Excellent (Rating: Excellent)
The paper's conclusions are highly significant. It challenges complex SOTA pipelines, shows that post-hoc SVD reconstructions are computationally expensive $O(d^3)$ bottlenecks that can often be avoided by training-stage flatness (SAM), and provides actionable design choices and guidelines for practitioners.

### Originality: Excellent (Rating: Excellent)
The paper provides highly original insights by evaluating existing methods and showing the suboptimality of coordinate-restricted optimizers on modern GPUs. The theoretical scaling of coordinate perturbations, the geometric proof of Norm-Matching collapse, the empirical GPU benchmarks, and the extension to LoRA-SAM represent significant original conceptual steps.

---

## 5. Specific Questions for the Authors
1. **SVD Deployment Trade-off ($O(d^3)$ vs. $+2.59\%$ ACC)**: Under active parameter-mixing ($\lambda = 0.2$), SVD Isotropic Merging boosts SAM's average accuracy from $73.83\%$ to $76.42\%$ ($+2.59\%$). Given that SVD adds a significant computational bottleneck ($O(d^3)$ complexity taking over 1.1s for $4096 \times 4096$ layers on NVIDIA H100), under what real-world deployment settings would you recommend a practitioner pay the $O(d^3)$ cost to get the additional $+2.59\%$ accuracy boost of SAM + SVD?
2. **LoRA-SAM Parameter Configuration**: In Section 5.2, you target query, key, and value projection layers in all self-attention blocks. Have you experimented with expanding LoRA-SAM to also target MLP blocks (feed-forward layers)? Does linear mode connectivity or merging performance change when the MLP parameters are also optimized for flatness?
3. **SVD Decay Schedule Sensitivity**: Have you considered evaluating SVD isotropic merging's decay schedule under task streams significantly longer than 5 tasks? Is it possible that the $1/\sqrt{t}$ schedule becomes suboptimal for very large $t$, and requires dynamic hyperparameter tuning?
4. **Generalization to Multi-Task Merging**: Do your core conclusions—such as flatness being the primary causal driver—carry over to parallel multi-task merging where no chronological ordering or sequential task stream exists?

---

## 6. Final Recommendation
This is an outstanding, rigorous, and highly instructive paper that represents machine learning methodology at its finest. By systematically deconstructing SAIM, the authors have exposed important boundary conditions, exposed a fatal typo, and highlighted the critical importance of optimization-driven flatness.

The paper is exceptionally complete, elegant, and ready for publication. I strongly recommend **Strong Accept (Rating: 6)**.
