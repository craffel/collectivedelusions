# Peer Review

## 1. Summary of the Submission
This paper addresses the challenge of active test-time model merging (parameter-space model fusion), specifically focusing on the **Overfitting-Optimizer Paradox** that plagues active adaptation methods like AdaMerging. During test-time adaptation (TTA), optimizing layer-wise merging coefficients on small, unlabeled test streams by minimizing prediction entropy often leads to transductive overfitting to local stream noise, resulting in multi-task representational decay. 

To mitigate this, the authors propose **Pruned Gradient Merging (PG-Merge)**. Guided by Occam's razor, PG-Merge applies a dynamic, non-parametric sparse gradient mask to the raw coefficient gradients, restricting active parameter updates to only the top-$p\%$ (e.g., $5\%$) most sensitive layer-wise coefficients while keeping the other parameters strictly frozen. To prevent adaptive optimizers like Adam from updating masked parameters via historical momentum, the authors apply a strict post-update parameter projection. They also discuss pairing PG-Merge with standard, momentum-free SGD to bypass momentum decay issues. 

Evaluated on a compact Vision Transformer (`vit_tiny_patch16_224`) across four image classification tasks (MNIST, FashionMNIST, CIFAR-10, SVHN), PG-Merge ($p=0.05$) improves the average joint accuracy to $62.70\%$, outperforming unconstrained AdaMerging ($61.08\%$) and matching or exceeding SOTA complex regularizers like RegCalMerge ($62.35\%$).

---

## 2. Strengths of the Submission
1. **Conceptual Simplicity and Philosophical Clarity:** The paper's core hypothesis—that limiting optimization degrees of freedom via gradient sparsity is sufficient to stabilize test-time merging—is elegant and conceptually straightforward. It exposes the potential redundancy of increasingly Byzantine and hyperparameter-heavy spatial regularizers.
2. **Systematic Ablation and Trajectory Analyses:** The ablation study over the sparsity ratio $p$ (Table 2) is very thorough and clearly identifies a "sparsity sweet spot" at $p=0.05$. The trajectory plots (Figure 3) provide excellent empirical evidence demonstrating that PG-Merge successfully decouples prediction entropy minimization from joint accuracy degradation, unlike unconstrained AdaMerging.
3. **Clear Mathematical Formulation:** The dynamic masking, sorting, thresholding, and post-update parameter projection steps (Equations 8-13) are clearly and precisely formulated.
4. **Insightful Optimizer State Analysis:** Appendix A contains a highly valuable theoretical analysis of the "internal state mismatch" and momentum decay of adaptive optimizers like Adam when paired with sparse coordinate masking.

---

## 3. Weaknesses of the Submission

### A. Scholarly Rigor and Literature Positioning (Major Concern)
A foundational requirement of any peer-reviewed scientific publication is that it must accurately describe the landscape of the field, properly attribute established concepts, and cleanly manage its bibliography. This submission falls short in several key areas:
1. **Mischaracterization and Non-Citation of MECTA:** In Section 2.4, the authors discuss MECTA, claiming that:
   > *"However, prior gradient selection methods have primarily focused on reducing backpropagation memory or speeding up training in standard supervised regimes."*
   
   This statement is a major academic error. MECTA stands for **Memory-Economic Continual Test-Time Model Adaptation** (ICLR 2023). It is explicitly a **test-time adaptation (TTA)** method, not a supervised fine-tuning or training-speedup method. 
   Furthermore, while the authors discuss MECTA in the text, they **fail to include a formal citation tag** (e.g., `\cite{...}`) and there is **no corresponding bibtex entry for MECTA in `references.bib`**.
2. **Omission of EATA:** The paper presents the philosophy of using parameter/coordinate sparsity to stabilize test-time adaptation as a novel discovery of this work. However, **EATA (Efficient Test-time Adaptation, ICML 2022)** pioneered exactly this concept, proposing to identify and update only a sparse subset of highly important parameters to stabilize the TTA loop and prevent transductive collapse. The complete omission of EATA represents a significant failure to contextualize the work within existing TTA literature.
3. **Sloppy Bibliography Management ("Ghost" Citations):** Out of more than 50 bibtex entries in `references.bib`, only 13 are actually cited in the body of the paper. Over 35 entries are "ghost" citations (e.g., `gu2024advancing`, `zaken2022bitfit`, dataset papers) that are never referenced. Conversely, other methods discussed in the text (such as `QWS-Merge` and `MECTA`) are completely missing from the bibliography. This suggests a careless compilation of references rather than a rigorous scholarly effort.

### B. Methodological and Technical Contradictions (Major Concern)
1. **The "Training-Free" Misnomer:** The paper repeatedly describes PG-Merge as a "training-free" framework. This is semantically misleading. True training-free merging methods (like Task Arithmetic, Model Soups, and TIES-Merging) combine parameters algebraically in a single offline step with **zero** optimization overhead. PG-Merge requires running **100 gradient steps** of backpropagation usingprediction entropy on an unlabeled test-time stream. Labeling an active gradient-descent adaptation loop as "training-free" is an unfair and inaccurate semantic stretch.
2. **The Adam vs. SGD Contradiction:** In Appendix A, the authors theoretically analyze the momentum decay and state mismatch of adaptive optimizers like Adam when paired with coordinate masking, and strongly advocate for pairing PG-Merge with **standard SGD without momentum** for a clean, mathematically self-consistent, and projection-free implementation. 
   However, in Section 4.1, the entire empirical evaluation is conducted using **Adam**. The scoreboard in Table 1 and the ablation study in Table 2 are completely generated using Adam. The authors provide **no empirical results** for the SGD-based PG-Merge that they theoretically advocate. It is highly inconsistent to advocate for a simple SGD pipeline in the appendix while evaluating the paper solely on a flawed, projection-reliant Adam setup.
3. **Under-Stated Hyperparameter Sensitivity:** The authors claim PG-Merge is "non-parametric" and criticize prior methods for delicate tuning. However, PG-Merge's performance is highly dependent on its own key hyperparameter: the sparsity ratio $p$. As shown in Table 2, joint average accuracy drops from $62.70\%$ ($p=0.05$) to $61.33\%$ ($p=0.30$), representing a highly sensitive performance drop over a narrow parameter range.

### C. Scale and Generalizability Concerns (Major Concern)
1. **Toy Experimental Setup:** The entire empirical validation is performed using a highly restricted, low-capacity setup:
   - **Backbone Model:** A compact `vit_tiny_patch16_224` containing only $5.7$M parameters.
   - **Datasets:** Simple, low-resolution datasets (MNIST, FashionMNIST, CIFAR-10, SVHN).
   - **Data Constraints:** Experts are trained on a tiny subset of only **1,024 images** per task, and adapted on a calibration stream of only **16 samples per task** (64 images).
   
   Because model behaviors, representation dynamics, and parameter conflicts scale non-linearly, findings from a $5.7$M parameter toy model cannot be assumed to generalize to large-scale foundation models (such as CLIP-ViT-B or LLMs) where model merging is actually used in practice.
2. **Marginal Practical Utility vs. Computational Cost:** 
   - Static Uniform Merging (Task Arithmetic) requires **zero** test-time optimization, **zero** backpropagation steps, and has **zero** runtime overhead, achieving **$62.16\%$** joint average accuracy.
   - PG-Merge ($p=0.05$) achieves **$62.70\%$** accuracy, which represents a tiny improvement of only **$0.54\%$**.
   
   Is a $0.54\%$ accuracy improvement worth the computational cost of running **100 gradient steps** of full-network backpropagation on the test stream? In most edge or real-time deployment settings, this trade-off is highly unfavorable.
3. **Persistent Failure to Close the Interference Gap:** The average individual expert ceiling is **$78.08\%$**. Static uniform averaging drops this to $62.16\%$ due to parameter interference (a $15.92\%$ gap). PG-Merge ($p=0.05$) improves this to $62.70\%$, meaning it closes only **$3.4\%$** of the remaining gap to the Expert Ceiling, while $96.6\%$ of the degradation remains unresolved. Active coefficient adaptation on tiny calibration sets appears highly ineffective in this setting, a limitation the paper fails to discuss.

---

## 4. Detailed Section Ratings

### Soundness: Fair
The mathematical formulation is clear, and the ablation study over $p$ is thorough. However, the technical contradiction between the theoretical advocacy for SGD (Appendix A) and the sole empirical evaluation using Adam, combined with the overstated "training-free" claims and the highly sensitive dependency on the hyperparameter $p$, limits its soundness.

### Presentation: Fair
While the paper is highly fluent and well-structured, its presentation rating is significantly lowered by careless scholarly practices. These include the mischaracterization of MECTA, the missing citation and bibliography entry for MECTA, the omission of EATA, the inclusion of dozens of uncited "ghost" bibliography entries, and discussing methods (like `QWS-Merge`) that are completely absent from the references.

### Significance: Fair
The significance is currently limited by the toy scale of the experiments (`vit_tiny` on 1,024-image subsets). Furthermore, the practical utility is questionable: achieving a marginal $0.54\%$ accuracy gain over static uniform merging at the extreme computational expense of 100 backpropagation steps on the test stream does not represent a strong contribution.

### Originality: Fair
While applying coordinate-sparsity specifically to merging coefficients is technically a new application, the core conceptual novelty is incremental. The underlying philosophy—stabilizing the test-time adaptation loop and preventing transductive collapse by updating only a sparse subset of parameters—is already a mature concept in the TTA literature (pioneered by EATA and MECTA). The authors' claims of being the first to "repurpose" gradient selection as a test-time regularizer are inaccurate due to their mischaracterization of MECTA.

---

## 5. Overall Recommendation
**Recommendation: 3: Weak Reject**

**Justification:**
This submission proposes a conceptually elegant, minimalist alternative to complex model-merging regularizers. The clear mathematical exposition, systematic ablation over sparsity, and insightful optimizer momentum decay analysis in Appendix A are definite strengths. 

However, in its current form, the paper's weaknesses outweigh its merits. The severe scholarly and literature positioning errors—including mischaracterizing and failing to cite MECTA, omitting the seminal EATA paper, and maintaining a careless bibliography—must be corrected. Methodologically, the contradiction between the Adam vs. SGD formulation remains unresolved. Empirically, the toy-scale evaluation (`vit_tiny` on 1,024-image subsets) raises significant generalizability concerns, and the actual practical improvement over the zero-compute static baseline is extremely marginal ($0.54\%$) relative to the high computational expense of 100 backpropagation adaptation steps. 

I encourage the authors to address these critiques. Scaling the experiments to standard CLIP-ViT or LLM backbones, resolving the scholarly omissions, and empirically validating the SGD-based PG-Merge formulation would significantly strengthen this work and elevate it to a clear accept.

---

## 6. Questions and Constructive Feedback for the Authors
1. **Adam vs. SGD Validation:** Since Appendix A theoretically demonstrates that momentum-free SGD is the mathematically clean and optimal choice for PG-Merge, why is the entire empirical evaluation conducted using Adam? Please provide comparative experimental results showing PG-Merge's performance under SGD vs. Adam, and evaluate whether SGD indeed makes the post-update parameter projection (Equation 13) redundant.
2. **Correct MECTA Citation and Characterization:** Please correct the discussion of MECTA in Section 2.4. Acknowledge that MECTA is a test-time adaptation framework, add its full bibtex reference to `references.bib`, and use a proper `\cite{...}` tag in the text.
3. **Incorporate EATA Discussion:** Please cite and discuss EATA (ICML 2022) in the Related Work section, clarifying how PG-Merge's gradient selection on merging coefficients compares to EATA's parameter selection on model weights.
4. **Clean up Bibliography:** Please remove the dozens of uncited "ghost" references in `references.bib` and ensure that all methods mentioned in the text (such as `QWS-Merge`) have corresponding bibliography entries.
5. **Scale Up Evaluation:** To demonstrate the generalizability of PG-Merge, can you provide evaluation results on standard large-scale foundation backbones (e.g., CLIP-ViT-B/32 or CLIP-ViT-L/14) and standard multi-task vision datasets?
6. **Computational Efficiency Analysis:** Please include a brief wall-clock time or FLOP complexity analysis comparing PG-Merge (100 steps of full backpropagation on the calibration stream) with the static Uniform Merging baseline, and moderate the "training-free" and "zero computational overhead" claims to reflect this trade-off.
7. **Baseline Tuning:** Could the authors clarify if PolyMerge was properly tuned? The complete collapse of PolyMerge on MNIST ($13.48\%$) suggests that the baseline might have been evaluated under sub-optimal hyperparameters or learning rates.
