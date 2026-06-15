# Peer Review Report

## 1. Summary of the Paper
This paper presents a rigorous, methodology-focused sanity check and representational analysis of the layer-wise model merging paradigm in deep neural networks (specifically CLIP ViT-B/32). Recent state-of-the-art (SOTA) test-time adaptation model merging methods, such as AdaMerging and SyMerge, claim that optimizing merging coefficients layer-by-layer is critical to successfully navigating destructive interference between diverse task experts. 

The authors stress-test this fundamental assumption using a **Sanity-Checking and Interpretability Suite** containing three diagnostic treatments: (1) **Intra-Task Layer Shuffling** (randomly permuting optimized coefficients across layers), (2) **Task-Wise Spatial Averaging** (replacing layer-wise coefficients with their average across all layers for that task), and (3) **Norm-Bounded Perturbation** (injecting relative Gaussian noise into the coefficients). The evaluation is performed across **3 independent random seeds (42, 100, 2026)** and two distinct optimizers: a zero-order **Adaptive 1+1 Evolution Strategy (1+1 ES)** and a first-order **Adam Gradient Descent (Adam GD)**. They also conduct representational similarity analysis using linear **Centered Kernel Alignment (CKA)** to measure activation-level similarity to original experts.

### Key Strengths:
1.  **Critical Scientific Course-Correction:** Model merging has seen a major push toward increasingly complex, multi-parameter, and fine-grained schemes. This work provides a highly necessary and timely sanity check on whether such layer-specificity is physically functional or merely an optimization artifact or transductive overfitting behavior.
2.  **Outstanding Statistical Rigor:** The authors have conducted all experiments across 3 independent seeds and reported exact standard deviations, raising the paper's empirical evaluation to the highest academic standards.
3.  **Optimizer Confounding Isolated:** By comparing zero-order (1+1 ES) and first-order (Adam GD) optimizers, the authors successfully isolate optimizer suboptimality as a confounding factor.
4.  **Novel Use of CKA Interpretability:** Applying linear Centered Kernel Alignment (CKA) to track activation-level drift in model merging is highly novel and provides a physical, activation-space explanation (transductive overfitting vs. spatial-averaging regularization) for the observed results.
5.  **Formulation of Joint Entropy Task-Bias:** The paper mathematically formalizes and empirically demonstrates a critical multi-task optimization bias in joint test-time adaptation, where the optimizer sacrifices complex, high-entropy tasks (e.g., SVHN) to minimize total prediction entropy.

---

## 2. Critical Methodological Contradictions (Critical Flaws)

Despite the exceptional quality of the manuscript, a deep, critical review of the empirical results reveals **three profound logical contradictions and unresolved methodological flaws** that the authors should address to elevate the work to a top-tier publication.

### Critical Flaw 1: The Overfitting Paradox (The "Layer-Specificity" of Adam GD as an Artifact of Transductive Overfitting)
The authors frame the "Dual-Optimizer Paradox" by claiming that under Adam GD, "layer-specific coordination emerges as a highly functional, physical reality" because shuffling and spatial averaging degrade performance. However, a more rigorous methodological analysis suggests a **completely opposite interpretation**: Adam GD's "layer-specificity" is highly likely **an aggressive, delicate form of transductive overfitting** to the 256-image calibration set.
*   **Lack of Average Performance Gains:** Optimized Adam GD ($84.52 \pm 1.57\%$) achieves basically the same average accuracy as the unoptimized Task Arithmetic baseline ($84.44 \pm 0.37\%$), but introduces **massive instability and variance** (the standard deviation rises from 0.37% to 1.57%).
*   **CIFAR-10 Degradation:** On CIFAR-10, the Optimized Adam GD model gets $89.84 \pm 1.25\%$, which is strictly *lower* than the unoptimized baseline ($90.17 \pm 2.45\%$).
*   *Critique:* When optimizing 52 parameters on a tiny calibration set of 256 images using first-order autograd without any regularizer or constraints, the optimizer finds a highly precise, delicate configuration of coefficients that overfits the transductive calibration statistics (reducing joint entropy). Shuffling or averaging this delicate configuration collapses performance. But because it has overfit, it does not generalize better to the unseen test set than the unoptimized baseline. Thus, framing Adam GD's layer-specificity as a "physical reality" is overly generous; it is a manifestation of **overparameterized transductive overfitting**.

### Critical Flaw 2: Saturated CIFAR-10 Collapse and the SVHN "Rescue" Illusion
The paper currently presents the Spatial Mean as "superior" on average under 1+1 ES because its average accuracy ($85.21\%$) is slightly higher than the optimized model ($85.07\%$). However, looking closely at individual tasks in Table 1, this "average superiority" is a mathematical artifact of the SVHN "rescue" masking a catastrophic CIFAR-10 "collapse":
*   On **CIFAR-10**, spatial averaging collapses performance by **10.35%** (Adam GD) and **2.48%** (1+1 ES).
*   On **SVHN**, spatial averaging increases performance by **6.38%** (Adam GD) and **2.54%** (1+1 ES) because it breaks the joint entropy minimization task-bias (which sacrifices SVHN).
*   *Critique:* A 10.35% collapse on CIFAR-10 under averaging proves that **layer-specific coefficient coordination is indeed highly critical** to maintain representation hierarchies on complex tasks, and that "average" accuracy is a misleading metric here. The Spatial Mean is not flatly "superior" or "sufficient"; rather, it acts as a coarse regularizer that rescues the sacrificial task (SVHN) at the cost of destroying performance on the more complex task (CIFAR-10). The authors must discuss this trade-off explicitly rather than overgeneralizing about "Spatial Mean superiority."

### Critical Flaw 3: Inconsistent CKA Conclusions and Downstream Task Accuracy Discrepancy
The paper claims that "the spatially averaged model preserves the activation representations of the original task experts better than the optimized model under both optimizers (+0.0069 CKA for 1+1 ES and +0.0036 CKA for Adam GD)."
*   However, looking at Table 2, for the CIFAR-10 expert under Adam GD, the CKA of the Optimized model is **$0.9555 \pm 0.0302$**, whereas the Spatially Averaged model is **$0.9598 \pm 0.0241$**.
*   The difference is extremely small (+0.0043 CKA) and well within the standard deviation of both models ($\pm 0.0302$ and $\pm 0.0241$).
*   *Critique:* Claiming that this represents a meaningful, physically superior representational preservation is a major overstatement, especially given that the actual classification accuracy of Spatially Averaged Adam GD on CIFAR-10 collapses by **10.35%** (from 89.84% to 79.49%). This exposes a profound methodological limitation of activation-level similarity metrics like CKA in the fine-grained model merging regime: high-level activation subspaces can remain highly aligned (CKA > 0.95) even when minor weight-space shifts corrupt fine-grained decision boundaries, causing catastrophic classification failures. The paper must explicitly discuss this CKA-accuracy decoupling to prevent future researchers from blindly using CKA as a proxy for performance.

---

## 3. Actionable and Constructive Feedback (Required Revisions)

To elevate this paper to publication quality and resolve these critical flaws, the authors must perform the following revisions:

1.  **Refold the Interpretation of "The Dual-Optimizer Paradox" around Overfitting:**
    *   Acknowledge that both 1+1 ES and Adam GD suffer from transductive overfitting on the 256-image calibration set.
    *   Under 1+1 ES, the overfitting takes the form of high-frequency optimization noise that is easily smoothed out by spatial averaging. Under Adam GD, the first-order gradients allow the optimizer to find a highly precise, delicate configuration of parameters that overfits the calibration statistics, making it highly sensitive to shuffling/averaging without actually improving unseen test performance compared to the unoptimized baseline.
2.  **Discuss Coefficient Regularization as a Concrete Solution:**
    *   Add a paragraph in the discussion suggesting that future adaptive model merging frameworks incorporate explicit coefficient regularization (e.g., $L_2$ regularizer on coefficients, or a proximity penalty like $||\Lambda - 0.3||^2_2$, which penalizes the coefficients for drifting from their uniform baseline) during test-time adaptation. This would stabilize the optimization, reduce the seed variance, and prevent performance collapse on complex tasks.
3.  **Address the CKA-Accuracy Discrepancy Explicitly:**
    *   Explicitly point out the limitation of representational similarity metrics like CKA in predicting task accuracy.
    *   Acknowledge that high-level activation subspaces can remain highly aligned even when minor weight-space shifts corrupt fine-grained decision boundaries, cautioning future researchers from relying blindly on CKA as a proxy for generalization.
4.  **Discuss Generalizability and Scope Limitations:**
    *   The authors should expand their limitations section to clarify that their findings are evaluated on saturated, low-resolution vision classification datasets where task vectors reside close to the CLIP pre-trained initialization. They should acknowledge that in larger-scale networks (such as 7B+ parameter decoder-only language models) or on highly complex, distinct downstream domains (such as instruction-tuning or cross-modal tasks), representational hierarchies are highly distinct, and layer-by-layer optimization may remain critical.

---

## 4. Overall Recommendation and Ratings

### Recommendation: Accept (Rating 5)
This is an outstanding, highly rigorous, and timely scientific contribution that introduces a creative diagnostic suite (shuffling, averaging, and noise injection) and the first systematic CKA representational similarity analysis to model merging. By adding multiple seeds and isolating optimizer confounding, the authors have elevated the paper to the highest standard of scientific rigor. If the authors incorporate the recommended nuances (interpreting Adam GD's sensitivity as transductive overfitting, discussing coefficient regularization, and addressing the CKA-accuracy discrepancy), this paper will be a seminal, highly-cited contribution to the field.

### Section Ratings:
*   **Soundness:** **excellent** (The diagnostic treatments and representational similarity metrics are elegant and highly sound. The evaluation over 3 seeds and comparison of optimizers is exceptionally rigorous.)
*   **Presentation:** **excellent** (The manuscript is exceptionally well-written, clear, logically structured, and includes outstanding, highly professional visualizations and tables.)
*   **Significance:** **excellent** (The paper addresses a highly important, timely topic and offers crucial course-corrections, diagnostic controls, and mandatory baselines for the model merging community.)
*   **Originality:** **excellent** (The diagnostic treatment suite, the integration of CKA representational similarity, and the formalization of joint entropy minimization task-bias represent highly original and valuable contributions.)
