# Peer Review of "Barycentric Proximity-Anchored Merging: A Critical, Deconstructive Audit of Parameter-Frugal Test-Time Model Merging"

## 1. Summary of the Paper
This paper investigates the boundaries of "parameter frugality" in test-time adaptive model merging. While state-of-the-art adaptive methods introduce substantial architectural complexity (e.g., millions of parameters in normalizing flows or thousands of layer-wise adapters), this work proposes **Barycentric Proximity-Anchored Merging (BPAM)** to explore the absolute limits of parameter reduction. BPAM restricts the weight-space blending coefficients to exactly $K$ global task-wise scalars ($K=8$ for the evaluated benchmark). To stabilize optimization and prevent activation scale distortion, the authors formulate a **Convex Barycentric Simplex Projection** (with a ray-scaling heuristic) and a **Mean-Field Proximity Penalty** (which anchors coefficients toward a uniform centroid). 

Evaluating three spatial configurations on an 8-task image classification benchmark using CLIP ViT-B/32, the authors show that:
1.  **BPAM-Restricted** (merging limited to a single projection layer) collapses performance to **51.38%**, proving that whole-model blending is necessary.
2.  **BPAM-Static** (merging the whole encoder with 8 global scalars, frozen heads) achieves **69.21%** average accuracy, matching linear Task Arithmetic.
3.  **BPAM-Full** (merging the whole encoder and concurrently tuning classification heads) achieves **75.22%** average accuracy, showing that downstream head tuning drives performance in low-parameter regimes.

The paper takes a deconstructive approach, analyzing why their proximity penalty is redundant in standard settings, why MNIST and SVHN experts still perform well when assigned zero weight, and mapping where layer-wise degrees of freedom become essential.

---

## 2. Strengths and Weaknesses

### Strengths
1.  **Excellent Scientific Honesty and Transparency:** The authors deserve high praise for their intellectual integrity. Rather than hiding negative results, they openly report and methodologically deconstruct why their proposed Mean-Field Proximity Penalty is empirically redundant under standard settings ($\beta = 0.0$ vs. $\beta = 10^{-2}$ yields virtually identical results).
2.  **Clear Narrative and Presentation Quality:** The paper is beautifully written, exceptionally well structured, and clear. The mathematical formulation of the convex simplex, the ray-scaling projection, and the co-adaptation dynamics are presented with precision.
3.  **Insightful Architectural Deconstruction:** The analysis of localized bottleneck merging (BPAM-Restricted) vs. whole-model blending is an instructive lesson that confirms the necessity of whole-model coordinate interpolation for multi-task model merging.
4.  **Rigorous Geometric and Feature Analysis:** The discussion of the "0-Weight Performance Mystery" is supported by Centered Kernel Alignment (CKA) feature similarity analysis, providing concrete evidence of representation sharing across fine-tuned experts.

### Weaknesses
While the paper's transparency is commendable, several critical methodological flaws, weak baselines, and a lack of empirical support for theoretical claims severely undermine its significance and soundness:

1.  **Major Baseline Omission (The Zero-Shot CLIP Baseline):**
    The entire narrative surrounding the **"0-Weight Performance Mystery"** (how the model achieves 88.09% on MNIST and 78.15% on SVHN when their coefficients are $0.0000$ and the base model is completely suppressed) is highly suspect due to a missing baseline. Standard, untouched pre-trained CLIP ViT-B/32 already possesses robust zero-shot classification capabilities on SVHN and MNIST. By completely omitting the zero-shot base model accuracies from Table 1, the authors make this appear to be a profound "reconstructed representation space" mystery. If the base model already achieves ~85% on MNIST and ~75% on SVHN, then getting 88% and 78% is merely the default capability of the pre-trained encoder, not a profound property of representation sharing in model-merging. Failing to report the zero-shot base model performance on all 8 datasets is a severe methodological oversight.

2.  **Unconstrained Scaling Consistently Outperforms the Proposed Method:**
    The core mathematical contribution of BPAM is the **Convex Barycentric Simplex Projection**, which the authors claim functions as a "critical, scale-preserving structural safeguard" against "activation scale distortion."
    However, their own empirical results (Table 1) show that **Unconstrained Scaling** (which completely strips away the simplex constraints and proximity penalty) consistently and substantially outperforms their constrained method:
    *   Under frozen heads, Unconstrained Scaling achieves **71.51%** average accuracy (+2.30% absolute improvement over BPAM-Static's 69.21%).
    *   Under active heads, Unconstrained + Head Tuning achieves **77.12%** average accuracy (+1.90% absolute improvement over BPAM-Full's 75.22%).
    
    This demonstrates that their proposed scale-preserving constraint actually **degrades** performance. While they justify this constraint based on a theoretical fear of "activation collapse," **they provide absolutely no empirical evidence of this collapse occurring in any setting.** Methodologically, introducing a complex constraint that significantly hurts performance based on a purely theoretical fear—without demonstrating a single instance where unconstrained scaling actually fails—is highly unsound.

3.  **Strictly Dominated by Static, Zero-Compute Baselines:**
    The proposed method is strictly dominated by existing static, zero-compute alternatives combined with simple head tuning:
    *   **Frozen Heads:** BPAM-Static (69.21%) underperforms compared to static **TIES-Merging** (72.90%) by **-3.69%**.
    *   **Active Heads:** BPAM-Full (75.22%) underperforms compared to **TIES-Merging + Head Tuning** (78.50%) by **-3.28%**.
    
    To achieve its inferior performance, BPAM requires **200 epochs of test-time optimization** taking **14.2 minutes** of GPU runtime, whereas TIES-Merging requires **0.0 minutes** of runtime. Simply applying decision-boundary head tuning on top of TIES-Merging is strictly superior to running joint adaptive optimization under low-parameter constraints. This completely undermines the utility of the proposed method; there is no practical or performance-based incentive for any practitioner to adopt BPAM.

4.  **Ad-Hoc Projections and Unsound Joint Co-Adaptation Scales:**
    *   **Ray-Scaling Projection:** To project parameters onto the simplex, the authors employ an ad-hoc ray-scaling heuristic rather than an exact orthogonal Euclidean projection. Standard projected gradient descent (PGD) relies on orthogonal projections to preserve convergence guarantees. By employing an ad-hoc scaling heuristic, the authors alter the optimization trajectory in a way that lacks theoretical convergence guarantees.
    *   **Optimization Scale Mismatch:** In BPAM-Full, the authors optimize 8 scalar parameters and 388,096 classification head parameters concurrently using a uniform learning rate ($\eta = 10^{-3}$). This is highly unsound. The classification head parameters outnumber the weight-space scalars by nearly five orders of magnitude. Joint optimization with a uniform learning rate is highly likely to cause the classification heads to overfit and dominate the loss, rendering the weight-space optimization stagnant. The authors claim they "extended their codebase" to support asymmetric co-adaptation schedules, but they **present absolutely no empirical results or ablations in their tables** evaluating this schedule, leaving it as a speculative assertion.

5.  **Complete Lack of Statistical Significance Metrics:**
    Test-time adaptation is highly sensitive to batch ordering and local data shifts. Despite this, the authors present all experimental results as single-run deterministic values. They provide **no error bars, no standard deviations, and no statistical significance tests** (such as p-values) over multiple random seeds or batch shuffles. Reporting marginal improvements (such as +0.11% for BPAM-Static over Task Arithmetic) as meaningful scientific findings without verifying their statistical significance is a major empirical weakness.

6.  **Impractical Memory Overhead during Calibration:**
    To compute the joint KL-divergence objective against expert teachers, the adaptation phase requires feeding each batch through **all $K$ expert teacher networks** simultaneously. For this 8-task benchmark, this requires hosting and running **9 parallel foundation models in GPU memory** (8 teachers + 1 merged model) during the 14.2-minute calibration phase. In real-world settings where model merging is used precisely to *reduce* hosting costs, requiring a 9x memory footprint for adaptation is highly self-defeating.

---

## 3. Soundness
**Rating: Fair**

**Justification:**
While the mathematical derivations and optimization equations are clear and theoretically reproducible, the technical soundness is severely compromised by several factors:
1.  The omission of the zero-shot base model baseline, which invalidates the "0-weight performance mystery" narrative.
2.  The fact that unconstrained scaling significantly outperforms their proposed constrained method, meaning their core mathematical contribution actually hurts performance, with no empirical evidence provided to prove that unconstrained scaling ever collapses.
3.  The ad-hoc ray-scaling projection heuristic lacking theoretical convergence guarantees.
4.  The joint optimization of parameters differing by five orders of magnitude (8 vs. 388K) using a uniform learning rate and optimizer without any empirical validation of their supposed code extensions/asymmetric schedules.

---

## 4. Presentation
**Rating: Good**

**Justification:**
The overall presentation is highly polished, well structured, and clear. The writing style is articulate, and the mathematical formulations are easy to follow. However, the rating is capped at "Good" because the paper overstates several conceptual findings (such as framing zero-weight performance as a "mystery" without providing the zero-shot base model baseline) and introduces a complex regularizer that is shown to be completely redundant under standard settings.

---

## 5. Significance
**Rating: Fair**

**Justification:**
The significance of this work is primarily conceptual and educational, functioning as a "boundary probe" baseline that maps the absolute limits of parameter-frugal adaptive merging. 

However, its practical significance is extremely low. Because the proposed BPAM method is strictly dominated by static, zero-compute alternatives (like TIES-Merging combined with downstream head tuning) and is vastly outperformed by higher-capacity adaptive methods (by over 14%), practitioners have no incentive to adopt BPAM in real-world systems. Furthermore, the extreme memory footprint of loading all $K$ expert teachers concurrently during calibration severely restricts its real-world utility.

---

## 6. Originality
**Rating: Fair**

**Justification:**
The originality of the paper is highly incremental. The concept of test-time adaptive merging coefficients was already pioneered by AdaMerging. BPAM's technical formulation (optimizing $K$ global scalars, applying convex simplex constraints, and using L2 regularizers and KL-divergence losses) consists entirely of standard, classical mathematical components. Its primary novelty is conceptual, packaging these standard techniques into a "deconstructive audit" narrative.

---

## 7. Questions and Actionable Feedback for Authors
1.  **Zero-Shot Baseline:** You must add the zero-shot classification accuracies of the untouched, pre-trained CLIP ViT-B/32 base model on all 8 datasets to Table 1. This is critical to verifying whether the SVHN and MNIST performances under zero weight are indeed due to representation sharing or simply due to the base model's default capabilities.
2.  **Activation Collapse Evidence:** Since Unconstrained Scaling outperforms BPAM by up to 2.30%, you must provide concrete empirical evidence (such as activation norm plots, gradient scales, or loss curves) showing that unconstrained scaling actually leads to "activation scale distortion" or "catastrophic representation collapse" in some scenarios. If no such collapse can be demonstrated, the simplex projection constraint is an unnecessary restriction that simply degrades performance.
3.  **Statistical Significance:** Please run your test-time adaptation experiments over multiple random seeds/batch shuffles and report standard deviations and statistical significance tests for Table 1. This is essential to confirm whether marginal improvements (e.g., +0.11% for BPAM-Static over Task Arithmetic) are scientifically meaningful.
4.  **Asymmetric Co-Adaptation Schedule:** You mention that you extended your codebase to support asymmetric learning rates for the classification heads and weight scalars. Please provide a systematic ablation table comparing different values of $\eta_{head}$ and $\eta_{weight}$ to back up your claims regarding this schedule.
5.  **Teacher-Free Adaptation:** To address the severe memory bottleneck of loading all $K$ expert teachers in GPU memory during adaptation, have you explored teacher-free adaptation objectives (such as entropy minimization or self-training)? Discussing this in the future work or limitations section would strengthen the paper.

---

## 8. Overall Recommendation
**Overall Recommendation: 3: Weak Reject**

**Justification:**
This paper has clear merits: it is exceptionally well-written, mathematically clear, and exhibits rare intellectual honesty by openly deconstructing its own redundant regularizer. The analysis of localized bottleneck merging vs. whole-model blending is highly instructive, and the representation sharing analysis is interesting.

However, the weaknesses currently outweigh the merits. The proposed BPAM method is strictly dominated by static, zero-compute baselines combined with simple head tuning (such as TIES-Merging + Head Tuning, which outperforms BPAM-Full by 3.28% absolute while requiring zero weight optimization). Furthermore, their core proposed simplex projection constraint actually degrades performance compared to unconstrained scaling (with no empirical evidence provided to prove that unconstrained scaling ever collapses), and their "0-weight performance mystery" narrative is highly suspect due to the complete omission of the zero-shot pre-trained base model baseline. 

The paper requires major revisions—specifically, adding the zero-shot base baseline, providing empirical evidence of unconstrained scaling collapse, reporting statistical significance metrics, and validating their asymmetric learning rate schedule—before it can be recommended for acceptance.
