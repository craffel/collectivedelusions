# Peer Review: PAC-Bayes Merge

## Summary of the Paper
This paper presents **PAC-Bayes Merge**, a statistical learning-theoretic framework for post-hoc parameter-space model merging. The paper aims to resolve the "Overfitting-Optimizer Paradox" (or the transductive overfitting trap) which occurs when layer-wise merging coefficients are dynamically optimized on extremely small few-shot calibration datasets (e.g., $M = 10$ samples per task). 

To resolve this overparameterized coordinate-space optimization challenge, the authors:
1. Restrict layer-wise ensembling coefficients to follow a continuous, low-degree (typically cubic) polynomial trajectory across normalized network depth.
2. Frame the trajectory parameters as the mean of a randomized isotropic Gaussian posterior distribution $Q$ and specify a spherical Gaussian prior $P$ centered at the stable uniform ensembling consensus baseline $\Theta_{\text{uniform}}$.
3. Mathematically prove that minimizing Alquier's linear PAC-Bayesian bound analytically yields a quadratic $L_2$ Consensus-Pulling penalty centered at the stable uniform consensus.
4. Derive a non-isotropic variant (PAC-Bayes-FIM Merge) where prior variances are scaled inversely by the empirical Fisher sensitivities of coordinates evaluated at the uniform consensus point.
5. Implement randomized Monte Carlo expected risk training and posterior ensemble test-time evaluations to bridge the theory-to-practice gap.

The approach is evaluated on a "projected representation sandbox" using MNIST, FashionMNIST, CIFAR-10, and SVHN datasets projected via random Johnson-Lindenstrauss (JL) mappings to 192 dimensions and passed through a 14-layer residual MLP backbone.

---

## Strengths and Weaknesses

### Strengths
1. **Mathematical Grounding:** The paper introduces a highly rigorous and elegant theoretical foundation for model merging. Applying the PAC-Bayesian framework directly to the ensembling trajectory parameters is a creative conceptual direction that connects weight-space fusion to statistical learning theory.
2. **Analytical Derivations:** The mathematical derivations in Section 4 are complete and presented with high clarity, demonstrating exactly how a Gaussian PAC-Bayesian prior centered at the uniform baseline analytically justifies a smooth quadratic $L_2$ penalty.
3. **Clarity of Presentation:** The overall narrative, structure, and writing style are highly professional and easy to follow.
4. **Comprehensive Scaling Blueprint:** Appendix A provides an exceptionally detailed, step-by-step scaling blueprint for physical computer vision backbones (e.g., Vision Transformers) and autoregressive Large Language Models (LoRA adapter merging), which strongly benefits future practitioners looking to build upon this work.

### Weaknesses (Critical)

1. **Systemic Inconsistencies in Data Reporting (Scientific Integrity):**
   There is a major, severe discrepancy between the narrative claims made in the text (Abstract, Intro, Conclusion) and Table 2 (Ablation Table) versus the actual main experimental results in Table 1 (which match the raw data in `results.json`).
   - **Textual Discrepancies:** The Abstract, Section 2.1 (L21), and Section 6 (L6) state that their advanced Fisher-guided non-isotropic PAC-Bayes-FIM Merge achieves a Joint Mean accuracy of **36.13%**, outperforming Static Uniform (**33.57%**), properly tuned Ties-Merge (**29.68%**), DARE-Merge (**33.24%**), and unconstrained layer-wise tuning (**36.09%**).
   - **Table 1 Discrepancies:** In Table 1, the true numbers are actually **35.37%** (for PAC-Bayes-FIM), **35.51%** (for Offline Unconstrained), **33.35%** (for Static Uniform), **29.59%** (for Ties-Merge), and **32.76%** (for DARE-Merge).
   - **Table 2 (Ablation) Discrepancies:** Table 2 reports that the default model yields a Joint Mean of **36.09 $\pm$ 2.23%**, whereas Table 1 reports it as **35.24 $\pm$ 2.85%**. Almost all other numbers in Table 2 are completely different from the actual experimental keys in `results.json` (e.g., `ablation_0.001` is actually **35.37%** vs. **36.15%** reported in Table 2; `ablation_0.5` is actually **33.95%** vs. **34.34%** in Table 2; `ablation_sigma_0.01` is actually **35.36%** vs. **36.22%** in Table 2; `ablation_sigma_0.15` is actually **35.38%** vs. **36.08%** in Table 2).
   This represents a critical breakdown in data reporting and statistical integrity. Almost all figures in the main text and Table 2 are mismatched compared to the true experimental results.

2. **Empirical Redundancy of the Proposed Regularization:**
   An explicit regularizer is designed to improve generalization over an unregularized baseline. However, the empirical results show that the proposed PAC-Bayes regularizer is empirically redundant or even slightly harmful compared to a simple, unconstrained baseline:
   - **Main Evaluation ($M = 10$):** In Table 1, the unregularized *Offline Unconstrained* baseline achieves a Joint Mean of **35.51 $\pm$ 2.63%**. Meanwhile, the proposed *Ours (Deterministic Compiled)* achieves **35.37 $\pm$ 2.81%** and *Ours (FIM Deterministic Compiled)* achieves **35.37 $\pm$ 2.84%**. Both proposed variants are actually **worse** than the unregularized, unconstrained model by **-0.14%** absolute.
   - **Extreme Scarcity Sweep ($M = 2$):** In the scarcity sweep (Figure 3 and `scarcity_results.json`), under extreme calibration scarcity of $M = 2$ samples per task, *Offline Unconstrained* yields **34.16 $\pm$ 3.13%**, whereas *Ours (PAC-Bayes)* yields **33.86 $\pm$ 3.36%** and *Ours (PAC-Bayes-FIM)* yields **33.43 $\pm$ 3.40%**. Again, the unregularized model outperforms the proposed regularized models by **+0.30%** and **+0.73%** absolute, respectively.
   This means that the massive mathematical machinery of PAC-Bayes Merge provides zero empirical benefit, and the claims in the text that it "outperforms unconstrained tuning" are entirely false.

3. **Weak, Toy Experimental Sandbox:**
   The paper evaluates exclusively on a highly simplified toy setup where MNIST, FashionMNIST, CIFAR-10, and SVHN are projected via random Johnson-Lindenstrauss mappings to 192 features and passed through a tiny, 14-layer deep MLP with a width of 64. 
   - Due to this toy setup, the task-expert base ceilings themselves are exceptionally weak (e.g., SVHN expert ceiling is only **17.57%**, which is barely above random guessing of 10%; CIFAR-10 is **25.07%**). 
   - Merging highly dysfunctional experts that are barely better than random guess makes it very difficult to draw generalizable deep learning conclusions. It is highly questionable whether these results would scale to actual physical vision backbones (e.g., standard ResNets or Vision Transformers on raw pixels), and the paper would be significantly stronger if the authors had evaluated on standard architectures as described in their Appendix A blueprint, rather than relying solely on this toy sandbox.

---

## Detailed Evaluation Ratings

### Soundness: Fair
The theoretical derivations, polynomial trajectory formulation, and Monte Carlo optimization strategies are highly sound, mathematically rigorous, and exceptionally clear. However, the core empirical soundness is "fair" (bordering on poor) because the central hypothesis—that the proposed PAC-Bayesian regularizer suppresses transductive overfitting and improves generalization over unconstrained tuning—is directly contradicted by their own experimental results. Across all evaluation configurations, the unregularized *Offline Unconstrained* baseline performs identically to or better than the proposed regularized variants, making the proposed regularizer empirically redundant.

### Presentation: Poor
While the paper is structurally well organized and uses professional LaTeX formatting, the presentation is rated as "poor" due to critical scientific integrity issues. Almost all data points cited in the Abstract, Introduction, Conclusion, and Table 2 (Ablation Table) are completely mismatched/inconsistent with the true experimental results presented in Table 1 and recorded in `results.json`. The narrative claims that their method outperforms the unconstrained tuning baseline are entirely false based on the actual numbers in Table 1.

### Significance: Poor
Post-hoc model merging is an important and active area of research. However, because the evaluation is confined to a weak, toy experimental sandbox (Johnson-Lindenstrauss projections with 64-width MLPs where SVHN expert accuracy is only 17.57%), and because the proposed method fails to provide any empirical benefit over a simple unregularized baseline, the significance of the paper's contribution in its current form is very low.

### Originality: Fair
The application of PAC-Bayes directly to ensembling trajectories is a creative direction. However, the individual methodological components are highly derivative of existing work. The polynomial trajectory parameterization is taken directly from RBPM. The derivation of an $L_2$ penalty from a Gaussian KL divergence is a standard Bayesian textbook derivation, and the Monte Carlo expected risk optimization and posterior ensembling are standard variational inference techniques (such as BayesByBackprop). Thus, the novelty is incremental.

---

## Overall Recommendation

**Recommendation: 2: Reject**

**Justification:**
This paper has clear merits in its elegant mathematical formulation and complete theoretical derivations. However, as an empiricist evaluation, the paper falls short of the standards of a peer-reviewed ML conference for the following reasons:
1. **Critical Discrepancies in Data Reporting (Scientific Integrity):** Almost all numeric claims in the main narrative (Abstract, Intro, Conclusion) and in Table 2 (Ablation Table) are completely different from the actual experimental results presented in Table 1 and recorded in `results.json`. For instance, the main text claims their method gets 36.13% and beats unconstrained tuning (36.09%), while Table 1 reveals their method gets 35.37% and is beaten by unconstrained tuning (35.51%). Table 2 reports an ablation default of 36.09% whereas Table 1 reports 35.24%. This massive reporting mismatch severely damages the credibility of the submission.
2. **Empirical Redundancy of the Proposed Method:** The central empirical claim is that the proposed PAC-Bayesian regularizer resolves the overfitting of unconstrained tuning. However, the actual experimental results show that the unregularized, unconstrained layer-wise baseline outperforms the proposed PAC-Bayes regularizer across standard calibration ($M=10$) and extreme scarcity ($M=2$). The proposed regularizer is empirically redundant.
3. **Toy Sandbox Evaluation:** Confining the experiments to a toy Johnson-Lindenstrauss projected sandbox with highly weak and dysfunctional experts (e.g., 17.57% SVHN and 25.07% CIFAR-10 accuracy) heavily limits the generalizability and practical significance of the findings.

The authors are strongly encouraged to correct their data reporting to accurately match their actual results, investigate why their regularizer underperforms a simple unconstrained baseline, and evaluate their framework on physical CV backbones or LLMs (following their excellent Appendix A blueprint) before re-submitting.
