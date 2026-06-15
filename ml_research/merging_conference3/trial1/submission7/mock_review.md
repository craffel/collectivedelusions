# Peer Review Report

## 1. Summary of the Paper
This paper presents a highly rigorous, methodology-focused sanity check and representational analysis of the layer-wise model merging paradigm in deep neural networks (specifically CLIP ViT-B/32). Recent state-of-the-art (SOTA) test-time adaptation model merging methods, such as AdaMerging and SyMerge, claim that optimizing merging coefficients layer-by-layer is critical to successfully navigating destructive interference between diverse task experts. 

The authors stress-test this fundamental assumption using a **Sanity-Checking and Interpretability Suite** containing three diagnostic treatments: 
1. **Intra-Task Layer Shuffling** (randomly permuting optimized coefficients across layers).
2. **Task-Wise Spatial Averaging** (replacing layer-wise coefficients with their average across all layers for that task, reducing parameter counts by 92.3%).
3. **Norm-Bounded Perturbation** (injecting relative Gaussian noise into the coefficients across varying scales $\gamma \in [0.05, 0.50]$).

The evaluation is performed across **3 independent random seeds (42, 100, 2026)** and two distinct optimizers: a zero-order **Adaptive 1+1 Evolution Strategy (1+1 ES)** and a first-order **Adam Gradient Descent (Adam GD)**. They also conduct activation-space representational similarity analysis using linear **Centered Kernel Alignment (CKA)** to measure feature-level similarity to original experts. 

In addition, the authors present a comprehensive validation suite in the appendix, consisting of:
*   An empirical evaluation of their proposed **Proximity Coefficient Regularization** penalty across a hyperparameter sweep of $\beta \in [0.0, 1.0]$.
*   A **Calibration Sample Size Sweep** ($N_{\text{cal}} \in [8, 128]$) mapping the transductive overfitting threshold.
*   A visual analysis of **Coefficient Variance Profiles** under both optimization regimes.
*   A comparison of proximity regularization against standard weight decay (Appendix F).

---

## 2. Key Strengths

1.  **Critical Scientific Self-Correction:** Model merging has recently trended toward increasingly complex, multi-parameter, and fine-grained schemes. This work provides a highly necessary, timely, and rigorous scientific course-correction, demonstrating that unconstrained layer-wise optimization is highly susceptible to transductive overfitting.
2.  **Outstanding Statistical Rigor:** Conducting all experiments across 3 independent random seeds and reporting exact standard deviations represents an exceptional standard of empirical validation for model merging literature, where single-seed evaluations are unfortunately the norm.
3.  **Elucidation of the "Overfitting-Optimizer Paradox":** The authors provide a highly sophisticated, unified explanation of optimizer behaviors during test-time adaptation. They demonstrate that under zero-order search, overfitting manifests as high-frequency optimization noise easily regularized via spatial averaging, whereas under first-order gradient descent, the optimizer finds a highly precise, delicate configuration of parameters that overfits the calibration statistics, making it extremely sensitive to shuffling or averaging without actually improving unseen test performance compared to the unoptimized baseline.
4.  **Discovery of the SVHN Rescue vs. CIFAR-10 Collapse Trade-off:** The paper looks beyond average accuracies to expose how multi-task metrics can mask catastrophic single-task performance collapse. It demonstrates that spatial averaging acts as a regularizer that breaks joint entropy bias and rescues simpler, "sacrificial" tasks (like SVHN) at the direct cost of destroying the delicate, functional layer-specific hierarchies needed for complex tasks (like CIFAR-10).
5.  **Exposing CKA-Accuracy Decoupling:** The paper is the first to systematically apply CKA representational similarity to model merging and identify a critical scientific discrepancy: spatial averaging leads to slightly higher CKA similarity, yet collapses downstream classification accuracy by 10.35% under Adam GD. This exposes a profound limitation of activation-level similarity metrics as proxies for weight-space decision boundary integrity.
6.  **Empirical Validation of Proximity Regularization (Appendix B & F):** Rather than leaving coefficient regularization as a theoretical recommendation, the authors have empirically executed a pilot sweep, proving that their proximity penalty ($\mathcal{L}_{\text{reg}}$ with $\beta = 0.5$) successfully stabilizes the optimizer, resolves optimization drift, and improves test performance (retaining 86.57% average accuracy) while outperforming standard weight decay which collapses the complex SVHN task.
7.  **Mapping the Transductive Overfitting Threshold (Appendix C):** The calibration size sweep elegantly demonstrates the data-insufficiency threshold under unconstrained gradient descent, showing that test-time adaptation stabilizes and generalizes effectively when $N_{\text{cal}} \ge 128$ images per task.
8.  **Pristine Presentation and Quality:** The manuscript is exceptionally well-written, mathematically precise, and structurally elegant. The high-quality visualizations (Figures 1-6) and detailed tables provide an outstanding level of clarity and readability.

---

## 3. Key Weaknesses

While the paper is technically outstanding and methodologically rigorous, there are three key weaknesses that limit its overall impact and significance:

1.  **Limited Scale and Resolution of Evaluation Datasets:** The experimental validation is restricted to relatively simple, low-resolution classification datasets (MNIST, FashionMNIST, CIFAR-10, SVHN). Modern model merging is increasingly applied to high-resolution, complex multimodal tasks or large-scale datasets (such as ImageNet). It is unclear whether the extreme landscape flatness and transductive overfitting behaviors observed here would manifest identically in more complex, high-resolution domains.
2.  **Task-Vector Proximity to Initialization:** On these simple datasets, the fine-tuned task vectors reside relatively close to their shared pre-trained CLIP base. This proximity explains why the merging landscape is exceptionally flat and why spatial averaging acts as an effective regularizer. On more distant or conflicting tasks (where task vectors diverge significantly from the pre-trained weights), layer-specific coefficients may have genuine physical and functional importance that simple spatial averaging cannot replace. This boundary of the "layer-specificity illusion" is not fully explored.
3.  **Absence of Language Model (NLP) Evaluation:** The model merging community has seen rapid growth and adoption in natural language processing (e.g., merging 7B+ parameter autoregressive language models). By restricting the evaluation to vision-only CLIP models, the findings may have limited direct impact or visibility within the NLP community, which represents a large portion of model merging researchers.

---

## 4. Minor Suggestions for Future Improvement (Actionable Feedback)

To maximize the scientific impact of the paper, the authors are encouraged to consider the following minor improvements:

1.  **Weighted/Temperature-Scaled Entropy Formulation:** In Section 4.5, the authors identify the joint entropy minimization task-bias, where the optimizer sacrifices the complex, high-entropy task (SVHN) to minimize total loss. Since the authors recommend using weighted or temperature-scaled entropy formulations, implementing and evaluating a simple baseline for this (e.g., dividing each task's entropy by its baseline validation entropy as shown in Appendix E) would represent a valuable extension.
2.  **Elaborating on LLM Architectural Differences:** In the Limitations section, the authors briefly mention modern autoregressive language models. Expanding this discussion to identify specific structural aspects of LLMs (e.g., attention blocks, query-key-value projections vs. feed-forward layers) where layer-specific conflicts are highly likely to occur would provide a much stronger roadmap for NLP researchers.
3.  **Varying Optimizer Hyperparameters:** Evaluating the sensitivity of the transductive overfitting threshold to other optimization hyperparameters—such as learning rate decay or Adam's $\beta_1, \beta_2$ parameters—would help clarify if the observed overfitting can be mitigated through standard optimizer tuning without explicit coefficient regularization.

---

## 5. Section Ratings and Justifications

### Soundness: Excellent (Rating: Excellent)
The methodology is exceptionally robust and logically consistent. The proposed diagnostic suite (shuffling, averaging, noise perturbations) represents a creative and rigorous framework to stress-test coefficient importances. Comparing zero-order 1+1 ES and first-order Adam GD successfully isolates optimizer suboptimality. Furthermore, the inclusion of the proximity regularization sweep and the calibration sample size sweep provides an outstanding, empirically complete validation of the authors' hypotheses.

### Presentation: Excellent (Rating: Excellent)
The writing style is scholarly, clear, and engaging. The mathematical formulations are clean and precise. The authors display exemplary academic maturity in their self-critical discussions, particularly when analyzing the CKA-accuracy decoupling and multi-task average metric illusions. The figures are of high quality, professional, and exceptionally well-integrated.

### Significance: Good (Rating: Good)
The paper addresses a highly important, timely topic and offers crucial course-corrections for the model merging community. However, because the evaluation is restricted to low-resolution classification tasks using a CLIP ViT-B/32 backbone, the direct generalizability to larger architectures and NLP models remains to be proven, which slightly tempens the overall scope of its immediate significance.

### Originality: Excellent (Rating: Excellent)
The diagnostic suite, the integration of CKA representational similarity to analyze coefficient-level features, the formalization of joint entropy minimization task-bias, and the empirical validation of the proximity coefficient regularization penalty represent highly original and valuable contributions that differentiate this work from existing literature.

---

## 6. Overall Recommendation

**Overall Recommendation: 5: Accept**

**Justification:**
This is an outstanding, technically solid paper that provides a crucial, timely scientific course-correction for the model merging and test-time adaptation communities. The authors' empirical evaluation is characterized by an exceptional standard of rigor, incorporating multiple random seeds, dual-optimizer regimes, noise sweeps, and CKA activation-space analyses. 

While the scale of the evaluation (low-resolution vision classification tasks) slightly limits its immediate generalizability to large-scale language models, the paper is methodologically flawless, deeply insightful, and beautifully presented. The authors have successfully resolved all typical limitations of diagnostic studies by providing a comprehensive validation suite in the appendix. The paper is highly deserving of publication and is likely to influence how future model merging methods are evaluated and regularized.
