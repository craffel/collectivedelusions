# Peer Review: GranMerge: Deconstructing the Generalization-Granularity Trade-off in Adaptive Model Merging

## 1. Summary of the Paper
This paper presents **GranMerge**, an empirical framework designed to deconstruct the "Generalization-Granularity Trade-off" in test-time adaptive multi-task model merging. The authors systematically study five nested levels of parameter resolution for weight-blending coefficients:
*   **Level 1 (Global):** 1 scalar coefficient per task ($K$ parameters).
*   **Level 2 (Layer-wise):** 1 scalar coefficient per layer per task ($L \times K$ parameters).
*   **Level 3 (Block-wise):** 2 coefficients (Attention vs. MLP) per layer per task ($2 \times L \times K$ parameters).
*   **Level 4 (Component-wise):** 4 coefficients per layer per task ($4 \times L \times K$ parameters).
*   **Level 5 (Tensor-wise):** 6 coefficients per layer per task ($6 \times L \times K$ parameters).

The blending coefficients are optimized on a small, unlabeled calibration stream ($N=256$) using prediction entropy as an unsupervised surrogate loss. The paper evaluates two optimization paradigms: first-order gradient descent (**Adam**) and zero-order stochastic search (**1+1 ES**), alongside two soft L2 regularizers (Elastic Spatial Regularization and Total Variation depth-wise smoothness) to mitigate transductive overfitting. The evaluation is conducted on a 12-layer Vision Transformer (ViT-Tiny) across 4 datasets (MNIST, FashionMNIST, CIFAR-10, SVHN).

The main findings are: (1) increasing granularity leads to severe transductive overfitting on calibration batches; (2) 1+1 ES exhibits better generalization than Adam under high-dimensional search spaces; (3) soft regularizers successfully stabilize both optimizers but are insufficient to arrest Adam's unconstrained overfitting; and (4) despite adaptation and regularization, no optimized configuration outperforms the simple, static Uniform Task Arithmetic baseline of 30.41%.

---

## 2. Overall Evaluation and Ratings
*   **Overall Score:** **3: Weak Reject** (A paper with clear merits, but also some weaknesses, which overall outweigh the merits. The paper has strong academic/diagnostic interest, but major methodological limitations and a lack of demonstrated practical utility prevent an accept recommendation in its current form.)
*   **Soundness:** **Fair** (The central claims about transductive overfitting are mathematically intuitive, but the extremely poor performance of the downstream experts makes the entire study highly vulnerable to being a low-fidelity toy-model artifact. The interpretation of ES as an "implicit regularizer" is also highly questionable and likely reflects optimization failure.)
*   **Presentation:** **Good** (The writing is exceptionally clear, structured, and pleasant to read. However, there are minor textual inconsistencies and a critical omission of the hyperparameter values used for the regularizers.)
*   **Significance:** **Fair** (Since the proposed regularizations and optimizers merely act as damage control and fail to outperform the unoptimized static baseline, the practical significance is very limited. Additionally, it remains unclear whether these findings generalize to high-capacity foundation models with high-fidelity experts.)
*   **Originality:** **Good** (The systematic nesting of five structural granularities and the comparison of first-order vs. zero-order optimization trajectories in parameter-blending spaces provide useful empirical insights.)

---

## 3. Main Strengths
1.  **Highly Systematic Hierarchical Framework:** Standardizing the nested structural granularities of merging scales from Global down to Tensor-wise is a very clean, logical, and elegant formulation.
2.  **Thorough Multi-Axis Benchmarking:** Sweeping across multiple tasks, two completely different optimizer families (first-order vs. zero-order), and multiple random seeds shows a commendable commitment to thorough empirical characterization.
3.  **Clear and Structured Writing:** The paper is beautifully written, with an extremely clear narrative flow. The introduction and related work sections are highly cohesive and accurately position the work in the broader literature.

---

## 4. Main Weaknesses (Critical Flaws)

### Weakness 1: Weak, Under-converged "Expert" Models Confound the Study
A primary methodological weakness is the extremely poor performance of the downstream "expert" models used to conduct the study. Looking at the expert test accuracies in Table 1:
*   **MNIST Expert:** 61.03% (Standard classifiers easily achieve >98%).
*   **FashionMNIST Expert:** 62.47% (Standard classifiers easily achieve >90%).
*   **CIFAR-10 Expert:** 24.93% (Barely above the 10% random-guessing baseline; extremely weak).
*   **SVHN Expert:** 17.50% (Barely above the 10% random-guessing baseline; extremely weak).
*   **Overall Mean:** 41.48%

These are not "experts" at all; they are extremely poorly-trained, under-converged, low-fidelity networks. This raises two severe concerns:
1.  **Overfitting as an Artifact of Noise:** Because the experts are so weak, their corresponding task vectors ($\theta_k = W_k - W_{\text{base}}$) represent highly noisy, erratic, and unaligned directions in parameter space. The severe transductive overfitting observed at high granularities (like Level 5) is highly likely an artifact of this parameter noise. In high-fidelity foundation model settings (e.g., CLIP-Large), representations are highly structured, and merging task vectors is naturally far more stable. 
2.  **Surrogate Loss Failure:** Minimizing prediction entropy on small calibration batches ($N=256$) is particularly hazardous when the base model and experts are weak. Flat, noisy softmax outputs are easily "sharpened" into incorrect classes (highly confident but wrong predictions) by random parameter adjustments, making the entropy objective mathematically counterproductive in this regime.

Evaluating model merging on networks that fail to solve basic toy classification tasks significantly undermines the generalizability and trustworthiness of the paper's empirical conclusions.

### Weakness 2: Overinterpretation of ES "Implicit Regularization" (Optimization Sluggishness)
The authors claim that *zero-order 1+1 Evolution Strategies act as a robust implicit regularizer* because they use isotropic random mutations that are naturally self-limiting. 

However, a much more standard and mathematically grounded explanation is **optimization sluggishness (underfitting) due to the curse of dimensionality**. 
Zero-order optimization is notoriously inefficient in high-dimensional spaces. Optimizing 288 parameters (Level 5) using 1+1 ES for only 100 steps is extremely unlikely to converge or even move significantly from the starting parameters. 
Since the model is initialized at a uniform scale (which is close to the static Uniform baseline of 30.41%), the "superior generalization" of 1+1 ES at higher granularities is likely a trivial consequence of its **failure to optimize**. It simply underfits the calibration loss and remains stuck near its initial uniform state. 

We can observe this by tracking how 1+1 ES performance changes as dimensionality increases in Table 1:
*   **Level 1 (4 params):** 1+1 ES gets **24.84%** (Dimensionality is low; ES successfully optimizes/moves away from the initialization, leading to severe underfitting/overfitting).
*   **Level 2 (48 params):** 1+1 ES gets **29.17%** (Optimization slows down).
*   **Level 3 (96 params):** 1+1 ES gets **29.65%** (Slower still).
*   **Level 4 (192 params):** 1+1 ES gets **29.98%** (Even slower, staying closer to the 30.41% uniform baseline).
This clear trend strongly supports the "sluggish optimization" hypothesis. Framing this as beneficial "implicit regularization" is a very generous interpretation of optimization failure in high dimensions. To prove implicit regularization, the authors would need to show that 1+1 ES achieves comparable calibration loss (entropy) to Adam while maintaining better test accuracy, which is highly unlikely.

### Weakness 3: Supremacy of the Static Uniform Baseline & Lack of Practical Utility
A highly critical finding in Table 1 is that **not a single adaptive merging configuration, even when regularized with ESR and TV, outperforms the static, zero-overhead Uniform Task Arithmetic baseline of 30.41%.**
*   Uniform Task Arithmetic: **30.41%**
*   Best regularized adaptive model (1+1 ES Level 5): **30.17%**
*   Best Adam adaptive model (Adam Level 2): **29.18%**

This means that test-time adaptation, in this regime, is completely useless or even harmful. The proposed regularizations (ESR, TV) and ES optimizers merely act as "damage control" to mitigate the harms of adaptation, rather than providing any performance boost over the unoptimized baseline. This raises a major question about the viability of adaptive merging in low-fidelity edge regimes. 

Additionally, there is a **critical reproducibility omission**: Section 3.5 introduces regularization scale $\beta$ and TV balance $\gamma$, but the authors **never state the values of $\beta$ and $\gamma$** used to generate the regularized results in Table 1. This violates basic machine learning reproducibility standards.

---

## 5. Detailed Feedback and Suggestions for Improvement
1.  **Correct Textual Inconsistency in the Abstract:** 
    In the abstract (Line 10), the authors list the five granularities as "(Global, Block-wise, Layer-wise, Component-wise, and Tensor-wise)". However, in Section 3.2 and Table 1, Level 2 is defined as **Layer-wise** and Level 3 as **Block-wise**. The abstract should be updated to match the final monotonic parameter ordering: **(Global, Layer-wise, Block-wise, Component-wise, and Tensor-wise)**.
2.  **Add Optimization Diagnostics:** 
    The paper makes strong assertions about optimizer dynamics without providing direct empirical evidence. The authors should include:
    *   **Calibration Loss Curves:** Plots showing prediction entropy decreasing over optimization steps for Adam vs. ES across different granularities.
    *   **Parameter Trajectory Plots:** Visualizations of coefficient drift over optimization steps to empirically show if Adam coefficients take extreme, unphysical values while ES coefficients remain tightly clustered.
3.  **Validate on High-Fidelity Experts:** 
    To prove that the findings are not simply toy-model artifacts, the authors must validate their framework on at least one high-fidelity expert setting (e.g., downstream experts that are fully converged and achieve >85-90% accuracy). This will demonstrate whether the "Generalization-Granularity Trade-off" is a general physical property of model merging or merely a consequence of noisy task vectors.

---

## 6. Questions for the Authors
1.  Can you provide the final calibration entropy loss values reached by both Adam and 1+1 ES at Level 5? Did Adam achieve a substantially lower calibration loss than 1+1 ES, supporting the hypothesis that ES's generalization is due to optimization sluggishness rather than implicit regularization?
2.  What were the exact hyperparameter values for the regularization scale $\beta$ and TV balance $\gamma$ used in the experiments?
3.  If you train the downstream experts to full convergence (e.g., >95% MNIST, >80% CIFAR-10), does the "Generalization-Granularity Trade-off" behave similarly? Or do the cleaner, higher-fidelity task vectors naturally resist transductive overfitting?
