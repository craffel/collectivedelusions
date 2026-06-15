# Peer Review

## Summary of the Paper
This paper addresses the task-scale mismatch and dominance issues in multi-task model merging, specifically within the foundational **Task Arithmetic** framework. When combining independently fine-tuned expert checkpoints, tasks with large distribution shifts or complex objectives undergo larger parameter shifts, yielding high-norm task vectors that overwhelm other tasks when directly summed. To resolve this representation imbalance, the paper introduces **Norm-Equalized Task Arithmetic (NETA)**, a training-free and parameter-free closed-form method that equalizes the Frobenius norms of task vectors at each layer before merging, ensuring isotropic representation strength. 

The paper also:
1.  Introduces a continuous **$\alpha$-relaxed NETA** framework to smoothly balance peak task performance and representation fairness.
2.  Provides a physically-grounded **noise-damping stabilizer ($\beta$)** to prevent the amplification of minor updates.
3.  Introduces an analytical, closed-form **scale-compensation factor ($\gamma^l$)** to correct for directional norm contraction.
4.  Exposes a critical vulnerability in test-time adaptation (TTA) methods like AdaMerging, termed the **"Overfitting-Optimizer Paradox"**, where joint prediction entropy minimization overfits to easy, low-entropy tasks and suppresses complex, high-entropy ones.
5.  Analyzes the boundary conditions of this paradox, attributing the resilience of Layer-Wise AdaMerging to its higher spatial degrees of freedom.

---

## Strengths

### 1. Conceptual Rigor and Insightful Critique (The Overfitting-Optimizer Paradox)
The paper's strongest contribution is the conceptualization and empirical demonstration of the **Overfitting-Optimizer Paradox** in Test-Time Weight Adaptation. Exposing that joint prediction entropy minimization on unlabeled calibration sets acts as an opportunistic proxy which discriminates against harder, high-entropy tasks (e.g., FashionMNIST) is a highly valuable, high-signal finding. This provides a timely warning to the model merging community regarding unconstrained test-time optimization under task-difficulty imbalances.

### 2. High Scientific Integrity and Transparency
The authors demonstrate exemplary scientific transparency. They do not hide unfavorable results or oversell NETA. Specifically:
*   They honestly discuss that NETA's isotropic regularization slightly reduces SVHN accuracy to achieve representation fairness.
*   They conduct a systematic grid search over $\lambda_0$ (Table 3), transparently reporting that when both Task Arithmetic and NETA are fully tuned to peak performance, Task Arithmetic still retains a marginal advantage in multi-task average accuracy ($89.16\%$ vs. $89.06\%$).
*   They mathematically qualify the directional norm contraction of merged updates and derive a training-free, automated scale-compensation factor $\gamma^l$ to resolve it.

### 3. Clear, Mathematically-Grounded Design
The proposed NETA method is simple and elegant, aligning perfectly with Occam's razor. The design decisions—such as layer-wise normalization (rather than model-wide), composite input grouping (Group 0), noise damping stabilizer ($\beta$), and scale compensation ($\gamma^l$)—are logically derived and physically well-justified.

### 4. High Presentation Quality
The writing is exceptionally polished, clear, and easy to follow. The mathematical notation is clean, and the narrative flow is highly cohesive.

---

## Weaknesses

### 1. Limited, Toy-Scale Visual Suite
The paper evaluates NETA solely across four visual classification datasets: **MNIST, FashionMNIST, CIFAR-10, and SVHN**. In the model merging literature, standard benchmarks using the CLIP ViT backbone typically evaluate on a diverse **8-dataset visual suite** representing a wider variety of specialized domain shifts, higher resolutions, and distinct tasks (including CIFAR-100, GTSRB, RESISC45, DTD, and EuroSAT). Restricting the evaluation to these four relatively simple, low-resolution datasets limits the generalizability of the findings.

### 2. Sub-Sampling of Evaluation Sets
All evaluations are performed on a subset of 1024 randomly sampled test images from each dataset.
*   **Empirical Concern**: A subset of 1024 images represents a very small sample size. At this scale, 1 image corresponds to $\approx 0.098\%$ classification accuracy. The reported improvement of NETA over Task Arithmetic on MNIST is $+0.26\%$, which equates to **fewer than 3 images** out of 1024. Such tiny margins are easily subject to subset selection bias and random noise.
*   **Lack of Justification**: The authors justify sub-sampling as a practical constraint "to manage the computational overhead... under strict Slurm queue limits." However, CLIP ViT-B/32 is an extremely lightweight model, and running zero-shot visual classification on 10,000 images takes less than a minute on a single GPU. Evaluating on the full test sets would have drastically enhanced the statistical power and credibility of the quantitative results.

### 3. Empirical Inconsistency in Reported Standard Deviations
There is a significant mathematical anomaly in the reported standard deviations of Table 1:
*   **Task-Wise AdaMerging** on FashionMNIST reports **$77.54\% \pm 0.00\%$** across 3 seeds.
*   **Layer-Wise AdaMerging** on MNIST reports **$98.44\% \pm 0.00\%$** across 3 seeds.
*   Meanwhile, **standard Task Arithmetic** reports non-zero standard deviations (**$96.03\% \pm 0.26\%$** on MNIST and **$82.10\% \pm 0.64\%$** on FashionMNIST).

The authors explain that prediction entropy gradients drive the coefficients to exact clamping boundaries, which—combined with a discretized test set of 1024 images—leads to identical classification counts. However, this explanation contains a major logical loophole:
1.  If the three seeds represent trials with **different fine-tuned expert checkpoints**, then even if the merging coefficients are clamped to identical boundaries, the underlying expert weights themselves are different. Merging different expert weights must produce different merged weights, leading to variance in test accuracy.
2.  If the three seeds represent trials with **different sampled 1024-image test subsets**, then evaluating even a completely identical model on different subsets must yield non-zero variance.
3.  The only way standard Task Arithmetic could have non-zero standard deviations while AdaMerging has exactly $0.00\%$ is if the expert checkpoints and the test sets are completely identical across all three seeds, in which case Task Arithmetic should also have $0.00\%$ standard deviation. This logical contradiction suggests a potential bug in the evaluation script (e.g., AdaMerging inadvertently reusing the same model checkpoint or evaluating on a fixed, non-random test subset).

### 4. Lack of Architecture Diversity
The empirical evaluation is restricted to a single model backbone: CLIP ViT-B/32. To demonstrate generalizability, the method should be evaluated on larger Vision Transformers (e.g., ViT-L/14) or other model families, such as Large Language Models (LLMs) or text encoders, where model merging is also widely popular.

---

## Detailed Comments & Questions for the Authors

1.  **Clarification of Seeds**: Could the authors precisely define what the "three independent random seeds" control? If they control the selection of the 1024-image test subset, why does AdaMerging achieve exactly $0.00\%$ standard deviation while Task Arithmetic does not? If they control the expert checkpoint fine-tuning, how can a merged model with different expert weights yield identical classification counts?
2.  **Baseline Tuning**: How were TIES-Merging and DARE global scaling factors optimized? Evaluating them using a fixed default $\lambda_0 = 0.30$ may significantly underrepresent their capabilities.
3.  **The Small Update Inflation Risk**: While the noise-damping stabilizer $\beta$ prevents division by zero, NETA still scales up extremely small task updates. If a task has a very small update (e.g., representing a layer that required minimal downstream adaptation), does scaling its update up to equal the layer average introduce irrelevant representation noise or cause negative interference?
4.  **Scaling to 8 Datasets**: Have the authors tried running NETA on the full 8-dataset visual suite? If so, does the "Overfitting-Optimizer Paradox" persist across more complex datasets like CIFAR-100 or RESISC45?

---

## Ratings

*   **Soundness**: **Good**. The proposed NETA method is mathematically rigorous, physically well-justified, and clearly explained. However, the standard deviation anomaly ($\pm 0.00\%$) and the small evaluation subset of 1024 images represent weaknesses in the empirical execution.
*   **Presentation**: **Excellent**. The paper is exceptionally well-written, highly polished, and structured logically.
*   **Significance**: **Good**. The proposed NETA method is a simple and elegant baseline. More importantly, the conceptualization of the **Overfitting-Optimizer Paradox** is highly significant and will likely influence the trajectory of test-time weight adaptation research.
*   **Originality**: **Good**. While layer-wise norm balancing is a straightforward vector normalization technique, the conceptual critique of joint entropy minimization and the derivation of scale compensation $\gamma^l$ are highly original.

---

## Overall Recommendation

**Rating: 4: Weak Accept**

**Justification**: This is a technically solid, highly transparent, and exceptionally well-written paper that advances the sub-area of multi-task model merging. The introduction of NETA is simple and elegant, and the exposing of the **Overfitting-Optimizer Paradox** of test-time prediction entropy minimization represents a major conceptual contribution that the community is highly likely to build on. However, the paper's impact is currently limited by its toy-scale evaluation suite (4 low-resolution datasets), its sub-sampled 1024-image evaluation, and a logical contradiction in the reported standard deviations of the test-time adaptation baselines. Resolving these empirical weaknesses would easily elevate this paper to a strong accept.
