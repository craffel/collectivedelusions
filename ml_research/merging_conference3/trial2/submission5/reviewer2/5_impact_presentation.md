# Presentation, Impact, and Significance Evaluation

## Major Strengths

### 1. High Conceptual Signal & Exposing a Critical Failure Mode
The paper's strongest contribution is the conceptualization and empirical exposing of the **"Overfitting-Optimizer Paradox"** in test-time adaptation. Demonstrating that prediction entropy minimization inherently suppresses harder, high-entropy tasks under difficulty imbalances is a highly valuable, high-signal finding. It directly challenges the current popularity of unconstrained joint prediction entropy minimization in the model merging literature and will likely prompt researchers to develop more robust, balance-aware adaptation objectives.

### 2. High Scientific Integrity and Transparency
The authors exhibit exceptional scientific honesty. Rather than hiding unfavorable results, they:
*   Acknowledge that NETA's isotropic regularization slightly reduces SVHN performance to achieve representation fairness.
*   Conduct a full grid search over $\lambda_0$ (Table 3), transparently reporting that when both methods are fully tuned to peak performance, standard Task Arithmetic remains slightly superior in average accuracy ($89.16\%$ vs. $89.06\%$).
*   Clearly discuss and mathematically qualify the "directional norm contraction" of merged updates and introduce a closed-form scale-compensation factor $\gamma^l$ to address it.
This level of rigor and transparency is exemplary and extremely refreshing.

### 3. Clear, Intuitive, and Physically-Grounded Design
The proposed NETA method is elegant in its simplicity, aligning perfectly with Occam's razor. The design decisions—layer-wise normalization, composite input grouping (Group 0), noise damping stabilizer ($\beta$), and scale compensation ($\gamma^l$)—are all logically derived from physical and geometric heuristics of deep network weight spaces.

### 4. High Writing Quality & Structure
The paper is exceptionally well-written. The narrative is cohesive, the arguments are well-developed, and the mathematical derivations are clear and easy to follow.

---

## Areas for Improvement

### 1. Scale Up the Experimental Evaluation
The empirical evaluation is limited in two major ways:
*   **Dataset Diversity**: Evaluating only on MNIST, FashionMNIST, CIFAR-10, and SVHN is a toy-scale visual suite. To be fully convincing to the model merging community, NETA should be evaluated on the standard **8-dataset visual suite** commonly used with CLIP backbones (adding CIFAR-100, GTSRB, RESISC45, DTD, and EuroSAT).
*   **Full Test Set Evaluation**: The sub-sampling of test sets to 1024 images is highly non-standard and introduces sampling noise. The paper should evaluate on the full test sets of these datasets. Since CLIP ViT-B/32 is very lightweight, this would run extremely quickly and eliminate any subset-sampling bias.

### 2. Address the Standard Deviation Discrepancy (0.00%)
As detailed in the Soundness evaluation, the report of exactly **$0.00\%$ standard deviation** for certain tasks under AdaMerging is mathematically inconsistent with standard Task Arithmetic's non-zero variance. The authors must clarify:
*   How the "three independent random seeds" are defined and applied (Are they used to sample the test set? To initialize AdaMerging? To fine-tune the expert checkpoints?).
*   Why different random trials lead to zero variance in some methods but non-zero variance in others.
This inconsistency should be thoroughly investigated and resolved, as it raises concerns about potential evaluation script bugs.

### 3. Generalize to Other Architectures
The paper currently evaluates NETA solely on a single architecture: CLIP ViT-B/32. To demonstrate generalizability, the method should be evaluated on larger Vision Transformers (e.g., ViT-L/14) or other model families, such as Large Language Models (LLMs) or text encoders, where model merging is also highly popular.

---

## Overall Presentation Quality
The presentation quality is **excellent**. The paper is highly polished, professional, and structured logically. It clearly positions NETA relative to prior work (Section 2), provides rigorous formulations and pseudocode (Section 3), details empirical findings with clear tables and figures (Section 4), and ends with an honest discussion of limitations and future directions (Section 5).

---

## Potential Impact and Significance
The paper has **moderate-to-high significance**:
*   As a method, **NETA** itself provides an elegant, simple, and training-free baseline that achieves extremely competitive performance entirely zero-shot, satisfying Occam's razor.
*   As a critique, the **Overfitting-Optimizer Paradox** is highly significant and will likely influence the trajectory of test-time weight adaptation research. It provides a valuable warning to the community about the hidden dangers of unsupervised proxy objectives like joint entropy minimization under task difficulty imbalances.
