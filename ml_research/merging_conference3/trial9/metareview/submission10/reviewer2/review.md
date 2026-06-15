# Peer Review Report

## Summary of the Paper
This paper proposes **Dirichlet-PAC**, a learning-theoretic framework for test-time multi-task model serving utilizing parameter-efficient experts (e.g., LoRA) on a shared frozen backbone. The authors address the challenge of serving-time data scarcity (often fewer than 64 samples per task), which causes standard temperature-only Empirical Risk Minimization (ERM) and unconstrained routing parameters to suffer from transductive noise overfitting and temperature parameter divergence. 

To solve this, the authors:
1. Model the ensembling weight vector $\boldsymbol{\alpha}_b$ directly over the probability simplex $\Delta^{K-1}$ using a Dirichlet posterior.
2. Minimize an analytical PAC-Bayesian bound derived under McAllester's theorem, utilizing the exact closed-form analytical KL divergence between Dirichlet distributions over the simplex as a complexity-control penalty.
3. Extract task coordinates via **Subspace Energy Projection (SEP)**, which performs Singular Value Decomposition (SVD) on early-layer calibration activations from a prior split to construct orthonormal projection bases.
4. Propose an unsupervised variant, **PEM-Div**, which minimizes prediction entropy while maximizing batch-averaged routing diversity.

The authors evaluate their framework in a custom-designed, synthetic 14-layer "Analytical Coordinate Sandbox (ICS)" and on a suite of physical pre-trained BERT backbones (`bert-tiny` to `bert-base-uncased`) with Multi-LoRA adapters.

---

## Overall Recommendation
**Rating: 2: Reject**

**Justification:**
While the paper is highly polished, well-structured, and mathematically sophisticated, it suffers from several severe and fundamental flaws that render its practical utility close to zero and its theoretical guarantees disconnected from its empirical results. Specifically:
1. **Real-World Underperformance:** On physical BERT models of all scales (Tiny, Mini, Medium, Base), the proposed Dirichlet-PAC and PEM-Div methods consistently **underperform** simpler, static baselines—including a standard linear average of weights (Uniform Merging) and uncalibrated routing (SABLE Norm). On BERT-Medium, for instance, the proposed method suffers from a severe **--6.67%** absolute degradation in accuracy compared to SABLE Norm and Uniform Merging. 
2. **Theory-Practice Mismatch:** The core theoretical PAC-Bayesian bounds are derived and proven to be mathematically exact under **Stochastic Expert Routing** (where queries are routed to a single expert). However, the actual experiments in both the sandbox and physical BERT models evaluate **continuous activation-space blending**. As the authors admit, because of non-linearities in downstream layers, the linear surrogate loss used in the PAC bound is mathematically mismatched with the physical blended model, meaning the "rigorous" generalization guarantees derived in the theory do not actually apply to the models evaluated.
3. **Undisclosed "Supervision" in Unsupervised SEP:** The claim that SEP is completely unsupervised is a mischaracterization. To perform SVD on task-specific representation matrices $Z_k$ in the prior split, the calibration activations must be grouped by task. This grouping requires knowing the task identity (labels) of the calibration samples.
4. **Missing Methodology for Key Variant:** The top-performing unsupervised variant, **PEM-Div**, is never mathematically formulated in the Methodology section. The reader is left to guess how this key contribution is constructed and optimized.
5. **Speculative Bloat:** Much of the mathematical density of the paper (e.g., martingale streaming extensions, Wedin-Davis perturbation under quantization) is entirely speculative and lacks any empirical validation.

Because the proposed method is mathematically disconnected from the evaluated model, mathematically incomplete regarding its best unsupervised variant, and—most importantly—**hurts classification accuracy** compared to zero-adaptation baselines on real-world networks, it fails to meet the bar for acceptance.

---

## Soundness
**Rating: Fair**

**Justification:**
- **The Stochastic vs. Continuous Gap:** The PAC-Bayesian generalization bounds are derived under the assumption that the model executes a single stochastic routing path per query. However, the actual model runs parallel expert paths and blends activations continuously. Due to non-linear downstream layers, the linear surrogate loss is not exact for the continuous ensembled model. Thus, the mathematical soundness of the generalization certificates is compromised because they do not apply to the physical model being tested.
- **Missing PEM-Div Formulation:** Leaving the core "diversity penalty" of the PEM-Div baseline undefined in Section 3 makes it impossible to theoretically evaluate the correctness of the unsupervised optimization objective.
- **Unsupervised SVD Clashing:** To perform SVD on $Z_k$ for each task $k$ during prior calibration, the prior split must be grouped by task. Grouping samples by task is a form of weak supervision. If the prior split were truly unlabeled and ungrouped, task-specific projection bases could not be constructed.

---

## Presentation
**Rating: Fair**

**Justification:**
- **Mathiness and Redundancies:** The writing style is overly defensive and overloaded with dense, speculative mathematics designed to create a sense of extreme rigor. Proposition 1 (Scale Invariance and Basis Independence) is standard linear algebra that does not require a formal proposition and proof in a machine learning paper.
- **Omission of Key Equations:** The paper completely fails to define the mathematical formula of the diversity penalty or batch-averaged weight entropy maximization for the **PEM-Div** variant in Section 3, despite PEM-Div being highlighted as a major positive result in Section 4.
- **Unvalidated Sections:** Sections 5.1 and 5.3 present dense derivations for quantization and streaming martingales, yet the paper contains absolutely zero empirical evaluations of low-bit quantization or streaming sequential adaptation. These sections are speculative bloat and should be removed.

---

## Significance
**Rating: Poor**

**Justification:**
- **No Practical Utility:** Practitioners will not deploy a framework that requires running an optimization loop on streaming queries (introducing serving-time latency and computational overhead) only to **lose up to 6.67% in classification accuracy** compared to static weight averaging or uncalibrated routing.
- **Pre-Conditioned Sandbox:** The paper's claims of "superiority" and "necessity" are based entirely on a highly artificial, synthetic 14-layer sandbox where the representational noise is explicitly hard-coded to be proportional to the routing entropy. This circular design pre-conditions the sandbox to favor the proposed entropy-regularized router. On real-world networks (BERT) where such artificial noise is absent, the proposed method collapses under its own optimization variance, losing to unregularized/uncalibrated baselines.

---

## Originality
**Rating: Fair**

**Justification:**
- The paper combines standard, well-established statistical tools (Dirichlet distribution properties, SVD, McAllester's PAC-Bayes bound) and applies them to test-time PEFT serving. While the combination is neat, it represents a straightforward application of classical PAC-Bayes theory to a simplex-constrained problem. 
- The novelty is further diminished because the most competitive unsupervised variant, PEM-Div, is mathematically undefined, and the "completely unsupervised" coordinate extraction is conceptually mischaracterized.

---

## Major Strengths
1. **Relevance and Timeliness:** Addressing test-time model serving of PEFT experts under data scarcity is an important, high-impact problem for resource-constrained edge-computing and multi-tenant cloud deployments.
2. **Expositive Structure:** The paper is well-organized, with clear section headings, clean tables, and comprehensive bibliography of model merging and dynamic serving literature.
3. **Inclusion of BERT Benchmarks:** The authors deserve credit for evaluating their method on real-world BERT backbones, which provides a transparent view of the model's actual performance (even though the results expose significant shortcomings).

---

## Major Weaknesses

### 1. Failure to Beat Simple, Static Baselines on Real Models
In Table 3, the proposed Dirichlet-PAC and PEM-Div methods are evaluated on real-world BERT backbones across four scales. In **every single scale**, the proposed methods are either tied with or significantly outperformed by simpler, zero-adaptation baselines:
- On **BERT-Medium**, SABLE Norm and Uniform Merging both achieve **96.00% ± 3.89%** accuracy. Dirichlet-PAC drops to **89.33% ± 3.89%** (a **-6.67%** degradation), and PEM-Div drops to **88.67% ± 3.40%** (a **-7.33%** degradation).
- On **BERT-Base**, Uniform Merging achieves **95.33% ± 4.52%**, while Dirichlet-PAC achieves **92.00% ± 8.59%** (a **-3.33%** degradation).
- The authors' defense that physical BERT tasks represent a "clean, orthogonal" space where static priors are already optimal is highly unconvincing. They never evaluate BERT on overlapping or noisy configurations. Thus, they have **zero empirical evidence** that their method is superior on real models under *any* circumstances.

### 2. Disconnect Between PAC Theory and Continuous Activation Blending
The mathematical core of the paper derives an analytical PAC-Bayesian bound under the assumption of Stochastic Expert Routing (Equation 14). However, the actual ensembling model evaluated in the experiments is Continuous Activation-Space Blending (Equation 5). 
Because of the non-linearities in downstream blocks and classification heads, the expected loss of the continuous blended model is not equal to the linear surrogate loss used in the PAC bound. Therefore, the "mathematically rigorous, watertight" generalization guarantees derived in Section 3 **do not apply** to the actual models being evaluated. The theoretical framework is disconnected from the empirical system.

### 3. Mischaracterization of "Unsupervised" Coordinate Extraction
The authors repeatedly market Subspace Energy Projection (SEP) as "completely unsupervised and task-agnostic." However, Section 3.2 reveals that learning the projection basis $V_{k, d}$ requires constructing a task-specific representation matrix $Z_k$ from the prior split $\mathcal{S}_{\text{prior}}$. Separating the activations into $Z_1, \dots, Z_K$ requires knowing which sample belongs to which task, which is a form of task supervision. True unsupervised serving implies that the prior split is a random, unlabeled stream where task associations are unknown. Under true unsupervised conditions, SEP cannot be executed.

### 4. Mathematical Omission of the Best-Performing Unsupervised Variant
Throughout Section 4, the unsupervised variant **PEM-Div** is praised for its "stunning victory," outperforming its supervised counterpart and achieving state-of-the-art results. However, **the mathematical formula for PEM-Div is completely missing from Section 3.** The methodology only defines basic PEM (Equation 13). The "batch-averaged ensembling weight entropy maximization" or "diversity penalty" is never defined, leaving a major, top-performing proposed variant mathematically unverifiable and completely irreproducible.

### 5. Omission of Standard Regularization Baselines (Strawman Evaluations)
The paper justifies the highly complex, mathematically dense PAC-Bayesian bound by showing it outperforms unregularized Temp-Only ERM. However, the standard engineering choice to prevent temperature parameter collapse is simple **L2 regularization (weight decay)** or **L1 regularization** on the log-temperature parameters $\mathbf{w}$. The authors completely omit these obvious regularized baselines, comparing their complex framework only against a completely unregularized "strawman" baseline.

### 6. Speculative "Mathiness" and Bloat
Sections 5.1 (Wedin-Davis perturbation under quantization) and 5.3 (sequential streaming via Azuma-Hoeffding martingales) occupy significant space and introduce dense notations, but they are **entirely speculative** and have no corresponding empirical validations. They serve merely to artificially inflate the mathematical density of the paper and should be discarded.

---

## Constructive Feedback & Questions for the Authors

1. **Address Real-World Underperformance:** Can you explain why a practitioner should adopt a complex framework requiring serving-time optimization when it results in a **-6.67%** drop in accuracy on BERT-Medium and a **-3.33%** drop on BERT-Base compared to simply averaging the weights?
2. **Provide the Mathematical Formula for PEM-Div:** Please formally define the optimization objective and diversity penalty for PEM-Div in Section 3, including how it is integrated into the PAC-Bayesian bound and what hyperparameter scales it.
3. **Resolve the Theory-Practice Gap:** To claim "watertight generalization guarantees" for the evaluated models, you must either:
   - Derive a PAC-Bayesian bound that directly accounts for the non-linearities of continuous activation blending, OR
   - Transition your empirical evaluations (Tables 1, 2, and 3) to Stochastic Expert Routing to match your current derived bounds.
4. **Compare Against Simple Regularization:** Please include baseline results for Temp-Only ERM regularized by standard L2 weight decay on the log-temperatures $\mathbf{w}$. You must demonstrate that the complex PAC-Bayesian penalty provides a statistically significant improvement over simple weight decay to justify its high complexity.
5. **Evaluate Real Models Under Overlap/Noise:** Since you argue that the BERT results represent a clean, orthogonal space where static priors are already optimal, you must construct a BERT multi-task benchmark with overlapping manifolds or controlled representational interference to empirically prove that your method is superior to static weight averaging on physical networks under noisy conditions.
6. **Remove Speculative Sections:** Please remove or move Sections 5.1 and 5.3 to an appendix, as they have zero empirical validation in the paper and contribute solely to "mathiness" and presentation bloat.
