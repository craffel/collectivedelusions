# Peer Review

## Summary of the Paper
The paper introduces **GranMerge**, a systematic empirical framework that deconstructs the **Generalization-Granularity Trade-off** in adaptive test-time model merging. Model merging is an appealing paradigm for combining multiple task-specific expert neural networks (sharing a pre-trained base) into a single unified model without the high cost of full retraining. Recent works have focused on test-time adaptive model merging (e.g., AdaMerging), which optimizes merging coefficients on a small, unlabeled calibration stream at deployment time. 

GranMerge systematically evaluates five nested levels of parameter resolution—Global (Level 1), Layer-wise (Level 2), Block-wise (Level 3), Component-wise (Level 4), and Tensor-wise (Level 5)—under two optimizer families (first-order Adam and zero-order 1+1 Evolution Strategies). It also proposes two soft spatial-depth regularizers: Elastic Spatial Regularization (ESR) and Total Variation (TV) depth-wise smoothness. 

Through exhaustive multi-seed sweeps on a 12-layer Vision Transformer across four classification tasks (MNIST, FashionMNIST, CIFAR-10, SVHN) in a resource-constrained setting, the paper reveals that:
1. **Granularity-Generalization Trade-off**: High structural granularity (Level 5 Tensor-wise) leads to severe transductive overfitting on small test-time calibration streams ($N=256$).
2. **Optimizer-Specific Dynamics**: First-order Adam is highly vulnerable to rapid, chaotic generalization collapse by aggressively minimizing prediction entropy. Meanwhile, zero-order 1+1 ES exhibits higher test-set generalization robustness. The authors deconstruct this robustness, demonstrating that it is heavily driven by **optimization sluggishness** (failure to optimize in high dimensions) under the curse of dimensionality, which keeps coefficients near the robust uniform initialization.
3. **Regularization and Baseline Supremacy**: While soft spatial-depth regularizers (ESR + TV) stabilize both optimizers at Level 5, **no test-time adaptive configuration outperforms the static, zero-overhead Uniform Task Arithmetic baseline of 30.41%**. This is attributed to a fundamental misalignment between the unsupervised prediction entropy surrogate loss and true classification accuracy on compact calibration batches.

---

## Strengths and Weaknesses

### Strengths
* **High Practical Utility and Honest Reporting**: The paper reports an extremely valuable negative result: that unconstrained, fine-grained test-time adaptive merging actually degrades performance, and that none of the adaptive configurations outperform a simple, zero-overhead static Uniform Task Arithmetic baseline. For practitioners looking to deploy multi-task networks on resource-constrained or edge systems, this is an incredibly high-signal finding that prevents over-engineered, fragile, and computationally expensive test-time optimization loops.
* **Brilliant Deconstruction of ES Robustness**: The analysis of zero-order optimization dynamics in high dimensions is exceptional. Rather than lazily attributing ES's higher generalization to "implicit noise regularization," the authors scientifically demonstrate that it is a byproduct of optimization sluggishness (underfitting) under the curse of dimensionality. This prevents the optimizer from moving far from the initialization (the robust uniform baseline scale), thereby preserving the base representations. This intellectual honesty is highly refreshing.
* **Unified, Systematic Taxonomy**: Unifying fragmented prior works (such as task arithmetic, AdaMerging, and SplineMerge) into a single nested hierarchy of five structural granularities is a strong conceptual contribution that simplifies future exploration.
* **Highly Reproducible and Transparent**: The paper and its appendix are exceptionally detailed. Providing exact architectural details of the custom `ViTTiny`, precise pre-training and fine-tuning protocols, test-time adaptation hyperparameters, and standard deviations across multiple independent seeds ensures outstanding reproducibility.

### Weaknesses
* **Extremely Weak, Poorly Converged Experts**: A major limitation is that the task-specific experts being merged are extremely weak. As shown in Table 1 (and the Appendix), the upper bound "Individual Experts" mean accuracy is only 41.48% (MNIST at 61.03%, FashionMNIST at 62.47%, CIFAR-10 at 24.93%, and SVHN at 17.50%). For 10-class datasets, CIFAR-10 and SVHN are barely functioning above random chance (10%). Merging experts that are poorly trained introduces massive high-frequency parameter noise into the task vectors $\theta_k$. This noise is highly likely to amplify vulnerability to transductive overfitting. In modern production environments, practitioners would never deploy or merge experts with such low performance. 
* **Toy-Scale Backbones and Small Datasets**: The empirical evaluation is restricted to an extremely compact `ViTTiny` backbone ($d_{\text{model}}=64$, 2 heads, 12 blocks) on four small-scale, simple visual classification datasets (MNIST, FashionMNIST, CIFAR-10, SVHN). While this is a useful edge-device stress test, it makes it difficult to assess whether these granularity-generalization dynamics scale to larger, industry-relevant backbones (e.g., standard ViT-Base/Large, ResNet-50, or small LLMs) where representation manifolds are highly structured and task vectors are significantly cleaner.
* **Surrogate Loss Limitations**: The paper relies on unsupervised prediction entropy as the surrogate objective for adaptation. Since prediction entropy is heavily prone to "confident but incorrect" decision boundary shifts under transductive noise, the severe overfitting at fine granularities is highly tied to this specific loss. The paper would be much stronger if it explored semantically richer unsupervised objectives (such as contrastive or self-supervised losses) that better preserve representation integrity.

---

## Soundness (Rating: Good)
The submission is technically sound. The formulation of task vectors, structural partitioning of coefficients, the unsupervised surrogate objective, and the optimization updates are mathematically correct and clearly explained. The experimental design is thorough, using multiple seeds, clear baselines, and appropriate statistical controls. 

The rating of "Good" (rather than "Excellent") is due to the weak quality of the underlying experts (CIFAR-10 and SVHN experts operating near random guess), which represents a significant experimental gap. Since the experts are poorly trained, the conclusions regarding transductive overfitting may be over-amplified by parameter noise, limiting how confidently we can generalize these findings to high-performance production models.

---

## Presentation (Rating: Excellent)
The paper is exceptionally well-written, clearly structured, and easy to follow. The mathematical notation is clean and consistent. The diagrams and tables are highly legible, and the appendix is extraordinarily comprehensive, providing everything required for an expert reader to replicate the results. The authors are incredibly careful, objective, and honest about evaluating both the strengths and weaknesses of their work, which is a hallmark of high-quality scientific writing.

---

## Significance (Rating: Good)
The paper addresses an important and highly active area of machine learning. By mapping out the Generalization-Granularity Trade-off, it provides a crucial cautionary tale for practitioners who might otherwise spend significant computational and engineering overhead on complex, high-dimensional test-time adaptation. Proving that a zero-overhead static uniform weight blend remains supreme over adaptive methods is a highly impactful, practical result.

However, the significance is slightly bounded by the toy-scale setting (ViT-Tiny on MNIST/CIFAR-10). To achieve a higher significance rating, the paper needs to demonstrate whether these dynamics hold in high-fidelity, production-grade regimes with fully-converged foundation models.

---

## Originality (Rating: Good)
The work provides new insights into the structural boundaries of weight blending. While the individual components (ViTs, Adam, 1+1 ES, L2 regularization) are standard, the unification of structural granularity into a 5-level hierarchy and the deep, mathematically-grounded analysis of first-order vs. zero-order overfitting dynamics are highly original. The deconstruction of ES robustness as "optimization sluggishness" is particularly clever and novel.

---

## Questions and Constructive Suggestions for the Authors
1. **Validation on Fully-Converged Experts**: Could the authors provide results when merging fully converged task-specific experts (e.g., standard ViT-Tiny fine-tuned to 98% on MNIST, 90% on FashionMNIST, 75% on CIFAR-10, and 90% on SVHN)? It is crucial to determine if the transductive overfitting at higher granularities is a fundamental law of model merging, or if it is primarily driven by the noisy parameter spaces of under-trained, low-fidelity experts.
2. **Evaluation on Standard Backbones**: To demonstrate real-world scalability, are there plans to evaluate GranMerge on at least one standard-sized backbone (such as a standard ViT-Base or a ResNet-50)? Even a single experiment on standard-sized networks would greatly enhance the paper's significance.
3. **Exploring Alternative Surrogate Losses**: Given the severe misalignment of prediction entropy (which leads to "confident but incorrect" predictions), have the authors considered evaluating other unsupervised test-time losses? For example, minimizing Centered Kernel Alignment (CKA) drift or using a self-supervised contrastive loss across augmented views might offer a cleaner, representation-level anchor that prevents transductive overfitting at fine structural granularities.

---

## Overall Recommendation
**Score: 4: Weak Accept**

### Justification
The paper is a technically solid, exceptionally well-written, and highly reproducible study that addresses a critical open question in model merging. Its primary strength lies in its intellectual honesty and high practical utility: by systematically mapping the structural spectrum and reporting that zero-overhead static uniform blends remain superior to complex test-time adaptive merging, it provides clear, actionable guidelines that protect engineers from over-engineered deployments. The mathematical and conceptual deconstruction of 1+1 ES's apparent robustness as "optimization sluggishness" is outstanding. 

The recommendation is a "Weak Accept" rather than a full "Accept" solely due to the limited, low-fidelity experimental scale. The use of a highly compact toy model (`ViTTiny`, $d_{\text{model}}=64$) coupled with extremely weak, poorly converged downstream experts (such as the CIFAR-10 and SVHN experts, which perform close to random guessing) leaves an experimental gap. It remains unclear if the identified trade-off curve and the severity of transductive overfitting hold when merging high-fidelity, fully converged production models. If the authors can address this concern by providing even a small-scale validation on well-trained experts or standard backbones, this paper would easily deserve a higher score.
