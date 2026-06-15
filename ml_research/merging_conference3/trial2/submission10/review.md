# Peer Review Report

**Title**: Deconstructing Adaptive Model Merging: Exposing the Overfitting-Optimizer and Spatial Averaging Paradoxes  
**Confidential Recommendation**: Accept (Score: 5)

---

## 1. Summary of the Paper
This paper presents a highly rigorous, deconstructive study of **AdaMerging** (Yang et al., 2024), a prominent unsupervised test-time model merging framework. AdaMerging utilizes Shannon prediction entropy minimization on a small, unlabeled test-time calibration batch to optimize layer-wise or task-wise merging coefficients.

Adopting a critical "Occam's razor" lens, the authors deconstruct the adaptive merging process and expose two fundamental optimization anomalies:
1. **The Overfitting-Optimizer Paradox**: While SOTA layer-wise AdaMerging ($\approx 1000$ parameters) is prone to transductive test-time overfitting, the authors show that per-layer scaling coefficients are **structurally specialized** to the network's architectural hierarchy. Applying *Intra-Task Layer Shuffling* (randomly permuting coefficients across layers) collapses performance (from $88.05\%$ down to $78.61\%$), confirming that early layers (capturing general visual features) and late layers (capturing specialized task features) require specialized scales. Shuffling violates this hierarchy, thereby destroying representational routing. However, replacing these coefficients with their flat *Spatial Average* (mean) per task reduces parameters to $T$ scalars, acting as a spatial low-pass filter that smooths away individual test-time overfitting while successfully retaining the core task-level scales and outperforming standard Task Arithmetic ($84.96\%$ vs $84.64\%$), albeit with a $3.09\%$ trade-off compared to the unconstrained layer-wise model ($88.05\%$).
2. **The Spatial Averaging Paradox**: The authors show that while *post-hoc Spatial Averaging* succeeds ($84.96\%$), *direct* test-time optimization of these flat task-wise scales (Task-wise AdaMerging) fails spectacularly, collapsing performance to $81.19\%$ (below its uniform initialization of $84.64\%$). They explain this via **multi-task gradient imbalance**: prediction entropy is uncalibrated and highly sensitive to logit-scaling on easy tasks (e.g., MNIST). Under low-dimensional global constraints, easy-task gradients dominate the shared optimization, scaling up their coefficients and creating destructive parameter interference on harder tasks (e.g., CIFAR-10 and SVHN). High-dimensional optimization avoids this through local layer degrees of freedom, which post-hoc spatial averaging then regularizes.

To address the gradient imbalance, the authors propose and evaluate **Calibrated Prediction Entropy**, which normalizes task contributions. They demonstrate that while this successfully calibrates the gradients at initialization, direct flat optimization still fails ($80.59\%$), proving that low-dimensional bottlenecks are fundamentally incompatible with prediction entropy minimization in weight space.

---

## 2. Strengths of the Paper
* **Philosophical Focus on Occam's Razor**: This paper represents a breath of fresh air in the model merging literature, which has recently become dominated by increasingly convoluted, parameter-heavy adaptive schemes. By systematically stripping away unnecessary degrees of freedom and using simple diagnostic controls, the authors expose the true underlying dynamics of weight-space adaptation.
* **Nuanced and Scientifically Balanced Framing**: The authors adopt an exceptionally mature and scientifically accurate framing of the "Overfitting-Optimizer Paradox". They recognize that layer-wise coefficients capture *both* a beneficial, position-dependent representational routing signal (which explains why shuffling collapses performance and why spatial averaging drops accuracy by 3.09% compared to the SOTA model) and a transductive test-time overfitting component. This resolves previous logical contradictions and provides a cohesive, highly consistent narrative.
* **Fascinating Discovery of the Spatial Averaging Paradox**: The finding that *indirect* optimization (optimizing layer-wise scales and averaging them) succeeds where *direct* optimization (optimizing global scales directly) fails is highly intriguing, counter-intuitive, and represents a significant conceptual contribution to deep learning representation theory.
* **Sound Theoretical Explanation of Gradient Imbalance**: Explaining the Spatial Averaging Paradox through multi-task gradient imbalance caused by uncalibrated prediction entropy and logit-scaling characteristics is mathematically elegant and empirically well-supported.
* **Exceptional Empirical Scale and Rigor**: The authors have systematically addressed previous scale limitations by expanding their final evaluations to the **full standard test sets** of MNIST, FashionMNIST, CIFAR-10, and SVHN (56,032 images total across tasks). This dual-scale protocol (adaptation on 64 images, evaluation on full test sets) is exceptionally elegant, yields extremely tight confidence intervals, and completely resolves previous statistical variance on SVHN.
* **Excellent Baseline Comparison**: The inclusion of both standard **Task Arithmetic** ($84.64\%$) and **TIES-Merging** ($77.54\%$) in Table 1 provides an excellent and realistic baseline context, proving that static sign-consensus heuristics can collapse on highly diverse task configurations while post-hoc Spatial Averaging remains robust.
* **Excellent CKA Deconstruction**: Dissecting Linear CKA similarity to show that high CKA is a baseline property of small scaling factors rather than an indicator of functional adaptation is an outstanding, highly insightful, and much-needed empirical correction for the model merging community.
* **Layer-by-Layer CKA Plot**: The addition of Figure 4 (layer-by-layer CKA plot across all 12 blocks) beautifully and visually substantiates the hierarchical routing hypothesis, showing early layers remaining highly aligned while late layers specialize.
* **Superb Writing and Structure**: The paper is exceptionally well-structured, easy to read, and uses a very cohesive and clear vocabulary.

---

## 3. Minor Areas for Improvement (Constructive Suggestions)
Since the paper is technically solid, empirically rigorous, and highly complete, we recommend Accept and offer the following minor suggestions for future versions:

### Minor Suggestion 1: Resolve Text vs. Execution Mismatch in Dataset Splits
In Section 4.1 (Dataset Splits), the text states:
> *"Evaluation Split: 512 labeled images per task, used as the unseen test set for computing final classification accuracies."*

However, in the actual implementation and reported accuracies in Table 1, the merged models are evaluated on the **full, standard test splits** of all four datasets (MNIST: 10k, FashionMNIST: 10k, CIFAR-10: 10k, SVHN: 26,032; total 56,032 images evaluated per method).
* *Action*: Update Section 4.1 to clarify that while the 512-image split is used for fast sweeps (e.g., noise sweeps), the primary accuracies in Table 1 are computed on the full standard test sets. This highlights an enormous empirical strength of the paper.

### Minor Suggestion 2: Comparison with Additional Static Baselines
While Table 1 is already very comprehensive, including other recent static baseline methods like **DARE** (Yu et al., 2024) or showing a complete sweep over static scaling factors for Task Arithmetic (beyond the standard $0.3$) would provide an even more complete empirical comparison. This would demonstrate whether any static scale can match the post-hoc spatial mean.

### Minor Suggestion 3: Generalization to Other Backbones
To ensure the findings are not specific to the ViT-B/32 architecture, evaluating on Swin Transformers or modern convolutional architectures (like ConvNeXt) would prove that the Overfitting-Optimizer and Spatial Averaging Paradoxes generalize across neural network families.

### Minor Suggestion 4: Extension to Generative Language Models
While the authors discuss LLMs in the future work section, outlining how prediction entropy could be adapted (e.g., token-level perplexity or generation entropy) would provide a very clear roadmap for researchers seeking to extend these paradoxes to generative foundation models.

---

## 4. Ratings on Specific Dimensions

### Soundness: Excellent
The theoretical explanation of the Spatial Averaging Paradox via multi-task gradient imbalance is highly sound, logical, and empirically supported. The narrative around the Overfitting-Optimizer Paradox is balanced and scientifically accurate, and the evaluation on the full standard test sets (56,032 images total) provides exceptional statistical significance.

### Presentation: Excellent
The paper is exceptionally well-written, clearly structured, and easy to follow. Mathematical formulations are precise, terms are well-defined, and the authors include thorough reproducibility details.

### Significance: Excellent
This paper addresses an important and timely problem. It warns the community about the pitfalls of using unconstrained prediction entropy minimization as an unsupervised surrogate objective for weight-space optimization, and highlights the dangers of global low-dimensional bottlenecks. These insights have the potential to redirect future research toward more robust, calibrated weight-space combinations.

### Originality: Excellent
The discovery of the Spatial Averaging Paradox, the design of shuffling and averaging diagnostic treatments, and the critical deconstruction of CKA representation similarity are highly original, creative, and provide valuable new perspectives.

---

## 5. Questions for the Authors
1. **Symmetric Scaling and Pruning**: How do the authors expect post-hoc spatial averaging to perform when applied on top of pruned task vectors, such as in TIES-Merging or DARE? Would the spatial mean still act as a low-pass filter on pruned weight vectors?
2. **Alternative Surrogate Losses**: Since prediction entropy minimization is prone to pathological logit-scaling shortcuts in low dimensions, have the authors considered alternative unsupervised test-time objectives, such as self-supervised contrastive losses, to stabilize flat task-wise optimization?
3. **The Ultimate Minimalist Merging**: Since prediction entropy minimization requires an optimization loop over hundreds of iterations (which introduces computational overhead at test-time), are there non-iterative, statistics-based ways to compute the task-specific scales directly from the task vectors $\tau_t$ (e.g., using weight variances or norm ratios)?

---

## 6. Suggestions for Improvement and Actionable Feedback
1. **Fix Inconsistency in Section 4.1**: Update the text describing the evaluation split in Section 4.1 to match the actual execution on the full standard test sets (total 56,032 images).
2. **Future Work on Alternative Losses**: Add a brief discussion in Section 5 on whether self-supervised loss functions (e.g., contrastive or mask-reconstruction objectives) could bypass the uncalibrated nature of prediction entropy.
