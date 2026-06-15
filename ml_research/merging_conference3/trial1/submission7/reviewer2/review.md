# Peer Review Report

---

## 1. Paper Summary
This paper presents a rigorous methodology-focused sanity check and representational analysis of the core assumption underlying state-of-the-art (SOTA) layer-wise model merging frameworks, such as AdaMerging and SyMerge. These frameworks claim that optimizing fine-grained, layer-by-layer, task-wise merging coefficients is critical to resolving localized representational conflicts when merging diverse task-specific experts into a single backbone.

To test this assumption, the authors establish a **Sanity-Checking and Interpretability Suite** on a pre-trained **CLIP ViT-B/32** backbone across four diverse image classification tasks (MNIST, FashionMNIST, CIFAR-10, SVHN) using **3 independent random seeds** with unique, disjoint data splits. They optimize $L \times K = 52$ merging coefficients under a joint prediction entropy minimization objective using two distinct optimizers: a zero-order Adaptive 1+1 Evolution Strategy (1+1 ES) and a first-order Adam Gradient Descent (Adam GD). They design three rigorous control treatments on the learned parameters: (1) **Intra-Task Layer Shuffling**, (2) **Task-Wise Spatial Averaging** (collapsing the 52 layer-wise parameters into just 4 task-wise scalars, a 92.3% reduction), and (3) **Norm-Bounded Perturbation** (injecting relative Gaussian noise up to 50%). Additionally, they perform representational similarity analysis using linear Centered Kernel Alignment (CKA) to evaluate activation-level alignment with original experts.

Their empirical analysis exposes a profound, dialectical interaction termed the **Overfitting-Optimizer Paradox**:
1. **Under Zero-Order Optimization (1+1 ES):** Layer-specificity is an illusion. Replacing learned coefficients with their flat Spatial Mean actually *improves* average accuracy from $85.07 \pm 0.47\%$ to $85.21 \pm 0.11\%$ while reducing cross-seed variance, proving that Spatial Averaging acts as a spatial regularizer that smooths out high-frequency zero-order mutation noise. Shuffling coefficients causes minimal performance decay.
2. **Under First-Order Optimization (Adam GD):** Unconstrained optimization finds a highly precise, delicate configuration of layer coefficients that is extremely sensitive to shuffling (collapsing performance by 5.43% overall and 15.69% on CIFAR-10) or spatial averaging (collapsing CIFAR-10 by 10.35%), creating an illusion of physical layer-specificity. However, this delicate structure is a transductive overfitting artifact on the small calibration set: the optimized Adam GD model ($84.52 \pm 1.57\%$) fails to outperform the unoptimized Task Arithmetic baseline ($84.44 \pm 0.37\%$) on unseen test data while introducing 4x greater seed variance.
3. **Landscape Flatness:** Both optimizers reside in an exceptionally flat loss basin, tolerating up to 50% relative Gaussian noise on the coefficients with negligible decay.
4. **CKA-Accuracy Decoupling:** Spatially averaged models exhibit slightly higher activation similarity to original experts on average, but high-level linear CKA is a poor predictor of downstream classification accuracy under gradient descent (e.g., CIFAR-10 accuracy collapses by 10.35% under averaging despite maintaining $>0.95$ CKA), highlighting that weight-space decision boundary integrity decouples from activation-level alignment.
5. **Joint Entropy Bias & Solutions:** Mathematically and empirically exposes how joint entropy objectives sacrifice high-entropy complex tasks (like SVHN) to minimize simple, low-entropy tasks. They propose and validate **Proximity Regularization** (to prevent transductive drift) and **Scale-Normalized Weighted Joint Entropy** (to resolve task-bias), both proven highly effective in Appendices B, E, and F.

---

## 2. Strengths and Weaknesses

### Strengths
- **Exemplary Literature Contextualization and Attribution:** The submission does an outstanding job of situating itself within the historical lineage of model merging and deep neural network representational analysis. It traces the progression of model merging from linear mode connectivity (e.g., Model Soups, Robust Fine-Tuning) to task vectors (Task Arithmetic) and fine-grained layer-wise adaptive parameters (AdaMerging, SyMerge). It properly attributes ideas and accurately describes the current landscape of test-time adaptation (Tent, transductive overfitting) and representational similarity (linear CKA, CCA, SVCCA).
- **Outstanding Technical and Scientific Rigor:** The experimental design is an exemplar of scientific methodology. Using permutation shuffling, spatial averaging, and relative noise sweeps as control treatments is extremely elegant. Deploying both a zero-order derivative-free optimizer and a first-order autograd-based optimizer allows the authors to brilliantly isolate optimizer-dependent confounding. Evaluating across 3 independent random seeds with unique, disjoint data splits guarantees statistical consistency and eliminates seed-cherry-picking.
- **Deep Conceptual and Empirical Insights:** The conceptual formulation of the **Overfitting-Optimizer Paradox** is highly original and profound. Explaining how zero-order search noise is smoothed out by spatial averaging, whereas first-order autograd discovers a highly delicate but ungeneralizable transductive overfitting configuration on small calibration sets, is a brilliant contribution.
- **Representational Discrepancy Warning:** The empirical exposure of the decoupling between activation-level CKA and downstream top-1 accuracy provides a critical, highly valuable caution to both the interpretability and model-merging communities.
- **Constructive Solutions and Empirical Verification:** Rather than presenting a purely negative critique, the authors propose concrete, elegant, and highly effective solutions:
  - *Proximity Regularization:* Augmenting the calibration loss with an $L_2$ proximity penalty to restrict parameters from drifting excessively from their stable baseline (validated in Appendix B & F).
  - *Scale-Normalized Weighted Joint Entropy:* Weighting task prediction entropy by the inverse of its baseline uniform task arithmetic entropy to resolve task-bias (validated in Appendix E).
- **Intellectual Maturity and Honesty:** The paper maintains an objective, constructive tone. In Section 5, it thoughtfully discusses its limitations, noting that in larger-scale autoregressive decoder LLMs (7B+ parameters) or highly complex downstream domains (such as instruction-tuning, cross-modal tasks, or medical imaging), layer-wise weight coordinate adjustments may possess genuine, physical importance due to distinct structural specialization (syntax in early layers, facts in middle layers, generation in late layers; Attention vs. MLP blocks).

### Weaknesses
- **Main-Text Integration of Solutions:** The proposed solutions (Proximity Regularization and Scale-Normalized Joint Entropy) are highly promising and thoroughly validated in the Appendices (Appendix B, E, F). Moving a concise summary or a small table of these results directly into the main text (e.g., within Section 4) would greatly strengthen the paper's constructive contribution and offer immediate, practical tools for practitioners reading the main text.
- **Visual LLM Structural Projection:** The discussion in Section 5 about how early/middle/late layer specialization and Attention vs. MLP blocks in LLMs might change the "layer-specificity illusion" is intellectually excellent. However, a conceptual table or a structured block diagram illustrating where layer-specific conflicts are most likely to occur in autoregressive decoder architectures would make this discussion significantly more impactful.
- **Discussion of Concurrent Mitigation Frameworks:** The paper would benefit from briefly mentioning and discussing concurrent ideas in the literature that address activation distribution shifts during layer-wise merging, such as *Chain of Merges (CoM)*, which sequentially aligns activation statistics as data flows through the merged network.

---

## 3. Detailed Evaluations

### Soundness: Excellent
The submission is technically flawless, exceptionally rigorous, and methodologically appropriate. The dual-optimizer design, the use of shuffling, averaging, and noise controls, and the evaluation over 3 independent seeds with disjoint splits are of the highest standard. The authors' claims are fully supported by empirical and theoretical evidence, and the logged results are consistent.

### Presentation: Excellent
The paper is exceptionally well-written, logically structured, and polished. The mathematical formulations are clear and complete. The figures (Figures 1, 2, 3) are highly informative, featuring appropriate standard error bars and noise sweeps. The tables (Table 1, Table 2) are clean and complete. The overall narrative is engaging and easy to follow.

### Significance: Excellent
The paper addresses an important and timely problem in model merging and Test-Time Adaptation. It provides a vital, much-needed course-correction for the community, warning against the deployment of high-parameter test-time adaptive schedules without proper scientific controls (shuffling/averaging) and properly calibrated baselines. It will likely shape future model merging research and influence researchers to adopt explicit proximity regularization.

### Originality: Excellent
The paper is highly original. While most papers propose increasingly complex, unvalidated merging schemes, this paper takes a highly novel critical/methodological approach. Deconstructing the layer-specificity assumption, discovering the Overfitting-Optimizer Paradox, identifying the CKA vs. Accuracy decoupling, and formulating the joint entropy task-bias represent a highly original combination of insights that significantly advance the field.

---

## 4. Constructive Suggestions for Improvement
1. **Integrate Appendix Results into Main Text:** Please consider moving a concise table of the pilot results for Proximity Regularization (Appendix B) and Scale-Normalized Joint Entropy (Appendix E) into the main text (e.g., as a small Subsection in Section 4). This would balance the critique with immediate, actionable remedies.
2. **Add a Conceptual Table or Diagram for LLM Layer-Specificity:** In Section 5, when discussing how layer-specificity may become real in larger autoregressive LLMs, it would be highly valuable to include a conceptual table detailing layer/block-type (e.g., Early Syntax, Middle Facts, Late Generation, Attention Query/Key/Value Projection vs. MLP Gating) and their expected sensitivity to model merging conflicts.
3. **Reference Concurrent Layer-by-Layer Merging Controls:** Please add a brief citation and discussion of *Chain of Merges (CoM)* in the Related Work or Discussion sections. CoM is highly relevant as it sequentially addresses activation-level shifts during layer-by-layer merging, providing a useful contrast to static layer-wise coefficient optimization.

---

## 5. Overall Recommendation
**Rating: 6 (Strong Accept)**

**Justification:** 
This is an outstanding, technically flawless paper that provides an exceptional and vital scientific course-correction for the model merging and Test-Time Adaptation communities. Through a masterfully designed set of control treatments (layer shuffling, spatial averaging, relative noise sweeps) evaluated over 3 independent random seeds, the authors expose a profound Overfitting-Optimizer Paradox and representational CKA-accuracy decoupling. Their literature integration, proper attribution, and deep contextualization are exemplary. With stellar reproducibility, rigorous statistics, and highly effective proposed solutions (Proximity Regularization and Scale-Normalized Joint Entropy), this paper meets the absolute highest bar of scientific quality and will have an exceptional impact on the field of machine learning.
