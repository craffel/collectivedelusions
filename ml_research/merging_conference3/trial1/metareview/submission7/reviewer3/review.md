# Peer Review

## Summary of the Paper
This paper presents a methodology-focused sanity check and representational analysis of layer-wise model merging frameworks (such as AdaMerging and SyMerge). These SOTA methods optimize individual merging coefficients for each transformer layer and task using test-time adaptation (typically prediction entropy minimization on unlabeled calibration data). 

Using a pre-trained **CLIP ViT-B/32** backbone across four classification tasks (**MNIST, FashionMNIST, CIFAR-10, SVHN**), evaluated over **3 independent random seeds**, the author stress-tests the learned merging coefficients using three control diagnostic treatments:
1. **Intra-Task Layer Shuffling**: Shuffling learned coefficients across layers for each task.
2. **Task-Wise Spatial Averaging (Spatial Mean)**: Collapsing the layer-wise coefficients into a single uniform task-wise scalar (reducing parameters from 52 to 4).
3. **Norm-Bounded Perturbation**: Injecting varying levels of relative Gaussian noise into the coefficients.

These treatments are evaluated across two distinct optimization regimes: zero-order **Adaptive 1+1 Evolution Strategy (1+1 ES)** and first-order **Adam Gradient Descent (Adam GD)**. Additionally, **linear Centered Kernel Alignment (CKA)** at Layer 6 on CIFAR-10 is used to analyze activation-space representational similarity.

---

## Key Findings & Claims
1. **The Overfitting-Optimizer Paradox:** 
   * Under zero-order 1+1 ES, layer-specificity behaves like an illusion. Replacing complex layer-wise parameters with their Spatial Mean actually improves average test performance from $85.07 \pm 0.47\%$ to $85.21 \pm 0.11\%$, acting as a powerful spatial regularizer that smooths out optimization noise.
   * Under first-order Adam GD, unconstrained optimization finds a highly precise, delicate configuration of layer coefficients that is extremely sensitive to shuffling or spatial averaging, creating an illusion of functional layer-specificity. However, this delicate structure fails to outperform the unoptimized Task Arithmetic baseline ($84.44 \pm 0.37\%$) on the unseen test set while introducing 4x greater seed variance, revealing it as a delicate transductive overfitting artifact.
2. **Extreme Landscape Flatness:** Both optimizers tolerate up to **50% relative Gaussian noise** on coefficients with negligible average test decay, indicating that the optimized parameters operate in an exceptionally flat loss basin.
3. **CKA vs. Downstream Accuracy Discrepancy:** While spatial averaging marginally improves average activation similarity to original experts, high-level linear kernel alignment decouples from downstream classification performance. On CIFAR-10 under Adam GD, the spatially averaged model has a higher CKA than the optimized model but suffers a catastrophic **10.35% collapse** in test accuracy.
4. **Proposed Solution (Proximity-Based Regularization):** The author proposes adding an $L_2$ proximity penalty to pull coefficients toward the uniform baseline ($\lambda=0.3$), which stabilizes the optimization, reduces seed-to-seed variance, and outperforms standard optimizer weight decay (which catastrophically collapses complex task experts like SVHN).

---

## Strengths
1. **Rigorous and Destructive Diagnostic Controls:** The introduction of Shuffling, Spatial Mean, and Noise Perturbations is highly elegant and effective. It successfully stress-tests the core physical assumptions of layer-wise model merging.
2. **Exemplary Conceptual Framing:** The "Overfitting-Optimizer Paradox" is a powerful, clean conceptual framework that successfully connects optimizer choice, parameter density, and transductive overfitting.
3. **Exceptional Appendix Rigor:** The appendices are outstanding. The sweeps over learning rate (Appendix E), calibration size (Appendix D), weighted joint entropy (Appendix C), and standard weight decay vs. proximity regularization (Appendix F) provide complete, multi-dimensional support for all claims and solutions.
4. **Practical and Well-Justified Solution:** The proposed Proximity-Based Regularization is simple, computationally cheap, physically grounded, and empirically verified to prevent overfitting without collapsing task-expert capabilities.
5. **Exposing Interpretability Limits:** The empirical proof that activation-level alignment (CKA) can completely decouple from physical downstream classification accuracy under coordinate shifts is a highly significant and valuable warning for the broader interpretability community.

---

## Weaknesses

### 1. Saturated and Toy-Like Vision Benchmarks
The primary limitation of this paper is its empirical scope. It is restricted entirely to four standard, low-resolution, and largely saturated image classification datasets (MNIST, FashionMNIST, CIFAR-10, SVHN).
* **Trivial Task Vectors:** Because these tasks are simple, their task vectors lie extremely close to the pre-trained initialization. This creates an artificially flat loss landscape where almost any linear combination of weights performs reasonably well, potentially inflating the "landscape flatness" and "layer-specificity illusion" claims.
* **Lack of Scale:** Modern model merging is primarily utilized to combine large autoregressive language models (LLMs like LLaMA, Mistral) or highly complex diffusion models. A study restricted entirely to a 13-layer ViT-B/32 lacks the necessary complexity to prove that these findings generalize to high-dimensional parameter manifolds with deeper, more complex layer-specific specializations.

### 2. Hypocritical Baseline Selection
The author heavily criticizes prior model merging publications for comparing their complex methods against a "weak, manually selected uniform Task Arithmetic baseline" without proper tuning.
However, in Table 1, the author compares their optimized methods against a fixed **"Task Arithmetic (Baseline, $\lambda=0.3$)"**. The author never performs a search or sweep on the uniform baseline coefficient ($\lambda \in [0.1, 1.0]$) to find the optimal uniform scalar for this specific multi-task set. It is highly likely that a properly swept uniform baseline would outperform the fixed $\lambda=0.3$ baseline. By failing to tune their own uniform baseline, the author commits the exact methodological error they accuse the rest of the community of committing.

### 3. Highly Selective CKA Reporting (Cherry-Picking Risk)
In Section 4.4 and Table 2, the author reports CKA values **exclusively at Layer 6** on CIFAR-10 inputs. 
A CLIP ViT-B/32 backbone has 13 parameter groups (12 transformer layers plus 1 projection layer). The author provides no justification for why Layer 6 was selected, nor do they report the CKA profile across all 13 layers, or even the mean CKA across the model. Reporting CKA for a single, arbitrary intermediate layer represents a selective reporting risk, which weakens the generalizability of the CKA-accuracy decoupling argument.

### 4. Overfitting as a Choice of Hyperparameters rather than an Inherent Flaw
The author's primary thesis is that layer-wise model merging is prone to severe transductive overfitting on the calibration set. However, the author uses an extremely small calibration split of **256 images (64 images per task)** to tune 52 parameters.
In Appendix D (Figure 5), the author sweeps the calibration sample size per task. When the calibration size is scaled to 128 images per task (512 total), the unconstrained Adam GD model's test performance stabilizes and reaches **87.06% average accuracy** (and 87.70% on CIFAR-10), outperforming both the uniform baseline and the spatially averaged models. This empirical finding suggests that the "Overfitting-Optimizer Paradox" is not an inherent, unavoidable flaw of layer-specific model merging, but rather a direct consequence of the author's choice of an unnecessarily small calibration sample size.

### 5. Simplistic Zero-Order Optimization Control
The author implements a very basic **Adaptive 1+1 Evolution Strategy (1+1 ES)** as their zero-order optimizer. SOTA black-box optimization in machine learning employs much more sophisticated algorithms (e.g., CMA-ES, Bayesian Optimization, or random search with restarts/momentum). Using a simplistic 1+1 ES random-walk optimizer artificially inflates the "high-frequency optimization noise" in the learned coefficients, which naturally makes task-wise Spatial Averaging (which smooths out this noise) look highly effective. It is unclear if these findings would hold if a more sophisticated derivative-free optimizer had been deployed.

---

## Detailed Evaluation Ratings

### Soundness: Good
The paper is technically solid and exceptionally thorough. The authors run all experiments across 3 independent seeds and provide complete statistical reporting (means and standard deviations). However, the soundness is rated as "Good" rather than "Excellent" due to:
* The hypocritical baseline selection (using a fixed, unoptimized $\lambda=0.3$ baseline).
* The highly selective, single-layer reporting of CKA similarity.
* The reliance on toy-like, saturated vision datasets where task-vectors are trivial.
* The fact that "overfitting" is shown to be a direct consequence of an unnecessarily small calibration sample size (Appendix D) rather than a fundamental limitation of layer-wise merging itself.

### Presentation: Excellent
The writing is exceptionally clear, elegant, and highly structured. The equations are mathematically precise, the tables are clean, and the visual graphics in both the main text and appendix are outstanding. The author's discussion of the "SVHN Rescue vs. CIFAR-10 Collapse" trade-off demonstrates an admirable level of intellectual nuance and self-critical awareness.

### Significance: Good
The paper serves as a highly necessary and timely **course-correction** for the model merging community. It will likely force future publications to include rigorous shuffling/averaging controls, tuned uniform baselines, and explicit parameter regularization. It also provides a valuable caution for the interpretability community regarding CKA decoupling. However, its overall significance is bounded by its toy-like empirical scale (ViT-B/32 on MNIST/CIFAR/SVHN). Without scaling validation to LLMs or complex multi-task NLP/vision settings, the generalizability of these findings remains restricted.

### Originality: Good
The introduction of simple, elegant diagnostic controls (Shuffling and Spatial Averaging) to stress-test learned model-merging coefficients is highly creative. While the individual tools (1+1 ES, Adam, CKA, $L_2$ regularization) are standard, their combination and application to expose flaws in SOTA model-merging assumptions represents a valuable and original contribution.

---

## Overall Recommendation: 4 (Weak Accept)
This is a technically solid, exceptionally well-written, and thorough paper that provides a highly valuable course-correction for the model-merging community. The introduction of diagnostic controls (shuffling and averaging), the exposition of the CKA-accuracy decoupling, and the proposed proximity-based regularization are all strong, high-signal contributions. 

However, we recommend a **Weak Accept** (rather than a full Accept) because the empirical evaluation is restricted entirely to toy-like, saturated vision classification benchmarks on a shallow CLIP ViT-B/32 backbone. Since model merging is primarily utilized for large-scale autoregressive language models (LLMs) and complex multi-task generation, the generalizability of these findings remains unproven in larger, deeper parameter manifolds with more pronounced layer-specific specializations. Furthermore, the selective CKA reporting (single layer) and the fact that "overfitting" is largely a hyperparameter symptom of an unnecessarily small calibration split represent notable methodological gaps. 

If the authors can validate these findings on a larger scale (e.g., a 7B+ parameter LLM on instruction-tuning/coding/factual experts) or address the baseline and CKA reporting gaps, this paper could easily be upgraded to a strong accept.
