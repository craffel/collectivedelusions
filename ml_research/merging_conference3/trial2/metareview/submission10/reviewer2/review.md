# Peer Review

## Summary of the Paper
This paper presents a thorough, deconstructive study of unsupervised test-time model merging, focusing on the state-of-the-art **AdaMerging** framework (Yang et al., ICLR 2024). Model merging is an attractive, computationally efficient alternative to traditional multi-task learning, allowing multiple task-specific models (fine-tuned from a shared pre-trained base model) to be integrated in weight space without retraining or raw training data access. To avoid manual hyperparameter tuning of merging scales, AdaMerging dynamically optimizes scaling coefficients using a small calibration batch at test time by minimizing joint prediction entropy.

Guided by Occam's razor, the authors investigate whether the high-dimensional optimization of layer-wise scaling coefficients captures genuine multi-layer coordination of task representations or simply overfits to the small, unlabeled test-time adaptation split. To do so, they design and evaluate two structural diagnostic controls:
1. **Intra-Task Layer Shuffling:** Permuting learned layer-wise coefficients of each task across the different layers of the neural network to break architectural alignment.
2. **Spatial Averaging (Spatial Mean):** Replacing optimized layer-wise coefficients with their flat average over all layers, reducing the parameter degrees of freedom from hundreds (layer-wise) back to a single scalar per task.

Through these diagnostics, the authors expose two major optimization anomalies:
* **The Overfitting-Optimizer Paradox:** While high-dimensional layer-wise AdaMerging achieves a high average accuracy of $88.05\%$, shuffling these coefficients across layers collapses performance to $78.61\%$ (Adam GD), demonstrating that they are structurally specialized to the network's architectural hierarchy. However, taking their spatial mean post-hoc (Spatial Averaging) acts as a powerful low-pass filter, smoothing away individual layer-wise transductive overfitting from the small test-time calibration batch while successfully preserving the robust global task-level scaling signals to achieve $84.96\%$ (outperforming static Task Arithmetic).
* **The Spatial Averaging Paradox:** Intriguingly, *direct* optimization of flat task-wise scales (Task-wise AdaMerging) fails spectacularly, collapsing performance to $81.19\%$ (3.45% below its unoptimized uniform initialization). The authors mathematically and empirically explain this paradox through **multi-task gradient imbalance** on uncalibrated joint prediction entropy. Easy tasks (like MNIST/FashionMNIST) have sharp logit distributions and highly responsive prediction entropies, causing their gradients to dominate joint optimization and scale up their task vectors. Under a low-dimensional bottleneck (1 scalar per task), this joint scaling creates severe, destructive parameter interference in shared early projection layers, collapsing representation structures and performance on harder, heterogeneous tasks (CIFAR-10 and SVHN). Under high-dimensional layer-wise optimization, the optimizer has enough local layer degrees of freedom to minimize entropy locally (e.g., in late task-specific layers) without resorting to destructive global scaling trade-offs.

The authors also evaluate a **Calibrated Prediction Entropy** remedy, which normalizes each task's loss contribution at initialization. Although calibration balances gradients initially, direct task-wise optimization still fails (achieving $80.59\%$), proving that the pathology is structural: a global, flat task scale is fundamentally incompatible with joint prediction entropy minimization due to weight-space interference in shared layers. 

---

## Strengths

1. **Rigorous Literature Positioning and Contextualization:** 
   The paper is exceptionally well-situated within the historical and current landscape of weight-space model merging. It correctly references foundational works on static combinations (Model Soups, Task Arithmetic), static pruning/sign-consensus heuristics (TIES-Merging, DARE-Merging), and adaptive, test-time optimization schemes (AdaMerging, Representation Surgery). Furthermore, it draws valuable parallels to other deconstructive studies in the machine learning literature (e.g., critiques of Sharpness-Aware Isotropic Merging and FoldMerge), which helps frame this contribution within a broader, healthy trend toward critical, minimalist analysis of over-engineered ML pipelines.
2. **Proper and Honest Academic Attribution:**
   Unlike many submissions that inflate their own novelty by ignoring closely related workshop papers, the authors explicitly cite and acknowledge prior workshop observations (Fictional, Author D., NeurIPS PEL Workshop 2025) which first indicated that layer-wise coefficients overfit to the test sample. The authors then clearly and honestly articulate their "delta": they build upon and formalize this initial observation by designing rigorous structural diagnostics, formalizing the Spatial Averaging Paradox, and proving the structural bottleneck hypothesis through Calibrated Prediction Entropy. This represents a very high standard of academic integrity and rigorous research development.
3. **Elegance and Signal-to-Noise of Diagnostics:**
   The introduction of Intra-Task Layer Shuffling and Spatial Averaging is highly creative. These diagnostic treatments require absolutely zero training overhead, yet they provide extremely high-signal evidence. Shuffling isolates structural specialization (showing learned scales are deeply tied to the network hierarchy), while post-hoc averaging isolates transductive noise (showing spatial smoothing acts as an elegant low-pass filter).
4. **Optimization-Independent Verification:**
   By evaluating both a first-order gradient-based optimizer (**Adam GD**) and a zero-order derivative-free optimizer (**1+1 ES**), the authors confirm that the discovered paradoxes are fundamental properties of the loss landscape and parameter bottlenecks rather than optimization-specific artifacts, learning rate bugs, or local minima traps.
5. **High Empirical Standards and Statistical Soundness:**
   Evaluating on the full standard test splits of all four datasets (a massive total of **56,032 images**) completely eliminates data selection and evaluation split bias. Reporting seed-controlled means and standard deviations across three independent splits is a crucial best-practice, especially since challenging, heterogeneous datasets like SVHN exhibit significant standard deviations (up to $\pm 7.3\%$) under data-scarce test-time splits.
6. **Bridging to Classical Representation Theory:**
   The layer-by-layer Linear CKA representational similarity mapping across all 12 Transformer blocks beautifully bridges weight-space optimization with classical representation learning principles. It empirically validates the hierarchical routing hypothesis, proving that early layers maintain near-perfect representational alignment ($CKA > 0.995$) across all merging schemes because they learn domain-agnostic visual features, whereas high-dimensional adaptive optimizers distribute task adaptation to local, task-specific late layers.

---

## Weaknesses and Areas for Improvement

While the paper is exceptionally strong, there are a few areas that could be improved or expanded:

1. **Refining the proposed Algorithmic Remedy:**
   * The proposed *Calibrated Prediction Entropy* remedy normalizes each task's loss contribution using its static entropy value at initialization. However, as optimization proceeds, easy tasks can still scale up their coefficients to drive joint entropy to zero via the logit-inflation shortcut.
   * *Recommendation:* The paper would be strengthened by exploring a dynamic calibration scheme (re-evaluating normalization factors online or using running averages) or introducing a direct regularization penalty (such as an L2 scale penalty or logit clipping) to penalize weight-scaling shortcuts.
2. **Empirical CKA Baseline Comparisons:**
   * The authors assert that the near-perfect Linear CKA similarity ($CKA > 0.995$) observed in early-to-mid layers is a baseline property of task vector scaling ($\lambda \approx 0.3$) rather than a unique property of adaptive entropy minimization. 
   * *Recommendation:* To make this claim empirically ironclad, the authors should include a comparison against a *randomly scaled* Task Arithmetic baseline. If random scales also achieve $>0.995$ CKA, it would decisively prove that early-layer representational similarity is insensitive to scale perturbations.
3. **Generative LLM Contextualization:**
   * In Section 5.1 (Limitations and Future Directions), the authors discuss how these gradient imbalances might translate to token-level perplexity in Large Language Models (LLMs). This is an exciting discussion, but it would benefit from citing existing literature on LLM test-time adaptation or multi-task generative merging (e.g., how perplexity is utilized for merging instruction-tuned models) to make the connection more academically grounded.

---

## Evaluation of Individual Dimensions

### Soundness: Excellent
The experimental design is exceptionally thorough, seed-controlled across 3 seeds, and evaluated on a massive scale (56,032 images). The findings are validated across two independent optimization paradigms, and the structural diagnostics (shuffling and averaging) are mathematically sound. The conclusions are tightly supported by the presented empirical data.

### Presentation: Excellent
The paper is beautiful to read, with a clear narrative arc, high-quality figures, and extremely detailed and transparent tables. Equations are precise, and all key terms are well-defined. Footnotes and limitations are handled with a high degree of professional integrity.

### Significance: Excellent
By exposing the Spatial Averaging Paradox, this paper acts as a crucial, timely warning to the model merging community against blindly building increasingly complex, parameter-rich test-time adaptive pipelines. Furthermore, demonstrating that post-hoc Spatial Averaging acts as a self-regularizing, label-free scaling estimator that outperforms Task Arithmetic provides immense practical utility for zero-overhead multi-task combinations.

### Originality: Excellent
Exposing and mathematically characterizing the multi-task gradient imbalance in weight-space bottlenecks under uncalibrated prediction entropy is a highly original conceptual contribution. The design of Intra-Task Layer Shuffling and Spatial Averaging as diagnostic controls represents a highly creative and original contribution.

---

## Overall Recommendation

**Rating: 5: Accept** (or **6: Strong Accept**)
*Justification:* This is a outstanding, publication-grade deconstructive study that takes a critical, minimalist lens to state-of-the-art adaptive model merging. The paper is mathematically rigorous, empirically exhaustive, and beautifully written. It properly credits prior workshop literature while establishing a highly significant conceptual "delta" through the discovery and formalization of the Spatial Averaging Paradox. The empirical validation on a massive evaluation scale across multiple seeds and optimization paradigms is exemplary. It provides both vital theoretical warnings on weight-space bottlenecks and immediate practical utility through post-hoc Spatial Averaging. I strongly recommend accepting this paper.
