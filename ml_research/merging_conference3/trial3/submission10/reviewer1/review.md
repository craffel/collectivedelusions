# Peer Review of "ChebyMerge: Stable and Optimal Continuous Subspace Model Merging"

## Summary of the Paper
The paper introduces **ChebyMerge** (Stable and Optimal Continuous Subspace Model Merging), a mathematically rigorous framework designed for multi-task model merging under unsupervised test-time adaptation (TTA). Dynamic model-merging methods (such as AdaMerging) optimize separate layer-wise merging coefficients on-the-fly using unlabeled data streams. The authors expose a critical vulnerability in these unconstrained optimizers: the **Overfitting-Optimizer Paradox**, where high degrees of freedom allow the model to overfit to transductive, local sampling noise, causing representation collapse.

To resolve this, the authors propose projecting the high-dimensional coefficient space onto a low-dimensional continuous subspace spanned by orthogonal **Chebyshev polynomials of the first kind**. They prove that while previous polynomial approaches (e.g., PolyMerge) successfully filter out transductive noise, they suffer from extreme exponential ill-conditioning because their standard monomial basis ($1, \bar{l}, \bar{l}^2, \dots$) yields a Vandermonde-type Gram matrix whose condition number grows as $\mathcal{O}(4^d)$ (exceeding 10,400 for a cubic degree). ChebyMerge resolves this by achieving near-optimal uniform approximation with a condition number bounded by a tiny constant ($\approx 2.95$ for cubic degree), representing a **3,527x numerical improvement**. 

Furthermore, the authors identify the **Conditioning-Generalization Paradox**, explaining how PolyMerge's extreme ill-conditioning acts as an accidental, implicit spectral damping filter. They introduce **Controllable Spectral Decay (CSD)** to decouple conditioning from regularization, allowing explicit and controllable low-pass filtering. The paper evaluates these methods across 30 seeds in highly non-convex, coupled simulated stress-tests (Rastrigin-type loss surfaces) and via actual physical validation on pre-trained CLIP ViT-B/32 models.

---

## Strengths and Weaknesses

### Originality
* **Strength (Highly Original Conceptual Insights):** The true conceptual leaps in the paper lie in the formulation of the **Conditioning-Generalization Paradox** and the subsequent development of **Controllable Spectral Decay (CSD)**. The insight that monomial ill-conditioning accidentally acts as an implicit, noise-driven spectral damping filter (early stopping) is a highly profound and original contribution. It deepens our theoretical understanding of optimization dynamics in deep neural networks and moves beyond standard empirical reports.
* **Strength:** Exposing the **Overfitting-Optimizer Paradox** under unsupervised TTA provides an elegant, physically-grounded explanation of why unconstrained optimization (e.g., AdaMerging) leads to representation collapse, framing the issue as transductive noise memorization.
* **Weakness (Incremental Algorithmic Novelty):** From a purely methodological and paradigm perspective, the core algorithmic framework of constraining layer-wise merging coefficients to a low-dimensional continuous polynomial subspace is directly inherited from PolyMerge (2024). Replacing the standard monomial basis with the orthogonal Chebyshev basis is a standard, classical numerical analysis technique (a change of basis) rather than a fundamentally new machine learning paradigm. The algorithmic novelty itself is therefore somewhat incremental, even though the mathematical analysis of this change is beautifully executed.

### Soundness
* **Strength (Mathematical Rigor):** The theoretical proofs are mathematically sound and elegant. Theorem 1 rigorously connects the monomial Gram matrix to the Hilbert matrix in the continuous limit, explaining the exponential ill-conditioning. Theorem 2 correctly establishes why the Chebyshev Gram matrix maintains tightly clustered eigenvalues near 1 even when evaluated on uniform grids.
* **Strength (Excellent Experimental Design):** Designing a coupled, non-convex Rastrigin stress-test environment (Model II) with layer sensitivity scaling, non-diagonal covariance (functional couplings), and multi-scale transductive noise is a highly appropriate methodology. It allows for perfect ground-truth control to isolate optimization and numerical conditioning dynamics. Reporting averages across 30 independent seeds provides strong statistical significance.
* **Weakness (Unsupervised TTA Regressions on Physical Models):** A critical inspection of the physical experiments on pre-trained CLIP ViT-B/32 (Table 4) reveals that every single adaptive merging method—including ChebyMerge and ChebyMerge-CSD—performs **worse** than the non-adaptive, static Task Arithmetic baseline (81.50% accuracy). While ChebyMerge-CSD ($d=2$) mitigates PolyMerge's severe collapse (achieving 75.50% vs 70.50%), it still suffers from a **-6.00% absolute accuracy regression** compared to the static baseline. This raises a fundamental concern regarding the practical utility of unsupervised test-time adaptation for model merging under realistic constraints, suggesting that on-the-fly entropy minimization on small, unlabeled streams may be inherently prone to degradation.

### Significance
* **Strength (Theoretical Significance):** The paper's theoretical insights into the Conditioning-Generalization Paradox and the decoupling of conditioning from regularization could have a broad impact, potentially inspiring researchers working on continuous parameterizations in other deep learning subfields (such as neural fields, coordinate-based networks, and hypernetworks).
* **Weakness (Marginal Generalization Gains):** The practical significance of the method is currently limited to the niche of test-time model merging. Furthermore, the performance improvements of ChebyMerge over PolyMerge in stable/low learning rate regimes are extremely marginal (e.g., less than 0.1% average accuracy difference in simulation models, and identical accuracies under low learning rates in Table 5). The primary empirical advantage of ChebyMerge is **optimization robustness and safety** (preventing catastrophic divergence under high learning rates) rather than direct performance gains.

### Presentation
* **Strength (Exemplary Writing and Clarity):** The presentation is outstanding. The writing is clear, precise, and grammatically flawless. The terminology is sophisticated yet accessible, and the mathematical formulas are beautifully formatted and integrated into the text.
* **Strength (Scientific Honesty in Limitations):** The discussion in Section 4.5 regarding topological limitations (sequential depth assumption), asymmetric sensitivity profiles, and CSD decay factor robustness is exceptionally thorough and demonstrates high scientific integrity.

---

## Ratings

* **Soundness:** Excellent
* **Presentation:** Excellent
* **Significance:** Good
* **Originality:** Good (Algorithmically incremental, but Conceptually Excellent)

---

## Overall Recommendation

* **Rating:** 4: Weak Accept
* **Justification:**
  The paper is a technically flawless and mathematically beautiful piece of work. The core conceptual contribution—namely the identification and analysis of the **Conditioning-Generalization Paradox** and the demonstration of how monomial ill-conditioning behaves as an accidental, implicit regularizer—is highly original, thought-provoking, and scientifically rigorous. The proposal of **Controllable Spectral Decay (CSD)** to decouple numerical stability from parameter regularization is mathematically elegant and represents a principled advancement.

  However, from a novelty and paradigm perspective, the paper's core algorithmic change (monomial to Chebyshev basis) is a standard basis change from classical numerical analysis building directly on PolyMerge (2024), making the algorithmic "delta" somewhat incremental. Furthermore, the physical validation reveals a significant practical limitation: all adaptive merging methods (including ChebyMerge-CSD) underperform the static, non-adaptive Task Arithmetic baseline, which suggests that unsupervised test-time adaptation for model merging in realistic settings is still an open challenge with questionable practical utility.

  Therefore, while the theoretical insights and mathematical execution are outstanding and highly deserving of publication, the incremental nature of the core algorithmic change and the practical regression in physical settings lead to a recommendation of a Weak Accept. It is a highly solid and rigorous paper that the continuous-parameterization community will find valuable.

---

## Questions and Constructive Comments for the Authors

1. **Practical Utility of TTA:** In Table 4, why does unsupervised TTA under entropy minimization consistently degrade classification accuracy compared to static Task Arithmetic? Under what conditions (e.g., stream size, alternative objective functions, or downstream task combinations) do you expect ChebyMerge to actually outperform static uniform Task Arithmetic on physical architectures?
2. **Emphasizing CSD:** Since standard ChebyMerge exhibits a small performance gap compared to PolyMerge because it lacks PolyMerge's implicit spectral damping, CSD is the critical component that allows ChebyMerge to achieve state-of-the-art results. Have you considered making CSD a more central contribution in your abstract and introduction, rather than presenting it as an auxiliary feature, given that it directly resolves the Conditioning-Generalization Paradox?
3. **Hyperparameter Selection for CSD:** How sensitive is the performance to the choice of the CSD decay factor ($\gamma_{\text{CSD}}$) under different noise distributions? In practice, how should practitioners set this parameter when labels are entirely absent?
4. **Branched and Non-Sequential Architectures:** You mention graph-spectral projections as a future direction for non-sequential topologies. Could you provide a brief sketch of how a graph-spectral Chebyshev projection would be formulated for a simple parallel-branch layer?
