# Peer Review

## Summary of the Paper
This paper addresses the challenge of low-data calibration overfitting in dynamic weight-space model merging. When input-dependent routing coefficients are predicted sample-by-sample to dynamically merge task-specific experts on extremely small calibration datasets ($B_{\text{cal}} \le 64$), parametric routers overfit rapidly, leading to generalization collapse. The authors propose **Spectral and Rademacher-guided Routing Regularization (SR3)**, a first-principles approach derived from statistical learning theory. 

By bounding the Rademacher complexity of a dynamically merged coupled Softmax model class, they prove that the generalization gap scales with the product of the routing parameter norm ($\|W_k\|_2$) and the corresponding task-vector norm ($\|V_k\|_F$ or $\|V_k\|_{op}$). They propose scaling the router weight decay proportionally to the pre-computed task-vector norm (using Frobenius, SR3-F, or Spectral operator, SR3-S, variants). They also introduce several advanced layers: a smoothed $L_1$ Group-Lasso variant (SR3-L1), a warm-up/scheduling scheme to bypass early-stage optimization hurdles, and a hybrid adaptive capacity controller to resolve specialization-generalization trade-offs. The framework is validated on a synthetic multi-layer weight-space simulator and a physical PyTorch TinyMLP digit classification experiment.

---

## Strengths and Weaknesses

### Strengths
1. **Conceptual Elegance of the Core Idea:** The fundamental idea of scaling standard weight decay proportionally to the precomputed parameter-space distances of task-specific experts is incredibly elegant. It is simple, direct, and computationally clean—requiring zero custom routing architectures, extra training branches, or online overhead.
2. **Mathematical Rigor:** The paper provides a highly detailed learning-theoretic derivation (Theorem 3.1). Unlike prior ensembling literature that relies on simplified independent sigmoid gating approximations, this work directly tackles the coupled Softmax layer and integrates Maurer’s vector-valued contraction theorem, resolving a significant theoretical gap.
3. **Outstanding Scientific Transparency:** The authors demonstrate exemplary honesty in Section 4.4. They openly discuss the circularity of their synthetic simulator's evaluation, the non-smooth optimization barriers of direct $L_1$ minimization, and the "Double-Edged Sword" of asymmetric regularizers (how suppressing high-norm experts degrades their task-specific performance). This level of candor is refreshing and highly commendable.
4. **Comprehensive Ablations:** The ablation studies regarding feature subspace projection dimension, power iteration scalability, and alternative scheduling functions (linear, cosine, exponential) provide useful engineering insights.

### Weaknesses
1. **Unjustified Complexity Bloat (Over-Engineering):** While the core idea of scaling weight decay is beautifully simple, the subsequent additions represent a significant regression in simplicity:
   * The smoothed $L_1$ Group-Lasso variant introduces a smoothing parameter $\epsilon_{\text{smooth}}$.
   * Regularization scheduling transitions from quadratic to $L_1$ using linear/cosine/exponential warm-ups, introducing an epoch-duration parameter $T$.
   * Hybrid adaptive capacity controllers track exponential moving averages of gradient norms, introducing hyperparameters $\beta$ and $\gamma$ alongside exponential decay scaling.
   
   These intricate layers add substantial conceptual and hyperparameter-tuning overhead. Crucially, they deliver virtually zero empirical performance gains (e.g., Joint Mean on the simulator improves by less than $0.15\%$). This over-engineering severely detracts from the elegance of the primary contribution.
2. **Empirical Failure on Physical Networks:** In the physical PyTorch TinyMLP handwritten digit experiment—which represents the only unbiased evaluation breaking the synthetic simulator's circularity—**standard, uniform $L_2$ weight decay (at $92.13\% \pm 2.47\%$) and the simple TSAR heuristic (at $92.13\% \pm 2.92\%$) outperform all proposed SR3 variants on average over 10 seeds** (SR3-F: $90.50\% \pm 1.36\%$; SR3-S: $90.93\% \pm 1.94\%$; SR3-H: $91.20\% \pm 1.81\%$). When evaluated on real-world data, the complexity-blind, uniform $L_2$ regularizer is superior. This indicates that the highly engineered, asymmetric complexity bounds of SR3 do not translate to genuine gains in physical settings.
3. **Scale of Real-World Evaluation:** Validating a new regularization method on a shallow 2-layer MLP using the toy scikit-learn `load_digits` dataset is insufficient. Larger-scale physical experiments (such as merging fine-tuned low-rank LoRA adapters on Vision Transformers or small LLMs) are required to verify if these learning-theoretic scaling principles hold in modern deep networks.

---

## Detailed Evaluation Ratings

### Soundness: Good
The mathematical proofs are correct, airtight, and highly rigorous. However, the empirical soundness is limited. The synthetic simulator’s test-time generalization gap is defined using the exact Rademacher complexity formula that SR3 is designed to minimize, creating a circular evaluation setup. On the real-world dataset (TinyMLP), the proposed methods fail to outperform a standard, uniform $L_2$ regularizer.

### Presentation: Excellent
The paper is exceptionally well-written, clearly structured, and easy to follow. The mathematical notation is clean and rigorous, and the critical discussion section is outstanding.

### Significance: Fair
While connecting learning theory to weight-space ensembling is theoretically interesting, the practical utility of SR3 is currently low. Because standard uniform $L_2$ weight decay is both simpler to implement/tune and empirically superior in physical experiments, practitioners have little reason to adopt the much more complex SR3 framework.

### Originality: Good
The derivation of geometry-aware asymmetric weight decay is highly original and represents a creative combination of statistical learning theory and model merging. However, the advanced variants (L1, schedules, hybrid controllers) are incremental and heuristic.

---

## Overall Recommendation

**Rating: 3 (Weak Reject)**

**Justification:**
This submission has clear merits, particularly in its rigorous theoretical foundations and its commendable scientific transparency. However, from a perspective that highly values simple, direct, and effective methods, the paper's weaknesses outweigh its merits:
1. **The Core Premise vs. Uniform Baseline:** A simple, uniform $L_2$ regularizer is conceptually simpler, has only a single hyperparameter, requires no pre-computations, and outperforms the proposed asymmetric regularizer on the real-world dataset on average.
2. **Over-Engineering:** The paper attempts to patch optimization and task-suppression issues by piling on increasingly complex heuristics (smoothed L1, warm-ups, exponential gradient-norm decay controllers). This complexity bloat delivers negligible empirical benefit while introducing numerous arbitrary hyperparameters, running counter to the elegance of the core idea.
3. **Small-Scale Physical Validation:** The real-world verification is restricted to a toy MLP on handwritten digits. 

To improve the paper, the authors should:
* Strip away the over-engineered advanced variants (smoothed L1, schedules, and hybrid controllers) and focus exclusively on making the core asymmetric regularizer robust and simple.
* Investigate why the theoretical asymmetric bound degrades performance in physical settings compared to uniform $L_2$ regularization.
* Execute larger-scale physical validations (e.g., merging PEFT/LoRA adapters on ViTs or small LLMs) to demonstrate that the proposed method can deliver genuine, practical utility in modern architectures.
