# Peer Review

**Paper Title:** Curvature-Aware Analytical Model Merging: Direct Closed-Form Multi-Task Parameter Fusion

---

## 1. Summary of the Paper
The paper presents **Curvature-Aware Analytical Model Merging (ACM)**, a training-free framework designed to consolidate multiple independently fine-tuned task-specific neural networks (experts) into a single unified network directly in the parameter space. To overcome representation interference across tasks, the authors formulate layer-wise merging coefficient search as a joint quadratic loss minimization problem based on second-order Taylor expansions around the expert task minima. 

To make this mathematically tractable over modern high-dimensional parameter spaces, ACM projects each layer's weights onto the $K$-dimensional subspace spanned by the task vectors, allowing the computation and inversion of the **full, non-diagonal, cross-parameter projected Hessian** with zero diagonal approximation. Under Ridge (L2) or Lasso (L1) regularization, this yields an exact closed-form analytical solution solved in a single step using a small calibration dataset. The authors also propose a robust finite-difference estimation scheme with explicit gradient subtraction to cancel out residual gradients and prevent numerical instability. Empirical evaluations are conducted on stylized simulated environments (30 seeds) and physically on a Vision Transformer (ViT-Tiny) backbone across four image classification benchmarks (MNIST, FashionMNIST, CIFAR-10, SVHN).

---

## 2. Strengths
* **Outstanding Mathematical Rigor:** The paper is exceptionally well-formulated, establishing a rigorous first-principles quadratic optimization framework over task-specific local Hessian geometries in a low-dimensional subspace.
* **Innovative Gradient Subtraction:** The proposed robust finite-difference scheme with gradient subtraction ($\nabla \mathcal{L}_k(W_k + \epsilon v_j) - \nabla \mathcal{L}_k(W_k)$) is highly clever. It mathematically cancels out unperturbed residual gradients, ensuring complete noise immunity and resolving a standard numerical instability of finite-difference Hessian estimation.
* **Exceptional Intellectual Honesty and Transparency:** The authors are highly commendable for their deep, transparent, and rigorous discussion of their method's limitations. They do not hide negative results or hide underperforming variants; instead, they dedicate significant space to mathematically formalizing the local-global optimization gap, analyzing the collapse of sequential multi-layer coordination, and identifying the ill-conditioning risks of low-parameter layers.
* **Writing Quality & Reproducibility:** The paper is exceptionally well-written, structured, and easy to follow. The inclusion of training hyperparameters, baseline setups, code-level descriptions of autograd graph resolution, and a detailed pseudo-code algorithm makes the work highly reproducible.

---

## 3. Weaknesses

While the paper is theoretically elegant and beautifully written, a rigorous, critical examination reveals several fundamental flaws in its methodology, empirical execution, and practical utility.

### A. Severe Lack of Practical Utility (Outperformed by Curvature-Blind Scaling)
The central premise of the paper is that modeling full, non-diagonal, second-order curvature is essential to find optimal merging coefficients and reduce representation interference. However, the physical validation on Vision Transformers (Table 2) completely undermines this thesis:
* **The baseline to beat is Task Arithmetic (Best Tuned 0.4)**, which is a curvature-blind, globally uniform, zero-calibration baseline that simply scales task vectors by a single uniform factor. It achieves a **60.72%** Joint Average accuracy.
* **Vanilla ACM** ($60.89\%$) barely outperforms Task Arithmetic by a statistically insignificant **$0.17\%$** absolute accuracy, despite requiring calibration datasets, finite-difference perturbations, and matrix inversions.
* **Worse yet, all of the authors' proposed advanced variants perform significantly *worse* than Task Arithmetic:**
  * **ACM-Norm** (layer-wise scale normalization, pitched as essential to prevent sacrificial task bias) achieves only **58.89%** ($-1.83\%$ vs. Task Arithmetic).
  * **ACM-GlobalNorm** (globally scale-normalized ACM, designed to preserve depth-wise sensitivity profiles) achieves only **57.76%** ($-2.96\%$ vs. Task Arithmetic).
  * **Lasso ACM-GlobalNorm** (L1 regularized) achieves only **57.52%** ($-3.20\%$ vs. Task Arithmetic).
  * **Gauss-Seidel Coordinated ACM-GlobalNorm** (multi-layer iterative coordination) collapses to **36.65%** ($-24.07\%$ vs. Task Arithmetic).

This empirical breakdown demonstrates that the massive mathematical machinery introduced in this paper fails to translate into any meaningful physical advantage over a trivial uniform interpolation.

### B. Methodological Mismatch: The Local-Global Optimization Gap
The paper derives its analytical closed-form solution from local second-order Taylor expansions taken around individual expert minima. However, model merging involves shifting parameters far from any individual expert checkpoint (e.g., $W(\Lambda) - W_k \approx -0.7 v_k + \sum_{j \neq k} 0.3 v_j$). On highly non-convex, converged physical neural manifolds, the local quadratic surrogate breaks down over these large steps.

The authors' own theoretical derivation in Appendix B.4 confirms that this local-global approximation error scales **cubically** ($O(V_{\max}^3)$) with the task vector norm. This is a devastating structural limitation: in realistic settings, model merging is applied to fully converged, highly fine-tuned experts (where task vectors are inherently large). Thus, ACM is mathematically guaranteed to exhibit high approximation errors in standard deployment scenarios, explaining why it fails to outperform a simple uniform global regularizer like Task Arithmetic.

### C. Numerical Ill-Conditioning Misinterpreted as Representational Insight
The authors observe that at Layer 13 (the final LayerNorm layer), Scale-Normalized ACM (ACM-Norm) solves the merging coefficients to extremely large values (e.g., MNIST: $91.516$, SVHN: $-95.691$, FashionMNIST: $-40.076$). They dedicate substantial analysis to interpreting this as a "profound physical and mathematical insight" representing an "active cancellation mechanism" that "orthogonalizes the update pathways."

This interpretation is highly suspect and likely a technical flaw:
* Layer 13 is the global LayerNorm, which has only 384 parameters in ViT-Tiny.
* As the authors admit, because of this extremely low parameter count, the projected Hessian matrix $A^{13}$ has a tiny trace and exhibits extreme numerical ill-conditioning, with a condition number exceeding $10^4$.
* Solving an ill-conditioned $4 \times 4$ system over a tiny, stochastic 32-sample calibration set naturally yields wild, explosive coefficients. 
* This is not a "beautiful active cancellation mechanism"—it is simply a standard numerical artifact of severe ill-conditioning and overfitting to calibration noise. When Ridge/Lasso regularization is applied to stabilize this layer, these wild coefficients are suppressed, but the final accuracy of these regularized models (e.g., ACM-GlobalNorm at 57.76% or Lasso ACM-GlobalNorm at 57.52%) drops significantly below standard Task Arithmetic. This suggests that these extreme coefficients were numerical noise rather than a physically meaningful representational feature.

### D. Complete Failure of Multi-Layer Coordination
The authors assume that cross-layer second-order interactions are negligible (Assumption 3.1) to decouple the optimization layer-by-layer. When they attempt to resolve the resulting cross-layer coupling mismatch sequentially using a block Gauss-Seidel coordinated solver, the method collapses to **36.65%** Joint Average accuracy.

This collapse exposes a major methodological flaw: ACM's layer-wise decoupling assumption is fundamentally incompatible with deep sequential models. Modifying early layers sequentially causes the representation space to drift, rendering the local Hessian approximations evaluated at the original expert checkpoints completely invalid for downstream layers. This indicates that ACM is incapable of modeling sequential, cross-layer parameter interactions.

### E. Toy-Scale Physical Validation
The physical validation is exceptionally weak in terms of scale and diversity:
* The authors evaluate strictly on **ViT-Tiny** (5.7M parameters) and four low-resolution toy classification datasets (**MNIST, FashionMNIST, CIFAR-10, SVHN**). Modern weight consolidation research focuses on large language models (LLMs) with billions of parameters or large vision-language models on complex, real-world tasks.
* The "low-data deployment regime" of training for only 10 epochs on 2048 samples is highly artificial and designed to keep task vectors small, artificially protecting the local quadratic Taylor expansion from breaking down.
* There is no empirical validation showing that the purely hypothetical "Scaling Analysis" in Section 4.5 actually holds on larger models (e.g., ViT-Base, ResNet-50) or other modalities (such as NLP).

### F. Potential Strawman Comparison on Test-Time Adaptation
The Test-Time Adaptation (TTA) baselines (AdaMerging, PolyMerge, RegCalMerge) experience severe generalization collapse on physical ViT-Tiny, dropping as low as 38.96%. While the authors blame transductive overfitting, it is highly likely that forcing these baselines to optimize on an extremely restricted, low-data 32-sample calibration batch for only 15 steps represents a strawman experimental setup designed to guarantee their failure. Under standard streaming deployment settings, these methods are given sufficient data to adapt stably without representational collapse.

---

## 4. Questions and Detailed Comments for the Authors

1. **Practical Utility:** Given that standard, zero-calibration Task Arithmetic matches or beats almost all variants of ACM on physical models with zero computational or calibration overhead, why should practitioners adopt the highly complex, calibration-dependent, and computationally heavier ACM framework?
2. **The Local-Global Gap:** Since the local-global approximation error scales cubically ($O(V_{\max}^3)$) with task vector magnitude, how do the authors propose to apply ACM to fully converged, state-of-the-art models (such as large language models or large vision backbones) where task vectors are inherently large and local quadratic approximations are mathematically shown to fail?
3. **Scale of Evaluation:** Can the authors provide empirical results on larger backbones (e.g., ViT-Base or ResNet-50) and more complex datasets to substantiate their purely theoretical "Scaling Analysis" in Section 4.5?
4. **Active Cancellation Stability:** Is the "active cancellation" observed in Layer 13 actually stable under different random calibration seeds, or does it fluctuate wildly due to the $>10^4$ condition number of the projected Hessian? If it is stable, why does regularizing/pruning these wild coefficients (via Ridge or Lasso) result in a significant drop in multi-task accuracy?
5. **TTA Baselines:** Why were the TTA baselines restricted to an extremely low-data 32-sample batch? Can the authors provide comparisons where TTA baselines are allowed to adapt on larger calibration streams as recommended in their original publications?

---

## 5. Final Ratings and Recommendation

* **Soundness:** **Fair.** The mathematical proofs are correct, but the physical application exhibits major methodological flaws, including severe numerical ill-conditioning at LayerNorm layers, catastrophic collapse of multi-layer sequential coordination, and a fundamental mismatch between local quadratic surrogates and global weight consolidation (the local-global gap).
* **Presentation:** **Excellent.** The paper is outstandingly written, with clear mathematical exposition, high-quality figures and tables, and commendable transparency regarding negative results and limitations.
* **Significance:** **Fair.** While the paper has high theoretical value by formalizing the local-global gap and cross-layer coupling mismatches, its practical significance is low because the proposed method fails to consistently outperform standard, curvature-blind uniform scaling (Task Arithmetic) and is only validated on toy-scale configurations.
* **Originality:** **Good.** The formulation of full non-diagonal Hessian curvature within a low-dimensional task vector subspace and the robust gradient subtraction finite-difference scheme are elegant and clever contributions.

### Overall Recommendation: 3: Weak Reject
The paper is a beautifully written, mathematically elegant, and highly honest piece of work. The theoretical formulations of the subspace projection, robust finite-difference estimation, and local-global gap are outstanding. However, the work has major, unresolved weaknesses: the proposed method fails to consistently beat a simple uniform scaling baseline in physical settings, the sequential coordination variant collapses, the physical validation is restricted to a toy scale, and the extreme coefficients are highly likely to be numerical noise from ill-conditioning rather than a real feature. Overall, the serious empirical and practical weaknesses outweigh the theoretical merits. The paper requires a major revision—specifically, expanding the empirical evaluation to large-scale models/tasks and finding a way to resolve the local-global gap—before it can be considered for acceptance.
