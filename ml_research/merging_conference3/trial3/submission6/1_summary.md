# Intermediate Review Task 1: Paper Summary (`1_summary.md`)

## 1. Main Topic and Scope
This paper addresses the challenge of **multi-task model merging (parameter fusion)**, which aims to consolidate multiple task-specific expert neural networks (independently fine-tuned from a common pretrained base model) into a single, cohesive network directly in the parameter space without any additional training or fine-tuning.

The authors critique current state-of-the-art adaptive merging methods—such as AdaMerging, RegCalMerge, and PolyMerge—which rely on **Test-Time Adaptation (TTA)** guided by heuristic, unsupervised surrogate objectives (such as prediction entropy minimization on unlabeled target streams). The paper highlights major vulnerabilities in the TTA paradigm:
- **Heuristic Surrogate Objective:** Entropy minimization is an indirect proxy that is not mathematically guaranteed to minimize the joint multi-task supervised loss.
- **Transductive Overfitting (Overfitting-Optimizer Paradox):** Adapting coefficients on small, unlabeled batches of test data leads to catastrophic generalization collapse on unseen target samples.
- **Sacrificial Task Bias:** Test-time entropy minimization can heavily favor one task at the severe expense of others due to the lack of joint objective constraints.
- **Prohibitive Computational Overhead:** TTA requires hundreds of forward-backward optimization steps at deployment time, violating the "training-free" promise of model merging.

To overcome these limitations, the paper introduces **Curvature-Aware Analytical Model Merging (ACM)**, a training-free framework that computes optimal, layer-wise merging coefficients analytically in a single step using the local second-order loss-landscape geometry of individual task experts.

---

## 2. Methodology and Technical Approach
ACM approaches parameter fusion from first-principles mathematical geometry, formulating the joint loss minimization as a quadratic optimization problem. The core pillars of the methodology include:

1. **Subspace-Restricted Second-Order Taylor Expansion:**
   The joint multi-task loss is approximated around each task expert's local parameter state $W_k$ using a complete second-order Taylor expansion (including unperturbed expert gradients to handle incomplete convergence):
   $$\mathcal{L}_k(W(\Lambda)) \approx \mathcal{L}_k(W_k) + \nabla \mathcal{L}_k(W_k)^T (W(\Lambda) - W_k) + \frac{1}{2} (W(\Lambda) - W_k)^T H_k (W(\Lambda) - W_k)$$
   where $H_k = \nabla^2 \mathcal{L}_k(W_k) \in \mathbb{R}^{D \times D}$ is the full, non-diagonal Hessian matrix.

2. **Low-Dimensional Subspace Projection:**
   Modeling the full $D$-dimensional Hessian is a prohibitive $O(D^2)$ space and $O(D^3)$ time operation. ACM restricts the search space for merging coefficients $\Lambda^l \in \mathbb{R}^K$ at each layer $l$ strictly to the $K$-dimensional subspace spanned by the task vectors $V^l = [v_1^l, \dots, v_K^l] \in \mathbb{R}^{d_l \times K}$ (where $K \ll D$, typically $K \le 5$). 
   
   Under a layer block-diagonal Hessian assumption (Assumption 3.1), the optimization decomposes into $L$ independent quadratic minimization systems. Crucially, by projecting onto this low-dimensional subspace, ACM computes the **full, non-diagonal, cross-parameter Hessian curvature** along the task vector directions with **zero diagonal approximation**.

3. **Closed-Form Optimal Analytical Solution:**
   By setting the gradient of the Ridge-regularized (Tikhonov) quadratic surrogate to zero, ACM derives an exact closed-form analytical solution for the optimal layer-wise coefficients:
   $$\Lambda^{l, *} = (A^l + \gamma I)^{-1} (b^l - d^l)$$
   where $A^l$ is the projected Hessian matrix, $b^l$ is the projected target vector, $d^l$ is the projected gradient vector, and $\gamma > 0$ is the regularization strength.

4. **Scale and Global Normalization (ACM-Norm & ACM-GlobalNorm):**
   - **ACM-Norm:** Normalizes each task's projected Hessian contribution by its trace layer-by-layer to prevent a single dominant task (with large gradient/Hessian scale) from introducing sacrificial task bias.
   - **ACM-GlobalNorm:** Normalizes by the global trace across all layers, resolving layer-wise scale imbalance while fully preserving the natural depth-wise relative parameter sensitivity profile of the network.

5. **Gradient Subtraction Finite-Difference Scheme:**
   To estimate $(v_i^l)^T H_k^l v_j^l$ efficiently without forming the Hessian, ACM uses a cheap finite-difference perturbation scheme. By explicitly computing and subtracting the unperturbed expert gradient $g_{k,0}^l$, ACM cancels the residual gradient and achieves complete immunity to incomplete checkpoint convergence, bounding the truncation error to $O(\epsilon)$:
   $$(v_i^l)^T H_k^l v_j^l \approx \frac{1}{\epsilon} \langle v_i^l, g_{k,j}^l - g_{k,0}^l \rangle$$

---

## 3. Key Findings and Quantitative Results
The paper evaluates ACM through controlled simulation sweeps and physical validation on Vision Transformers:

- **Simulation Sweeps (30 Seeds):**
  - **Model I (Convex Landscape, diagonal Hessian):** ACM (87.46% $\pm$ 0.07%) performs comparably to the best TTA method, PolyMerge (87.72% $\pm$ 0.07%).
  - **Model II (Coupled Non-Convex Landscape, dense Hessian):** Under realistic parameter coupling, TTA methods degrade heavily due to optimization instability. ACM achieves **87.18% $\pm$ 0.26%**, significantly outperforming PolyMerge by **+1.69%** and AdaMerging by **+8.11%** while maintaining extremely low variance.
- **Physical Validation on ViT-Tiny (MNIST, FashionMNIST, CIFAR-10, SVHN):**
  - **Comparison to Diagonal Curvature:** Proposed **ACM-GlobalNorm** achieves **57.76% Joint Average accuracy**, outperforming diagonal Fisher Merging (**56.03%**) by **+1.73% absolute**, confirming the importance of off-diagonal (cross-parameter) second-order interactions.
  - **Comparison to Test-Time Adaptation:** ACM-GlobalNorm significantly outperforms PolyMerge (**38.96%**) and AdaMerging (**55.42%**), which suffer from severe generalization collapse due to transductive overfitting to small unsupervised test batches on physical backbones.
  - **Vanilla and Lasso Performance:** Vanilla ACM achieves **60.89%**, outperforming standard, exhaustively tuned Task Arithmetic (**60.72%**). Lasso ACM (Vanilla) achieves a highly competitive average of **60.67%**, proving that L1 regularization successfully stabilizes ill-conditioned bottleneck layers like LayerNorm.
  - **The Local-Global Gap:** While Vanilla ACM (60.89%) beats Task Arithmetic (60.72%), the normalized variant ACM-GlobalNorm (57.76%) is slightly lower than Task Arithmetic. The authors discuss this transparently, demonstrating that as task vectors grow during fine-tuning, local second-order Taylor approximations break down on highly non-convex physical neural manifolds, whereas Task Arithmetic acts as a robust global regularizer.

---

## 4. Explicitly Claimed Contributions (with Evidence)
1. **Theoretical Rigor:** Establishes a formal mathematical foundation for multi-task parameter fusion, formulating joint loss minimization as a quadratic optimization problem over task-specific local Hessian geometries. (Supported by Section 3.1-3.3 and Theorem 3.2).
2. **Low-Dimensional Subspace Projection:** Proves that restricting parameter search to the $K$-dimensional subspace of task vectors allows us to model full, non-diagonal cross-parameter Hessian interactions with zero diagonal approximation. (Supported by Section 3.2, Section 3.6, and Appendix B).
3. **Training-Free Closed-Form Solution:** Derives an exact analytical closed-form solution under Ridge regularization, eliminating test-time training, transductive overfitting, and sacrificial task bias. (Supported by Section 3.3 and Appendix B).
4. **Empirical Characterization & Transparency:** Evaluates ACM via simulations and physical benchmarks on Vision Transformers. Provides a highly transparent and detailed characterization of when local second-order approximations are optimal versus when simple uniform interpolation (Task Arithmetic) exhibits regularizing advantages on physical neural manifolds. (Supported by Sections 4.2, 4.3, and 4.5).
