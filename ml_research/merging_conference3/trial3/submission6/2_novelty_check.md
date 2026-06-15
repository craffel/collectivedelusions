# Intermediate Review Task 2: Novelty and Delta Check (`2_novelty_check.md`)

## 1. Characterizing the 'Delta' from Prior Work
To evaluate the novelty of Curvature-Aware Analytical Model Merging (ACM), we compare its conceptual and mathematical formulation against established paradigms in parameter consolidation:

### A. Delta from Task Arithmetic (Ilharco et al., 2022)
- **Prior Work:** Task Arithmetic blends task vectors uniformly with a single, static scale factor ($\lambda$) applied across all layers of the network. This assumes parameter interactions are uniform and independent, leading to representation interference and performance degradation when task vectors collide.
- **ACM Delta:** ACM rejects uniform scaling. It computes unique, layer-wise merging coefficients ($\Lambda^l$) for each task expert. It explicitly models cross-task directional alignments and parameter sensitivities, allowing for complex layer-wise scaling, including negative scaling factors that actively cancel representation interference.

### B. Delta from Fisher Merging (Matena & Raffel, 2022)
- **Prior Work:** Fisher Merging utilizes the diagonal of the Fisher Information Matrix (FIM) as a proxy for second-order loss sensitivity to weight expert parameters.
- **ACM Delta:** While essential to make computation over millions of parameters tractable, a diagonal approximation completely discards the **off-diagonal cross-term Hessian entries** representing parameter coupling. ACM resolves this fundamental bottleneck. By projecting search directions onto the low-dimensional $K$-dimensional subspace spanned by the task vectors, ACM can compute and invert the **full, non-diagonal, projected Hessian** with zero diagonal approximation.

### C. Delta from Test-Time Adaptation (AdaMerging, RegCalMerge, PolyMerge)
- **Prior Works:** These state-of-the-art adaptive methods optimize layer coefficients at deployment time. Since labels are unavailable, they rely on unsupervised test-time entropy minimization on local target data streams.
- **ACM Delta:** These methods require hundreds of forward-backward optimization passes, defeating the "training-free" promise, and are highly prone to transductive overfitting (the Overfitting-Optimizer Paradox) and sacrificial task bias. ACM completely departs from this paradigm. It is **entirely training-free at deployment**, solving for the optimal coefficients analytically in a single step using a tiny, one-time calibration dataset.

---

## 2. Characterization of Novelty
We evaluate the novelty of ACM across its core mathematical and methodological dimensions:

### A. Conceptual Novelty: High
The core insight—that we can retain the full, non-diagonal, cross-parameter Hessian curvature by projecting the massive $D$-dimensional parameter space onto the extremely low-dimensional $K$-dimensional subspace of task vectors—is conceptually significant. Rather than accepting diagonal approximations as a given bottleneck for tractability, the authors recognize that the search space itself is low-dimensional, making the full projected Hessian matrix ($K \times K$) easily computable and invertible ($O(K^3)$ per layer). This is a mathematically elegant and powerful formulation that bridges the gap between loss-landscape geometry and parameter space interpolation.

### B. Methodological Novelty: High
The proposed **Gradient Subtraction Finite-Difference Scheme** is a highly novel and mathematically rigorous technique. Standard finite-difference methods assume the expert model has reached perfect local convergence ($\nabla \mathcal{L}_k(W_k) \approx 0$). In practice, residual gradients are non-zero, causing severe numerical instability and noise amplification by $1/\epsilon$ when selecting small perturbation scales. ACM's explicit calculation and subtraction of the unperturbed expert gradient $g_{k,0}$ mathematically cancels this residual gradient, bounding the truncation error to a clean $O(\epsilon)$ regardless of convergence. This represents a significant contribution to the numerical stability of curvature-estimation methods in deep learning.

### C. Architectural Novelty: Moderate to High
The distinction between layer-wise scale normalization (ACM-Norm) and global scale normalization (ACM-GlobalNorm) is a valuable architectural contribution. By recognizing that normalizing layer-by-layer flattens and destroys the natural relative parameter sensitivity profiles of different layers within a task, the authors introduce global trace normalization. This preserves the relative depth-wise sensitivity structure of the network. Physical validation demonstrates that ACM-GlobalNorm outperforms ACM-Norm on individual tasks where global depth-wise scale preservation is key, specifically on FashionMNIST (70.02% vs 69.24%) and CIFAR-10 (77.05% vs 76.27%), confirming the physical value of preserving relative depth-wise sensitivity.

---

## 3. Overall Assessment of Novelty
The novelty of this paper is **significant**. It is not a heuristic or incremental combination of existing methods, but a principled reformulation of parameter fusion from first-principles mathematical geometry. The delta from prior works is clear, theoretically justified, and mathematically proven. The paper succeeds in moving the model merging field away from unsupervised, unstable test-time adaptation heuristics towards formal, stable, and training-free closed-form solutions.
