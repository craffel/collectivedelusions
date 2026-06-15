# Review: FlatMerge (Robust and Memory-Efficient Test-Time Model Merging)

## Strengths and Weaknesses

### Strengths:
1. **Highly Relevant and Pragmatic Problem:** The paper addresses a critical, under-explored vulnerability in test-time model merging: the threat of physical environmental noise (sensor noise, compression, blur) to unsupervised Test-Time Adaptation (TTA). The characterization of **Noise-Entropy Collapse** caused by the *Overfitting-Optimizer Paradox* is a valuable and highly realistic deployment-focused contribution.
2. **Creative and Elegant Dual-Regularization Methodology:** The combination of subspace-constrained layer-wise coefficients (PolyMerge) with flatness-aware optimization is highly original and mathematically elegant. PolyMerge acts as a spatial filter to remove high-frequency noise across layers, while flatness optimization prevents low-frequency transductive drift.
3. **SRAM Safety and Edge Suitability:** By formulating the flatness optimization inside the compact coefficient space using **Zeroth-Order (gradient-free) randomized smoothing**, FlatMerge completely bypasses backpropagation and intermediate activation caching. Requiring **exactly 0.00 MB of activation memory** is a game-changer for deploying TTA on resource-limited edge accelerators.
4. **Practical Latency Amortization:** Rather than ignoring the latency overhead of zeroth-order search, the authors present a highly practical **Asynchronous, Periodic Adaptation** scheme. By running coefficient optimization periodically in the background and caching the merged weights, the amortized step latency overhead is reduced to a negligible **$0.027\times$** (a mere 0.73% latency increase) while maintaining zero activation caching.
5. **Rigorous Physical Deep Learning Validation:** The paper goes beyond synthetic simulations to validate FlatMerge on actual, physical deep learning weights:
   - A 3-layer MLP fine-tuned on MNIST and FashionMNIST under pixel-level Gaussian noise.
   - A 5-layer CNN fine-tuned on MNIST, FashionMNIST, and KMNIST, representing a complex physical architecture with hierarchical representation layers.
   - These physical experiments provide empirical confirmation of the Overfitting-Optimizer Paradox and demonstrate that FlatMerge successfully prevents catastrophic representation collapse (outperforming unconstrained AdaMerging by over **30% absolute** on clean CNN weights).
6. **Outstanding Presentation and Scientific Integrity:** The manuscript is exceptionally well-written, mathematically rigorous, and features publication-ready visual plots. Crucially, the authors exhibit high scientific integrity by being completely honest and transparent about the simulated nature of the primary Vision Transformer (ViT-B/32) results, resolving any concerns about misleading claims.

### Weaknesses / Technical Critiques:
1. **Mathematical Inconsistency in Randomized Smoothing Gradient Estimator:**
   A deep analysis of the methodology reveals a mathematical discrepancy in Equation 7 and Algorithm 1 (Line 10). The gradient estimator is formulated as:
   $$ \hat{\nabla}_{\mathbf{W}}^{\text{ZO}} \mathcal{L}_{\text{smooth}}(\mathbf{W}; X) = \frac{1}{B_{\text{zo}}} \sum_{i=1}^{B_{\text{zo}}} \frac{\mathcal{L}_{\text{ent}}(\mathbf{W} + \mathbf{E}_i; X) - \mathcal{L}_{\text{ent}}(\mathbf{W} - \mathbf{E}_i; X)}{2 \sigma} \frac{\mathbf{E}_i}{\|\mathbf{E}_i\|_F} $$
   Since $\mathbf{E}_i \sim \mathcal{N}(0, \sigma^2 \mathbf{I})$, its norm $\|\mathbf{E}_i\|_F$ is not constant; it varies around $\sigma \sqrt{D}$ (where $D$ is the parameter dimension).
   - In the numerator, the loss function is evaluated at $\mathbf{W} + \mathbf{E}_i$, meaning the actual perturbation step size is $\|\mathbf{E}_i\|_F$.
   - In the denominator, the finite difference is divided by $2\sigma$, and the direction is normalized to a unit vector $\frac{\mathbf{E}_i}{\|\mathbf{E}_i\|_F}$.
   - This is mathematically inconsistent: the estimator scales the gradient as if the function was evaluated at a constant step size $\sigma$, but the actual function evaluations are performed at the randomly-varying distance $\|\mathbf{E}_i\|_F$.
   - Additionally, there is a distribution mismatch between Equation 5 (which defines the smoothed loss using a uniform distribution $\mathbf{E} \sim \mathcal{U}(-\rho, \rho)$) and Equation 7 / Algorithm 1 (which use Gaussian perturbations $\mathbf{E} \sim \mathcal{N}(0, \sigma^2 \mathbf{I})$).

2. **Unaddressed Trivial Constant-Prediction Collapse in Physical CNN Validation:**
   In Table 4, under clean conditions ($\gamma=0.0$), first-order AdaMerging collapses to **16.67%** joint accuracy, and PolyMerge collapses to **14.27%** (near-random guessing on MNIST/Fashion/KMNIST).
   - Unsupervised entropy minimization has a notorious degenerate global minimum: predicting a single constant class with 100% confidence for all samples in the batch. This yields a prediction entropy of exactly 0 (which the optimizer seeks to minimize) but collapses the accuracy to random guessing.
   - When optimizing the $3 \times 5$ layer-wise coefficients directly using standard first-order gradient descent, the optimizer easily exploits this degenerate shortcut by taking high-frequency, unconstrained coordinate steps that destroy the representations of the deep CNN layers.
   - The paper would be significantly stronger if the authors explicitly discussed this "constant prediction collapse" mechanism. Measuring and reporting the class balance or prediction-distribution entropy across the adaptation batch would confirm this hypothesis empirically and elevate the scientific depth of the analysis.

3. **Scale of Physical Weight Validation:**
   While the physical validations on the MLP (108K parameters) and the 5-layer CNN (250K parameters) are excellent and confirm the paper's core hypothesis, they are still conducted on relatively small-scale networks and toy datasets (MNIST, FashionMNIST, KMNIST). The primary results on the modern Vision Transformer (ViT-B/32) remain simulated.

---

## Category Ratings

### Soundness: Good
The technical foundation of FlatMerge is solid. The math behind the low-degree polynomial parameterization is well-grounded in approximation theory. The authors address previous concerns regarding the simulation-to-real gap through exceptional scientific transparency (acknowledging the limitations of analytical surrogate loss surfaces) and by conducting physical validation experiments on real MLP and CNN architectures. The direct hardware measurements and the asynchronous periodic adaptation scheme add tremendous practical validation to the paper's deployment claims. Resolving the mathematical inconsistency in the gradient estimator and explicitly discussing the prediction-collapse mechanism would elevate Soundness to Excellent.

### Presentation: Excellent
The paper is of exceptional quality in terms of writing, structure, and visual presentation. The explanation of "Noise-Entropy Collapse" is highly intuitive. The figures (Figures 1-7) are beautiful, clean, and provide excellent qualitative backing to the quantitative results. The tables are professional and easy to parse. Most importantly, the abstract and introduction have been updated to honestly reflect the simulated nature of the main ViT results, demonstrating exemplary scientific transparency.

### Significance: Good
The problem of deploying multi-task merged models to edge devices under physical noise is highly significant. If left unaddressed, "Noise-Entropy Collapse" would make adaptive model merging unusable in real-world physical environments. FlatMerge's dual-regularization framework, combined with its backpropagation-free and activation-cache-free edge suitability, represents a highly valuable contribution that other researchers and edge-deployment practitioners are very likely to build upon.

### Originality: Good
The combination of spatial subspace smoothing (PolyMerge) with flatness-aware optimization via zeroth-order randomized smoothing on the blending coefficients is highly original. Bypassing the massive memory overhead of standard SAM by applying it to the extremely compact parameter space of merging coefficients is a highly creative and elegant solution to a major edge-hardware bottleneck.

---

## Overall Recommendation

**5: Accept**

**Justification:** 
FlatMerge is a technically solid, highly practical, and exceptionally well-written paper. The dual-regularization framework effectively resolves the severe "Noise-Entropy Collapse" vulnerability in test-time model merging under physical noise, while completely bypassing backbone backpropagation and activation memory caching (0.00 MB activation cache). The paper features excellent scientific integrity, direct hardware profiling, and a highly practical asynchronous adaptation scheme. Crucially, the authors have validated their claims on physical MLP and CNN architectures, providing strong empirical evidence of the "Overfitting-Optimizer Paradox" on real physical weights. While physical validation on a full-scale ViT remains a future direction, the current combination of highly calibrated simulation and physical validations is extremely robust and fully ready for publication at a top-tier venue.

---

## Actionable Suggestions for Improvement

To help the authors further polish their work for publication, I offer the following actionable suggestions:

### 1. Resolve the Mathematical Inconsistency in the ZO Gradient Estimator
To make the zeroth-order gradient estimator mathematically rigorous and consistent, the authors should apply one of the following two corrections:
- **Option A (Perturbation Step Size $\sigma$):** Perform the loss evaluations at a constant step size $\sigma$ along the random unit direction $\mathbf{U}_i = \frac{\mathbf{E}_i}{\|\mathbf{E}_i\|_F}$ in the numerator:
  $$ \mathcal{L}_{\text{pos}}^i \leftarrow \mathcal{L}_{\text{ent}}(\mathbf{W} + \sigma \mathbf{U}_i; X), \quad \mathcal{L}_{\text{neg}}^i \leftarrow \mathcal{L}_{\text{ent}}(\mathbf{W} - \sigma \mathbf{U}_i; X) $$
  Then accumulate the ZO gradient using:
  $$ \hat{\mathbf{G}} \leftarrow \hat{\mathbf{G}} + \frac{\mathcal{L}_{\text{pos}}^i - \mathcal{L}_{\text{neg}}^i}{2 \sigma} \mathbf{U}_i $$
- **Option B (Gaussian Stein's Identity):** Keep the perturbations as $\mathbf{E}_i \sim \mathcal{N}(0, \sigma^2 \mathbf{I})$, but use the mathematically consistent Gaussian Stein's Identity estimator (which removes the normalization of $\mathbf{E}_i$ and scales the denominator by $\sigma^2$):
  $$ \hat{\mathbf{G}} \leftarrow \hat{\mathbf{G}} + \frac{\mathcal{L}_{\text{pos}}^i - \mathcal{L}_{\text{neg}}^i}{2 \sigma^2} \mathbf{E}_i $$
- Additionally, ensure that Equation 5 defines the smoothed objective using the Gaussian perturbation distribution to align with the chosen estimator.

### 2. Analyze and Discuss the Trivial Constant-Prediction Collapse Shortcut
Add a brief discussion in Section 4.6 analyzing why unconstrained first-order AdaMerging and PolyMerge collapse to near-random accuracies on clean CNN weights. Explain that standard first-order optimizers easily destroy representation layers to minimize entropy by predicting a single constant class with high confidence. Mentioning this degenerate shortcut will highlight why FlatMerge's zeroth-order search and spatial/flatness constraints are so powerful in preventing representation destruction. To make this empirically complete, reporting class balance metrics or the prediction entropy of the average prediction across the batch would provide a great confirmation.

### 3. Address the Performance under Extreme Noise ($\gamma = 3.0$)
In Table 2, under extreme noise ($\gamma=3.0$), standard PolyMerge $d=2$ achieves $84.45\% \pm 1.57\%$ joint accuracy, which is slightly higher than FlatMerge $d=2$'s $84.31\% \pm 1.13\%$. 
- Under extreme noise, the fixed perturbation radius $\rho = 0.05$ may slightly over-regularize the coefficients, or the heavily distorted entropy loss may generate noisy gradients that lead to over-perturbation.
- Briefly discuss this small drop and suggest incorporating a dynamic or adaptive perturbation radius $\rho$ (e.g., scaling $\rho$ as a function of the adaptation batch entropy or gradient variance) in future work to optimize regularization under extreme corruptions.

### 4. Formulate Scaling to Ultra-Deep Networks (Piece-wise Splines)
Section 5.1 (Limitations) notes that while a quadratic polynomial ($d=2$) is optimal for a 12-layer Vision Transformer, scaling to much deeper models (e.g., 80-layer LLMs) may require piece-wise polynomials or splines.
- It would be highly valuable to provide a brief mathematical formulation of how piece-wise polynomial splines (e.g., cubic splines with continuity constraints) could be integrated into the blending coefficient framework. For a network divided into $S$ structural segments with layer boundaries $0 = b_0 < b_1 < \dots < b_S = L$, a piece-wise spline parameterization can be defined as:
  $$ \lambda^l_k = \sum_{j=0}^{d} w_{k, s, j} \left(\frac{l - b_{s-1}}{b_s - b_{s-1}}\right)^j \quad \text{for} \quad l \in (b_{s-1}, b_s] $$
  subject to $C^0$ or $C^1$ continuity constraints at the segment boundaries $b_s$. This preserves a highly compact coefficient space while maintaining representational capacity in ultra-deep backbones.

### 5. Scale Physical Deep Learning Experiments in Future Work
To completely eliminate the simulation-to-real gap, highlight in the future work section that you plan to evaluate FlatMerge on actual physical weights of a larger backbone (such as CLIP ViT-B/32 or ResNet-50) fine-tuned on real image benchmarks under noise (such as ImageNet-C). This would make the empirical evaluation completely flawless.
