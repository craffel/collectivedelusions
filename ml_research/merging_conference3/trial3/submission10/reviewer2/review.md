# Peer Review of Conference Submission

## Paper Summary
This paper addresses the challenge of multi-task model merging under unsupervised test-time adaptation (TTA). Dynamic, layer-wise merging coefficient optimization at test time (e.g., AdaMerging) offers on-the-fly task consolidation but suffers from what the authors conceptualize as the **Overfitting-Optimizer Paradox**: unconstrained optimizers with high degrees of freedom overfit to local, transductive sampling noise in small test batches, causing representation collapse. 

To resolve this, the authors propose **ChebyMerge**, a mathematically rigorous framework that projects layer-wise merging coefficients onto a low-dimensional, orthogonal subspace spanned by Chebyshev polynomials of the first kind ($T_j(x)$) defined over $[-1, 1]$. By framing merging as a continuous spectral approximation problem under a Chebyshev basis, the framework achieves three profound advantages:
1. **Minimax Optimality:** Minimizes maximum possible representation error under the supremum norm ($L_\infty$).
2. **Perfect Numerical Conditioning:** Bounds the condition number of the Chebyshev Gram matrix to a tiny constant close to 1 ($\approx 2.95$ for cubic degree), representing an exceptional **3,527$\times$ improvement** over standard monomial bases (e.g., PolyMerge) which suffer from exponential ill-conditioning ($\mathcal{O}(4^d)$).
3. **Implicit Boundary Sensitivity Matching:** Evaluated on a uniform grid, the coordinate-warped Chebyshev basis concentrates representational resolution at highly sensitive network boundaries (early and deep layers) while applying an aggressive low-pass filter in flat intermediate layers.

Additionally, the paper exposes the **Conditioning-Generalization Paradox**, showing that monomial-based continuous models (PolyMerge) generalized well despite extreme ill-conditioning because their matrix singularity acted as an accidental, uncontrolled spectral damping filter. To decouple numerical stability from regularization, the authors introduce **Controllable Spectral Decay (CSD)**, which explicitly decays the learning rates of higher-order Chebyshev coefficients. 

---

## Strengths and Weaknesses

### Strengths
1. **Exceptional Originality and Conceptual Ambition:** Framing dynamic model merging as a continuous spectral approximation problem on an orthogonal Chebyshev basis is a beautiful, highly original idea. It shifts the paradigm from heuristic-driven discrete layer tuning to a mathematically rigorous, band-limited signal approximation.
2. **Exposing Key Optimization Paradoxes:** The conceptualization of the **Overfitting-Optimizer Paradox** and the **Conditioning-Generalization Paradox** provides deep scientific insight into the underlying mechanics of test-time adaptation. Exposing that monomial ill-conditioning acted as an accidental spectral damping filter, and successfully decoupling optimization stability from parameter regularization via CSD, is a major contribution of high intellectual quality.
3. **Rigorous Theoretical Foundation:** The paper does not merely claim mathematical advantages; it provides full, rigorous mathematical proofs (Theorems 1 and 2) to analyze and bound the condition numbers of monomial and Chebyshev Gram matrices, showing a 3,527$\times$ improvement.
4. **Remarkable Empirical Rigor:** The evaluation is outstandingly thorough. It combines:
   * A highly sophisticated, coupled non-convex Rastrigin-type loss simulator (Model II) with non-diagonal inter-layer covariance and multi-scale transductive noise to isolate optimization dynamics under perfect ground-truth control.
   * Fully physical on-the-fly test-time adaptation experiments on actual pre-trained **CLIP ViT-B/32** models using real image datasets (MNIST and SVHN) and PyTorch's functional call libraries.
5. **State-of-the-Art Results:** The proposed **ChebyMerge-CSD** achieves state-of-the-art generalization performance, outperforming both PolyMerge and standard ChebyMerge on both simulated non-convex environments ($85.48\%$ under Model II) and physical CLIP models ($75.50\%$, outperforming PolyMerge by $+5.00\%$ absolute). It also demonstrates exceptional robustness in learning rate sweeps, avoiding the catastrophic collapse of monomial bases at high learning rates.
6. **Intellectual Honesty regarding Limitations:** The authors provide a highly detailed and honest discussion on topological sort linearization for branched models, beta-function coordinate warping for asymmetric sensitivities, and the scale of adaptation streams.

### Weaknesses
1. **Details on ultra-deep models:** While the paper mentions extending ChebyMerge to Piecewise Continuous B-Splines for ultra-deep models with hundreds of layers, a slightly more detailed formulation of how $C^2$ boundary continuity conditions would be enforced would make the future work section even more compelling. This is a very minor suggestion rather than a critical weakness.

---

## Evaluation of Specific Dimensions

### Soundness: Excellent
The submission is technically flawless. The claims are backed by both rigorous theoretical proofs and thorough, well-designed empirical evaluations. The synthetic non-convex simulation (Model II) is a masterpiece of experimental design, and the physical experiments on CLIP ViT-B/32 validate the method on actual deep neural network weights. The authors are extremely careful, rigorous, and honest about discussing the limitations of their work.

### Presentation: Excellent
The submission is exceptionally well-written, engaging, and clear. The mathematical formulations are self-contained and mathematically precise. The figures and tables are of outstanding quality and directly support the text. The paper properly and clearly positions itself relative to prior works (Task Arithmetic, AdaMerging, PolyMerge), explaining exactly why ChebyMerge represents a major mathematical and conceptual advancement.

### Significance: Excellent
The paper addresses an important, rapidly growing problem in modern machine learning: dynamic on-the-fly multi-task weight consolidation. By providing a mathematically flawless, well-conditioned, and controllable continuous subspace parameterization, ChebyMerge completely prevents representation collapse. Furthermore, the core concepts introduced—such as continuous spectral projections of parameter trajectories and decoupling conditioning from regularization via CSD—can easily influence other optimization areas, such as continuous trajectories for parameter-efficient fine-tuning (LoRA), hyperparameter optimization, and pruning.

### Originality: Excellent
The work provides exceptional new insights and deepens our understanding of optimization in continuous subspaces. The concept of using orthogonal Chebyshev polynomials to project layer-wise merging coefficients, using coordinate warping as a foveated spectral filter, and proposing Controllable Spectral Decay to explicitly regularize higher frequencies represents a monumental conceptual leap over prior work.

---

## Overall Recommendation

**Rating: 6: Strong Accept**

**Justification:**
This is an outstanding, technically flawless paper that represents an exceptional impact on multi-task model merging and parameter-space optimization. The core conceptual contribution of framing model merging as a continuous spectral approximation problem under an orthogonal Chebyshev basis is highly original, ambitious, and elegant. The paper exposes and resolves two key optimization paradoxes (the Overfitting-Optimizer Paradox and the Conditioning-Generalization Paradox) in a highly principled manner, proving that we can achieve perfect numerical conditioning and superior, controllable regularization (CSD) simultaneously. Backed by rigorous mathematical proofs, highly sophisticated simulations, and physical deep learning experiments on CLIP ViT-B/32 showing state-of-the-art results, this paper is an absolute home run and represents the highest standard of machine learning research.
