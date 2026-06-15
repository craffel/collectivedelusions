# Novelty Check: PolyMerge & SplineMerge

## 1. Contextualizing within Model Merging Literature
Model merging has evolved rapidly from static, uniform scaling to dynamic, adaptive layer-specific scaling. To place the proposed work in this context, we analyze its positioning relative to key foundational and state-of-the-art works:
1. **Task Arithmetic (Ilharco et al., ICLR 2023)**: Merges task vectors using a fixed, uniform scalar coefficient ($\lambda_k$) across all layers. This approach is highly stable but sub-optimal as it completely ignores the structural and functional heterogeneity of different layers.
2. **AdaMerging (Yang et al., ICLR 2024)**: Introduces adaptive model merging by dynamically optimizing separate, independent layer-wise (or projection-wise) coefficients at test-time via unsupervised entropy minimization. While highly expressive, it introduces high parameter dimensionality ($L$ parameters per task) and is vulnerable to the newly identified "Overfitting-Optimizer Paradox" and "Degenerate Entropy Minimization Trap."
3. **SyMerge (Jung et al., ICML 2026)**: Uses low-rank adapters to capture cross-task synergy in model merging. It focuses on adapter-based parameter updates rather than coefficient-level constraints.
4. **PolyMerge & SplineMerge (Ours)**: Rather than unconstrained optimization of $L$ parameters or introducing specialized adapter architectures, this work proposes a **hard structural constraint**. By parameterizing the layer coefficient trajectory as a continuous low-degree polynomial of normalized depth, it prunes high-frequency optimization degrees of freedom, effectively regularizing test-time adaptation.

## 2. Technical Novelty of PolyMerge & SplineMerge
While parameterizing model properties (like weights or learning rates) using continuous functions or splines is a standard mathematical technique in other contexts (e.g., continuous-depth parameterization in Neural ODEs or weight-sharing in coordinate-based networks), its application to **test-time adaptive model merging** is highly novel and brings distinct theoretical and empirical insights:

### A. The Structural Subspace Regularization Paradigm
Traditionally, trajectory smoothness is enforced via soft penalties (e.g., Total Variation regularization). PolyMerge introduces a **hard structural constraint** through continuous subspace parameterization. This has significant conceptual novelty:
* **Analytical Noise Filtering**: Proposition 3.1 mathematically demonstrates that orthogonal projection onto the Vandermonde column space of the low-degree polynomial acts as a robust low-pass filter, reducing white noise by a factor of $\frac{d+1}{L}$ and completely rejecting alternating sign noise.
* **Bypassing the Online Calibration Dilemma**: In unsupervised online test-time adaptation (TTA), labeled validation data is unavailable, making continuous hyperparameter tuning (e.g., selecting the optimal TV weight $\beta$) impossible. PolyMerge converts this into a robust, discrete architectural choice (choosing a low degree like $d=2$), which generalizes universally without test-time tuning.
* **Degenerate Trap Prevention**: Unsupervised entropy minimization is prone to degenerate states where the network collapses to a constant-class predictor. The paper shows that these degenerate states require discontinuous, localized weight perturbations (high roughness). Restricting coefficients to a low-degree polynomial subspace physically blocks the optimizer from accessing these overfit regions, acting as a structural shield.

### B. SplineMerge (Piecewise Subspace Adaptation)
SplineMerge represents a highly novel combination of spline theory and test-time weight fusion. Real-world networks exhibit structural block transitions (e.g., early features, middle representations, late classification projections). A global polynomial suffers from "smoothness bias" (underfitting these sudden block transitions). SplineMerge solves this by partitioning the network into structural groups and optimizing low-dimensional piecewise parameters. On actual CLIP foundation models, SplineMerge (Piecewise Constant) perfectly resolves the underfitting-roughness trade-off, matching the peak accuracy of unconstrained TTA (96.00%) while cutting roughness by 1.63x.

### C. Black-Box Optimization Synergies
The parameter efficiency of PolyMerge is uniquely advantageous for derivative-free/black-box optimization (like Evolution Strategies), where search complexity scales exponentially with parameter dimensionality. By reducing the search space from $L$ layers to $d+1$ dimensions, PolyMerge enables black-box optimizers to find superior local minima, outperforming TV-regularized ES ($84.91\%$ vs $84.45\%$, $p < 10^{-4}$). This represents a novel and practical contribution to black-box model merging.

## 3. Distinction from Prior Literature
The paper is exceptionally thorough in distinguishing itself from closely related literature:
* It explicitly notes that while prior works celebrate "layer-specificity," this specificity often functions as an optimizer-induced illusion of transductive overfitting.
* It compares directly with Total Variation (TV) and $L_2$ regularization, showing both empirical and mathematical superiorities (e.g., paired t-tests under coupled non-convex landscapes showing $p < 0.05$).
* It positions itself as a democratic, hardware-free alternative by releasing its high-fidelity landscape simulator, making it highly reproducible.
* The addition of the physical validation experiments on real MLP models and actual pre-trained CLIP foundation models completely separates this work from early stylized versions, showing practical utility in real weight-space dynamics.
