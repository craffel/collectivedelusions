# Peer Review of "PolyMerge: Continuous Polynomial Subspaces for Parameter-Efficient Adaptive Weight Fusing"

## 1. Summary of the Paper
This paper investigates a critical, previously unaddressed limitation of adaptive model merging via test-time adaptation (TTA), which the authors term the **Overfitting-Optimizer Paradox**. In adaptive model merging (exemplified by AdaMerging), merging coefficients are optimized on-the-fly at test-time on small, unlabeled target streams using unsupervised surrogate objectives like Shannon entropy minimization. 

To systematically analyze this phenomenon, the authors present a **Controlled Simulation and Optimization Landscape Study**. They mathematically model how unconstrained, high-dimensional layer-wise optimization using first-order gradient descent (such as Adam) easily exploits high-frequency spatial degrees of freedom to fit transductive noise in the local adaptation batch, resulting in highly jagged coefficient profiles and catastrophic generalization collapse on held-out test distributions.

To resolve this paradox, the authors introduce **PolyMerge** and its piecewise-continuous spline extension **SplineMerge**:
- **PolyMerge** parameterizes the entire layer-wise coefficient profile as a continuous, low-degree polynomial of normalized layer depth:
  $$\lambda_{k, l}(\boldsymbol{\alpha}) = \sum_{j=0}^d \alpha_{k, j} \cdot \left( \frac{l}{L-1} \right)^j$$
  This hard-constrains the optimization search space to a smooth, low-dimensional polynomial subspace, reducing learnable parameters from $L$ layers to just $d+1$ coefficients and mathematically filtering out high-frequency transductive noise.
- **SplineMerge** partitions the layers into structural block groups (e.g., early, middle, late layers) and parameterizes local low-degree polynomials or piecewise-constant segments within each block, preserving local block transitions and capturing layer heterogeneity.

The authors evaluate their methods across extensive, 30-seed simulative sweeps (over 700 fully optimized trajectories) and conduct dual physical validations: an end-to-end differentiable optimization of a 12-layer PyTorch Residual MLP, and a physical zero-shot test-time adaptation of pre-trained CLIP Vision Transformers (\texttt{openai/clip-vit-base-patch32}) using real test images and prompts.

---

## 2. Strengths and Weaknesses

### Major Strengths:
1. **Compelling and Paradigm-Challenging Conceptual Insights**: The paper’s most outstanding contribution is its profound conceptual diagnostic of adaptive model merging. It demonstrates that the complex, jagged "layer-specificity" commonly celebrated in prior adaptive merging literature is often an **optimizer-induced illusion** (transductive overfitting artifacts rather than functional necessity). The finding that post-hoc spatial averaging (Mean Treatment) or uniform scaling ($d=0$) can rescue or even outperform unconstrained layer-wise optimization strongly challenges how the community thinks about the necessity of fine-grained layer-wise coefficients.
2. **Structural vs. Penalty-Term Regularization**: The authors make a highly persuasive argument for *structural* subspace constraints over *penalty-term* regularizers (like Total Variation or $L_2$ regularization) in TTA environments. Because labeled validation streams are completely unavailable online during test-time adaptation, continuous hyperparameter tuning (such as searching for the optimal penalty weight $\beta$) is impossible. A discrete structural constraint (like polynomial degree $d$) is a robust, discrete decision that generalizes universally, requiring no online validation.
3. **Exceptional Empirical Rigor and Reproducibility**: The authors run all simulation configurations across 30 independent seeds, execute paired t-tests to establish high statistical significance, and provide comprehensive tables. More importantly, they are completely transparent, explicitly labeling simulative vs. physical results and explaining physical hardware constraints.
4. **Dual Simulative and Physical Validations**: Rather than relying solely on a customized simulator, the authors validate their theoretical findings inside functional deep learning models: first on a PyTorch Residual MLP, and then on pre-trained CLIP foundation model weights using real images. This dual validation bridges the simulative-to-physical gap and proves that the overfitting paradox is a genuine physical phenomenon.
5. **Excellent Mathematical and Curvature Analyses**: The appendices provide mathematically rigorous proofs of the low-pass filter noise reduction, a formal Hessian-curvature flatness analysis showing why PolyMerge converges to flatter and more generalizable basins, and a dynamic programming formulation for boundary partitioning.

### Major Weaknesses:
1. **Mathematical Simplicity and Incremental Algorithmic Progression**: While the conceptual insight regarding the "illusion of layer specificity" is paradigm-challenging, the proposed mathematical solution (polynomial and spline curve parameterization) is highly straightforward. Parameterizing neural network properties (weights, activations, or hyperparameters) as a function of normalized depth is a well-established technique in deep learning. The paper does not introduce a fundamentally new mathematical paradigm; rather, it applies classic curve parameterizations to a model-merging setup.
2. **Underfitting-Roughness Trade-off in Global Polynomials**: The physical CLIP foundation model experiments (Table 4) reveal that global PolyMerge ($d=2$ and $d=4$) suffers from a notable **underfitting bottleneck**, dropping multi-task average accuracy on real CLIP weights to 89.00% and 90.00% respectively, which is below the static Task Arithmetic baseline of 94.00%. This indicates that global polynomial constraints are too rigid to capture the complex, non-monotonic layer-wise sensitivity transitions in physical foundation models, making global polynomials practically sub-optimal on real-world weights.
3. **Transductive Overfitting of the Automated Boundary Discoverer**: The authors implement a dynamic programming (DP) recurrence to automatically discover entropy-minimizing block partitions for SplineMerge. However, their own empirical results (Table 3) show that DP-discovered partitioning actually yields *lower* generalization accuracy than simple manual uniform block partitions (86.12% vs. 86.80%). Shifting boundaries at test-time on unlabeled target streams introduces another axis of transductive overfitting, which compromises the automated method and makes the manual uniform partition heuristic practically superior.
4. **Grand Branding of Known TTA Overfitting**: The paper frames standard test-time adaptation overfitting under a dramatic title ("The Overfitting-Optimizer Paradox"). While this makes the paper highly engaging, the fact that minimizing an unsupervised surrogate objective (like prediction entropy) on a tiny, transductive unlabeled stream leads to representation collapse and overfitting is a heavily documented phenomenon in the TTA literature.

---

## 3. Detailed Dimension Ratings

### Soundness: Excellent
The paper is technically flawless and exceptionally well-validated. The mathematical formulations are rigorous, the proofs are sound, and the dual physical validations (PyTorch MLP and pre-trained CLIP) conclusively confirm the theoretical assertions. The authors' high degree of scientific transparency (explicitly labeling simulative metrics and discussing limitations/assumptions) is exemplary and establishes excellent scientific integrity.

### Presentation: Excellent
The presentation is outstanding. The paper is clearly written, beautifully structured, and exceptionally easy to follow. The figures (especially Figure 1 showing the jagged vs. smooth coefficient profiles, and Figure 2 mapping the bias-variance curve) are highly informative and of publication quality. The appendices are exceptionally thorough and add massive mathematical weight to the main submission.

### Significance: Good
The paper addresses a highly important and relevant problem in the foundation model era: training-free, unsupervised multi-task model merging on the fly. The proposed subspace approach offers practical benefits, particularly in terms of parameter efficiency for derivative-free/black-box optimization algorithms (like Evolution Strategies), which are highly valuable for resource-constrained edge deployments where gradients are unavailable. However, the underfitting trade-off of global polynomials and the overfitting of automated partitioning slightly limit the ultimate impact of the automated variants on large-scale models, making manual SplineMerge the default practical choice.

### Originality: Good
The originality of the paper is highly positive, driven primarily by its **profound diagnostic insights**. Shifting the focus of the adaptive merging community from "how do we get more fine-grained, independent coefficients?" to "how do we constrain the optimization search space to prevent transductive overfitting?" is a highly valuable, original conceptual contribution. However, the rating is capped at "Good" rather than "Excellent" because the mathematical tools used to implement this constraint (polynomial and spline curve fitting) are standard and represent an incremental mathematical progression.

---

## 4. Overall Recommendation

**Overall Recommendation: 5 (Accept)**

### Justification of the Recommendation:
This is a technically solid, exceptionally well-validated, and beautifully written paper that provides a highly valuable and thought-provoking conceptual contribution to the field of adaptive model merging. 

The paper’s core strength lies in its **bold conceptual diagnosis**: it mathematically exposes the widely celebrated "layer-specificity" of adaptive weight merging as an optimizer-induced illusion driven by transductive overfitting. By proving that restricting the merging coefficient search space to continuous, low-dimensional subspaces (such as quadratic polynomials or piecewise-constant blocks) completely stabilizes adaptation, prevents degenerate constant-class predictors, and achieves superior downstream generalization, the paper challenges how the community thinks about adaptive weight fusing.

Although the specific polynomial parameterizations are mathematically straightforward, the conceptual insights, the robust arguments for structural over penalty-based TTA regularization, the massive statistical sweeps across 30 random seeds, and the dual physical validations on physical PyTorch neural networks and pre-trained CLIP foundation models are highly impressive. The SplineMerge piecewise parameterization perfectly resolves the underfitting-roughness trade-off of global polynomials, matching the peak performance of unconstrained TTA (96.00% average accuracy on CLIP) while providing substantial spatial smoothing and high parameter efficiency (reducing optimized parameters from 12 to 3).

The limitations of global polynomial underfitting, the overfitting of automated DP partitioning, and the simulative nature of the primary sweeps are minor compared to the paper's overall scientific rigor, transparency, and conceptual depth. This work will undoubtedly influence future research in training-free multi-task adaptation, and is a strong candidate for acceptance.
