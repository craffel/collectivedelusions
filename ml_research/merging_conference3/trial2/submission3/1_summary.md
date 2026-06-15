# Summary: PolyMerge (Polynomial Spline Parameterization of Layer-Wise Merging Strengths)

## 1. Paper Overview & Problem Statement
This paper investigates adaptive model merging, specifically test-time adaptation (TTA) of merging coefficients ($\lambda_{k,l}$) in multi-task expert model fusion (e.g., AdaMerging). In standard TTA, merging coefficients are optimized on unlabeled target data streams by minimizing unsupervised Shannon entropy. The authors identify and address a major limitation of this paradigm, termed the **Overfitting-Optimizer Paradox**:
* **The Paradox**: Unconstrained first-order gradient descent (like Adam) easily minimizes the unsupervised entropy loss on small adaptation batches by exploiting layer-wise (and block-wise) degrees of freedom. This leads to highly jagged, oscillating, and non-physical coefficient trajectories (high Total Variation / Roughness). Rather than capturing a physical hierarchy of layer importance, the optimizer fits transductive noise, leading to **catastrophic generalization collapse** on held-out test distributions (e.g., SVHN accuracy dropping from 73.24% to 63.16%).
* **The Degenerate Entropy Minimization Trap**: Under long optimization runs, unconstrained optimizers can exploit overparameterization to disrupt intermediate representation layers, turning the network into a constant-class predictor. This yields an unsupervised entropy of exactly zero but results in a complete accuracy collapse (0% accuracy).

## 2. Core Technical Contributions
To resolve this paradox, the paper introduces two key methodologies that restrict the optimization search space to a low-frequency, smooth continuous subspace:

### A. PolyMerge (Global Polynomial Subspace)
Instead of optimizing $L$ independent layer coefficients per task, PolyMerge parameterizes the entire coefficient profile as a continuous, low-degree polynomial of the normalized layer depth:
$$\lambda_{k, l}(\boldsymbol{\alpha}) = \sum_{j=0}^d \alpha_{k, j} \cdot \left( \frac{l}{L-1} \right)^j$$
* **Normalized Depth Scale**: Layer indices are normalized to $\bar{l} = \frac{l}{L-1} \in [0,1]$. This bounds basis functions to prevent numerical overflow, provides architectural scale-invariance, and improves the conditioning of the Vandermonde system.
* **Dimensionality Reduction & Noise Filtering**: By optimizing only $d+1$ parameters (where $d \in \{1, 2, 3\}$) instead of $L$ parameters (e.g., $L=12$ layers or $L=52$ projections), PolyMerge prunes high-frequency optimization noise (analytically proven in Proposition 3.1) and physically blocks the optimizer from accessing degenerate, discontinuous weight configurations.

### B. SplineMerge (Piecewise Splines)
To handle layer heterogeneity across deep networks (where structural blocks like attention and MLP have sudden step-wise sensitivity transitions), SplineMerge partitions the layers into $B$ disjoint block groups (e.g., Early, Mid, Late) and parameterizes each group with a localized low-degree polynomial or piecewise constant:
$$\lambda_{k, l}(\boldsymbol{\alpha}) = \sum_{j=0}^d \alpha_{k, b, j} \cdot \left( \frac{l - l_{\text{start}}^{(b)}}{l_{\text{end}}^{(b)} - l_{\text{start}}^{(b)}} \right)^j, \quad \forall l \in \mathcal{B}_b$$
This piece-wise formulation provides local flexibility to capture sharp transitions at structural block boundaries while strictly limiting parameter dimensionality and maintaining low-pass filtering.

## 3. Multi-Tier Evaluation Suite
The paper validates its claims through a highly comprehensive, multi-tier empirical framework:
1. **Controlled Simulation Landscape Study**: Evaluates optimization on a stylized convex sandbox (Model I) and a physically grounded, coupled non-convex stress-test (Model II with a Mahalanobis sensitivity metric and non-convex Rastrigin loss landscape) across 4 benchmarks (MNIST, FashionMNIST, CIFAR-10, SVHN) and 30 independent seeds.
2. **Physical MLP Validation**: Implements an end-to-end differentiable PyTorch pipeline with a 12-layer deep Residual MLP (`DeepResMLP`) across 10 random seeds. Gradients are backpropagated directly from predictions through GELU activations and residual skip-connections to coefficients.
3. **Physical Foundation Model Validation (CLIP)**: Deploys the framework on real pre-trained CLIP vision encoders (`openai/clip-vit-base-patch32` with CIFAR-10 and GTSRB experts) using real images and a genuine zero-shot cosine similarity TTA pipeline.

## 4. Key Empirical Findings
* **Overfitting Paradox Confirmed**: Unconstrained Adam TTA successfully minimizes local stream entropy but collapses generalization and explodes coefficient roughness (Total Variation) across all environments.
* **PolyMerge Generalization Peak**: Under simulation, PolyMerge ($d=2$, Adam) matches or beats heavily regularized baselines (86.57% average accuracy) while reducing seed variance and parameters. Under Model II (non-convex coupled landscape), PolyMerge ($d=2$) statistically significantly outperforms TV regularization ($p < 0.05$).
* **Derivative-Free Parameter Efficiency**: For black-box zero-order optimizers (1+1 ES) which scale poorly with dimension, PolyMerge's parameter reduction enables a massive advantage: PolyMerge ($d=2$, ES) achieves 85.35% average accuracy, outperforming TV-regularized ES ($84.45\%$, $p < 10^{-4}$).
* **SplineMerge Breakthrough**: On actual CLIP transformers, global polynomials suffer from an underfitting bottleneck due to strict smoothness (accuracy falls to 89.00%), whereas SplineMerge (Piecewise Constant) perfectly resolves this trade-off—achieving a peak average accuracy of **96.00%** (matching unconstrained TTA) while reducing coefficient roughness by **1.63x** with only 3 parameters.
