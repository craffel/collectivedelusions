# Soundness & Methodology

## Clarity of the Description
The methodology and mathematical formulations in Section 3 are written with exceptional clarity and detail. The paper does an excellent job of breaking down:
- **Multi-Task Model Merging:** Expressing the dynamic layer-wise unquantized parameters as a continuous linear combination of the pre-trained weights and task vectors.
- **Quantization Operators:** Formulating asymmetric and symmetric operators with precise mathematical notation for scale ($s$), zero-point ($z$), and dequantization, as well as distinguishing tensor-wise and channel-wise granularity.
- **Gradient Tracking:** Highlighting the asymmetric gradient flow where scale factors propagate gradients while zero-points are non-differentiable and detached.
- **Optimization Backends:** Formulating the Straight-Through Estimator (STE) first-order update and the derivative-free 1+1 Evolution Strategy (1+1 ES) precisely.

The equations are complete, self-contained, and easy to follow.

---

## Appropriateness of Methods & Rigor
The choice of methods and evaluation axes is highly appropriate and rigorous for an empirical audit:
- **Unquantized AdaMerging Baseline:** This is a vital baseline that is often omitted in merging literature. By comparing direct quantization-aware optimization to full-precision optimization followed by post-hoc quantization, the paper successfully decouples the effects of weight-space search from quantization constraints.
- **Cross-Schema Evaluation Matrix:** Systematically mapping performance across 4 optimization schemas and 5 evaluation schemas is an elegant and rigorous way to demonstrate operator overfitting.
- **Supervised Calibration Baseline:** Formulating a supervised baseline on the $N$-sample calibration stream is methodologically sound as it isolates whether performance collapse under skewed or small streams is due to data scarcity or the unsupervised entropy objective itself.
- **Stochastic Search Comparison:** Introducing 1+1 ES as a derivative-free comparator is highly appropriate to test whether the gradient approximations (STE) are the primary source of optimization failure.

---

## Technical Flaws & Methodological Weaknesses (Theorist's Perspective)

While the empirical design is highly sound, a theory-oriented reviewer will identify several major technical and theoretical limitations in the methodology:

### 1. Lack of Formal Mathematical Proofs and Guarantees
The paper uses extensive mathematical notation to formalize standard concepts (like quantization and STE), but it does not provide any formal proofs, theorems, or mathematical guarantees.
- **No Operator Overfitting Proof:** The paper claims that continuous coefficients overfit to the exact rounding boundaries of $Q_{\text{opt}}$. However, there is no mathematical proof or generalization theory (e.g., in terms of Rademacher complexity, PAC-Bayes bounds, or Lipschitz continuity) showing why the learned coefficients must fail to generalize under a different operator $Q_{\text{eval}}$.
- **No Gradient Bias Analysis:** The authors attribute the failure of STE to "gradient noise" and "biased gradients," but they do not mathematically analyze or bound the bias of the straight-through estimator on the continuous coefficient search space.
- **Heuristic Surrogate Claim:** The claim that Gaussian noise evaluates the "expectation over the noise distribution" ($\mathbb{E}_{\eta} [\nabla \mathcal{L}]$) and acts as a "smooth surrogate" is a qualitative explanation borrowed from randomized smoothing. The paper does not provide a formal derivation of the smoothing effect specifically for the discontinuous, multi-task quantized weight-merging landscape.

### 2. Methodological Confounders in the Subspace-Constraint Analysis
The authors evaluate a "Subspace-Constrained (LoRA-like) Merging" using a global, post-hoc Singular Value Decomposition (SVD) projection to compress the expert task vectors to rank-4.
- **SVD as a Poor Proxy for PEFT:** As the authors themselves honestly note, post-hoc global SVD is highly destructive, destroying representation capacity and collapsing the matched accuracy to $13.00\%$. 
- **The "Low-Capacity Generalization Illusion" Confounder:** Because the model's capacity is severely degraded and its output probability distributions are highly diffused (approaching random noise), the apparent closing of the generalization gap is a degenerate artifact of information loss rather than active robustness. Using SVD as a proxy fails to validate whether actual natively-trained low-rank adapters (like LoRA, which maintain high performance) are robust to operator shifts. This represents an unresolved methodological gap.

### 3. Speculative Scaling Projections
The paper claims that scaling up parameter count (e.g., to Pythia or Llama-3) is expected to expand the Cross-Schema Generalization Gap due to "exponentially increasing independent discrete rounding thresholds" and "dense scaling factors."
- **Lack of Mathematical Formulation for Scaling:** This scaling projection is entirely qualitative. The paper does not provide a formal mathematical model of how the density of rounding thresholds scales with model parameters, nor does it present any empirical data on larger models. In deep learning, over-parameterization can sometimes lead to smoother loss landscapes and flatter minima, which could theoretically *mitigate* rather than *exacerbate* quantization sensitivity. Without formal proofs or empirical scaling curves, the claim that the gap expands with scale remains a speculative, unverified hypothesis.

### 4. No Formal Convergence Analysis for the "Hybrid Pipeline"
The "Hybrid Optimization Pipeline" proposed in Appendix B and formalized in Algorithm 1 combines STE and 1+1 ES. However, the authors do not present any convergence proofs, theoretical validation, or empirical evaluations of this proposed pipeline. It remains a schematic, untested algorithmic recommendation.

---

## Reproducibility
The reproducibility of the work is **excellent**:
- The paper details the exact model backbone used (`vit_tiny_patch16_224` from `timm`).
- It defines the optimization hyperparameters (learning rate $10^{-2}$, 100 steps for Adam; 50 generations and $\sigma^{(0)}=0.05$ for 1+1 ES).
- The exact datasets (MNIST, FashionMNIST, CIFAR-10, SVHN) and individual expert performances are documented.
- All mathematical definitions of the quantization operators are fully detailed, including formulas for scales and zero-points.
