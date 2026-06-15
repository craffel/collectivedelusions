# Peer Review

**Paper Title:** Limits of Representational Isotropy on Curved Manifolds

---

## 1. Summary of the Paper
This paper presents a rigorous theoretical and empirical investigation into the feasibility, mathematical limits, and boundaries of translating representational isotropy (spectral balancing) techniques from flat Euclidean spaces to curved non-linear manifolds. The authors focus on the orthogonal group $\mathrm{O}(d)$ and its associated Lie algebra tangent space $\mathfrak{so}(d)$ as a diagnostic environment. 

The paper identifies two critical phenomena in Riemannian model merging:
1. **The Orthogonality Condition:** Riemannian model merging requires native, manifold-respecting pre-training or fine-tuning (like OFT). Unconstrained standard models yield high-norm residuals under Orthogonal Procrustes decoupling, causing severe representation drift (merged accuracy of $42.07\%$). Soft orthogonal regularization ($\lambda_{\text{ortho}}=2.0$) keeps parameters close to the manifold, reducing residuals and boosting accuracy to $84.55\%$. Attempting to bypass orthogonal training via a post-hoc SVD projection of an unconstrained base model collapses merged accuracy to $15.00\%$.
2. **The Tangent Space Spectral Pitfall:** While Euclidean spectral balancing is linear-safe, performing it in the tangent space (RIMO with $t > 1.0$) catastrophically scrambles representations (collapsing accuracy to $13.66\%$ on MLP and $18.44\%$ on ViT). This collapse is formalized via two new theorems:
   * **The Kernel Distortion Theorem:** Proves that standard numerical SVD solvers introduce non-symmetric coordinate gauges in multi-dimensional null spaces, injecting destructive skew-symmetric noise under projection.
   * **The Spectrum Distortion Theorem:** Proves that non-uniform singular value modifications violate the fundamental compatibility relation $R \Sigma = -\Sigma R^T$, causing projection to distort the active spectrum.

To mitigate these limits, the authors propose **RIMO-Pruned** (Rank-Preserving Spectral Pruning), which maintains low-rank tangent updates and achieves robust merged performance ($91.49\%$ on MLP and $88.16\%$ on ViTs) without test-time calibration. Additionally, they propose real Schur decomposition and a GPU-accelerated complex Hermitian solver that runs up to $12.2\times$ faster than Schur and $8.1\times$ faster than SVD (only $7.66$ ms on an NVIDIA H100 GPU), resolving the primary scalability bottleneck of manifold-level merging.

---

## 2. Strengths and Weaknesses

### Strengths
* **Exceptional Theoretical Rigor:** The mathematical formalization of the "tangent space spectral balancing pitfall" via the Kernel Distortion Theorem and Spectrum Distortion Theorem is beautifully derived and extremely solid. It elevates the paper from a simple empirical discovery to a foundational theoretical contribution.
* **First-of-its-Kind Diagnostic Study:** It is the first paper to evaluate the feasibility of representational isotropy and spectral balancing on non-linear manifolds / Lie algebras, establishing a clear geometric boundary that operations safe in Euclidean spaces can become highly destructive on curved manifolds.
* **Elegant Symmetry-Preserving and GPU-Parallelizable Solutions:** The derivation and implementation of the real Schur decomposition and the GPU-compatible Complex Hermitian solver (running in just $7.66$ ms on an NVIDIA H100 GPU) are brilliant, highly practical solutions that completely resolve the computational bottleneck of geometric merging.
* **Comprehensive Empirical Sweeps & Ablations:** The empirical evaluation features extensive sweeps over task scaling ($N=5$ experts), block sizes ($b \in \{32, 64, 128, 256\}$), random seeds ($3$ initializations), and architectures (MLP and Vision Transformers), providing exceptional scientific validity.
* **Insights into SOTA Baselines:** Exposing the domain-specific overfitting of AdaMerging in disjoint setups is a highly valuable, high-signal negative result for the model merging community.

### Weaknesses
* **Toy-Scale Evaluation and Custom Toy Architectures:** A major limitation is that the empirical evaluation is highly restricted in scale and complexity:
  1. **Low-Complexity Datasets:** The primary evaluations are conducted on **Split-MNIST**, a toy dataset with $28 \times 28$ grayscale images. Even for the Split-CIFAR-10 evaluation, the codebase uses a simple 3-layer MLP and only retains a **20% subset of the dataset** to keep CPU training fast.
  2. **Custom "Toy" Vision Transformer:** The Vision Transformer (ViT) utilized in Section 4.5 is an extremely small, custom toy model with an embedding dimension of **32**, a depth of only **1**, and **2** attention heads.
  * *Impact:* It remains unproven whether the proposed RIMO-Pruned or the GPU Complex Hermitian solver scales or generalizes to modern, large-scale foundation models (e.g., LLaMA, RoBERTa, or standard ViT-B/16) on high-complexity, diverse tasks.
* **Persistent Performance Gap with Euclidean Task Arithmetic:** In the orthogonally regularized regime, simple flat Euclidean Task Arithmetic achieves $94.00\%$ average accuracy, whereas the proposed RIMO-Pruned only achieves $91.49\%$. While RIMO-Pruned is a huge improvement over standard OrthoMerge ($84.55\%$), it still underperforms the flat-space baseline by $2.51\%$, which reduces its practical appeal.
* **Underperformance and Optimization Difficulty of Hard Orthogonal Constraints:** In the pilot experiment using hard orthogonal constraints during training (projected SGD on the Stiefel manifold), individual experts achieve high task accuracies, but OrthoMerge on these experts collapses to $72.08\%$ average accuracy. This indicates that hard-constrained optimization on the Stiefel manifold severely restricts representational capacity or is extremely difficult to optimize, casting doubt on its viability as a future direction.
* **Numerical Singularity of the Inverse Cayley Transform:** The inverse Cayley transform is defined if and only if $\det(R + I_d) \ne 0$. The paper does not discuss how it handles cases where $R_k + I_d$ is near-singular (e.g., when a rotation is exactly $\pi$ radians in some plane), which could introduce numerical instability during training or merging.

---

## 3. Section-by-Section Evaluation

### Soundness
**Rating: Excellent**
The paper is exceptionally rigorous and mathematically correct. The proofs for Proposition 3.1, Theorem 3.2, and Theorem 3.3 are flawless and highly elegant. The empirical evaluation is extensive and rule out single-seed optimization anomalies.

### Presentation
**Rating: Excellent**
The writing is exceptionally clear and structured, and the overall narrative is highly compelling and easy to follow. Figure 1 (the TikZ flowchart of the RIMO pipeline) is outstandingly professional. The equations are presented with clear physical interpretations.

### Significance
**Rating: Good-to-Excellent**
Exposing the spectral balancing pitfall in Lie algebras prevents future researchers from pursuing flawed SVD-based manifold-balancing schemes. Additionally, the GPU-compatible Complex Hermitian solver is highly practical and makes geometric model merging computationally scalable for modern, large-scale architectures.

### Originality
**Rating: Excellent**
The paper is highly original, being the first to study representational isotropy on curved manifolds and Lie algebras, deriving two new theorems to formalize the failure, and introducing novel symmetry-preserving decompositions.

---

## 4. Overall Recommendation
**Recommendation: 4: Weak Accept**

*Justification:* The paper is a tour de force in terms of mathematical rigor and theoretical analysis. It provides an exceptionally complete and elegant explanation of a newly discovered geometric pitfall in model merging and offers high-performance engineering solutions (like the $7.66$ ms GPU Complex Hermitian solver) and practical mitigations (RIMO-Pruned). However, because the empirical evaluation is restricted to extremely small-scale toy architectures and the Split-MNIST benchmark, and because there is a persistent performance gap with flat-space Task Arithmetic, a "Weak Accept" is the most objective and realistic rating for a top-tier machine learning conference. If the authors can demonstrate similar performance gains on larger-scale benchmarks (such as GLUE or CIFAR-100), this paper would easily be a Strong Accept.

---

## 5. Detailed Comments and Constructive Suggestions

### Constructive Suggestion 1: Scale to Larger Benchmarks and Standard Architectures
To fully realize the impact of this work and convince practitioners, the authors should evaluate RIMO-Pruned on more complex benchmarks and standard architectures.
* *Actionable Step:* Run a subset of the experiments on CIFAR-100 or a small-scale NLP benchmark (such as GLUE with RoBERTa-base or LoRA-fine-tuned LLaMA-7B) to demonstrate that RIMO-Pruned's recovery of performance generalizes beyond tiny custom toy networks.

### Constructive Suggestion 2: Address Cayley's Singularity limits
The inverse Cayley transform $Q_k = (R_k - I_d)(R_k + I_d)^{-1}$ is mathematically undefined when $R_k$ has an eigenvalue of $-1$.
* *Actionable Step:* Add a small footnote or paragraph in Section 3.2 acknowledging this singularity. Discuss whether any numerical safeguards (e.g., adding a small perturbation $R_k + (1+\epsilon)I$) were used during the experiments, or acknowledge it as a limitation compared to the matrix exponential/logarithm map.

### Constructive Suggestion 3: Discuss the Optimization Difficulty of Hard Constraints
The pilot experiment on hard orthogonal constraints (projected SGD on Stiefel manifold) collapsed merged performance to $72.08\%$ average accuracy, despite individual experts having $>93\%$ accuracy.
* *Actionable Step:* Expand the discussion in Section 4.6 (and Appendix G) explaining why this performance drop occurs. Is optimization on the Stiefel manifold prone to bad local minima, or does the hard constraint restrict the representational capacity of the model? This will provide a more honest and comprehensive evaluation of hard manifold constraints.

### Constructive Suggestion 4: De-congest the Main Text
The main text of the paper is highly dense, packed with multiple theorems, proofs, decompositions, and empirical findings.
* *Actionable Step:* Consider moving the detailed block size sensitivity analysis (Section 4.8) or SOTA baseline details (Section 4.7) completely to the appendix, and use the saved space to expand on the intuitive physical meaning of the Spectrum Distortion Theorem in the main text.
