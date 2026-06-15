# Paper Summary: Limits of Representational Isotropy on Curved Manifolds

## 1. Goal and Core Objective
The paper presents a deep theoretical and empirical investigation into the feasibility, mathematical limits, and physical boundaries of translating representational isotropy techniques (like Euclidean spectral balancing) to curved non-linear manifolds. Specifically, it diagnoses model merging on the Riemannian manifold of the orthogonal group $\mathrm{O}(d)$ and its associated Lie algebra tangent space $\mathfrak{so}(d)$.

## 2. Core Methodology (RIMO)
The proposed framework, **RIMO** (**R**iemannian **I**sometry-respecting **M**anifold **O**perations), maps task-specific fine-tuned expert models $\{W_k\}_{k=0}^{N-1}$ and a pre-trained base model $W_0$ into a unified weight matrix $W_{\text{final}}$ through a five-phase pipeline:
1. **Orthogonal Procrustes Decoupling:** Decoupling unconstrained weights into an orthogonal component $R_k \in \mathrm{O}(d)$ and a linear residual component $\rho_k$ via SVD:
   $$W_k = W_0 R_k + \rho_k$$
2. **Manifold Tangent Projection:** Projecting orthogonal rotations $R_k$ to the skew-symmetric Lie algebra tangent space $\mathfrak{so}(d)$ using the inverse Cayley transform:
   $$Q_k = (R_k - I_d)(R_k + I_d)^{-1}$$
3. **Magnitude-Corrected Tangent Aggregation:** Performing a magnitude-corrected linear combination of Lie algebra components to mitigate destructive interference:
   $$Q_{\text{com}} = c \cdot \left(\frac{1}{N} \sum_{k=0}^{N-1} Q_k\right)$$
4. **Isotropic Spectral Balancing (or Rank-Preserving Pruning):**
   * **RIMO (Balancing):** Performing SVD on $Q_{\text{com}}$, interpolating singular values towards the global mean $\bar{\sigma}$ using hyperparameter $t \ge 1.0$, reconstructing, and projecting back to $\mathfrak{so}(d)$ via $\mathcal{P}(M) = \frac{1}{2}(M - M^T)$.
   * **RIMO-Pruned (Pruning/Mitigation):** Performing SVD on $Q_{\text{com}}$, keeping only the top keep ratio of singular value pairs, and setting the rest to exactly zero to preserve the low-rank structure of tangent updates.
5. **Manifold Retraction & Residual Reconstruction:** Mapping the modified tangent generator back to the Lie group via the forward Cayley transform and combining with averaged residuals:
   $$W_{\text{final}} = W_0 R_{\text{merged}} + \rho_{\text{merged}}$$

## 3. Key Findings & Theoretical Contributions
* **The Orthogonality Condition:** Riemannian model merging requires native manifold-respecting models (like OFT). Unconstrained models yield high-norm residuals that cause severe representation drift (merged accuracy is only $42.07\%$). Soft orthogonal regularization ($\lambda_{\text{ortho}}=2.0$) keeps weights near the manifold, reducing residuals and boosting merged accuracy to $84.55\%$. Naive post-hoc SVD projection of an unconstrained base model collapses performance ($15.00\%$) because it distorts the coordinate system.
* **The Tangent Space Spectral Pitfall:** Isotropic spectral balancing in the Lie algebra tangent space (RIMO with $t > 1.0$) catastrophically scrambles representations (collapsing accuracy to $13.66\%$ on MLP and $18.44\%$ on ViT).
* **The Kernel Distortion Theorem:** Mathematically proves that standard numerical SVD solvers introduce non-symmetric coordinate gauge matrices in multi-dimensional null spaces, and subsequent skew-symmetric projections required to restore Lie algebra closure inevitably inject destructive skew-symmetric noise.
* **The Spectrum Distortion Theorem:** Mathematically proves that non-uniform singular value modifications (like isotropic balancing) violate the fundamental skew-symmetric compatibility condition $R \Sigma = -\Sigma R^T$. Thus, skew-symmetric projection inevitably distorts the active spectrum.
* **Symmetry-Preserving Alternatives:** Proposes real Schur decomposition and a GPU-accelerated complex Hermitian solver (using complex eigen-decomposition scaled by $j$, achieving $12.2\times$ speedup over Schur and $8.1\times$ over SVD) to maintain perfect mathematical symmetry without post-hoc projections.

## 4. Empirical Evaluation & Quantitative Results
The paper evaluates the methods on **Split-MNIST** using a 3-layer MLP and a Vision Transformer (ViT) architecture, as well as a 5-expert multi-task scaling analysis.
* **RIMO (Balancing)** fails completely across all architectures ($13.66\%$ average accuracy on MLP, $18.44\%$ on ViT).
* **RIMO-Pruned** successfully mitigates the pitfall, achieving **$90.47\%$** in standard training and **$91.49\%$** in orthogonally regularized training, outperforming standard OrthoMerge ($84.55\%$) and matching state-of-the-art Euclidean baselines (TA, SAIM, TIES, DARE).
* **AdaMerging** suffers from severe task-specific overfitting in disjoint multi-task setups (collapsing to $0.00\%$ accuracy on inactive tasks).
* **Hard Orthogonal Constraints:** A pilot experiment with projected SGD on the Stiefel manifold boosts OrthoMerge to **$72.08\%$** average accuracy, proving that eliminating residuals completely decouples coordinate warp from manifold interpolation.
