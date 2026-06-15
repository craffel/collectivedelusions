# 2. Novelty Check & Literature Positioning

## Characterization of Novelty
The novelty of this submission is **significant** and **conceptual** rather than purely incremental. Instead of merely proposing another empirical recipe for model merging, the paper conducts a deep, diagnostic exploration of a geometric boundary—specifically, why standard Euclidean-based representational isotropy techniques (such as SVD-based spectral balancing) are fundamentally incompatible with non-linear curved manifolds (specifically the orthogonal group $\mathrm{O}(d)$ and its Lie algebra $\mathfrak{so}(d)$). The derivation of the **Kernel Distortion Theorem** and **Spectrum Distortion Theorem** represents a substantial theoretical advancement that mathematically explains the catastrophic collapse observed in practice.

## Delta from Prior Work
1. **From OrthoMerge (Yang et al., 2026):** OrthoMerge introduced the concept of geometric model merging on the orthogonal manifold by solving the Orthogonal Procrustes problem and interpolating task updates inside the tangent Lie algebra $\mathfrak{so}(d)$. However, OrthoMerge does not investigate the spectral properties of these updates, nor does it address representational isotropy. This submission builds directly on the OrthoMerge framework but subjects it to a rigorous spectral analysis, demonstrating that naive attempts to balance rotation magnitudes in the tangent space are destructive.
2. **From SAIM (Sharpness-Aware Isotropic Merging, 2026):** SAIM showed that performing SVD-based spectral balancing on task vectors in Euclidean space reduces interference and enhances multi-task performance. This paper is the first to attempt to translate representational isotropy to curved manifolds, demonstrating that operations that are safe and linear-safe in Euclidean spaces become highly destructive in Lie algebra tangent spaces due to non-linear curvature mapping.
3. **From Standard Parameter-Space Merging (Task Arithmetic, TIES, DARE):** These methods average or prune weights in flat Euclidean spaces, ignoring non-linear neural geometry. This work provides a rigorous alternative by mapping updates to tangent spaces, identifying the limitations of soft orthogonal regularization, and demonstrating that rank-preserving spectral pruning (**RIMO-Pruned**) outperforms standard OrthoMerge.

---

## Scholarly Literature Context & Missing Attributions
While the submission is highly rigorous and cites many recent and classical works (including foundational matrix analysis texts and optimization manifolds), there are several key areas where the paper's positioning and historical context can be strengthened:

### 1. Pre-OFT Orthogonal and Unitary Neural Networks
The authors frame the geometric constraint parameterization largely around Orthogonal Fine-Tuning (OFT; Qiu et al., 2023) and OrthoMerge (Yang et al., 2026). However, the use of Lie algebras ($\mathfrak{so}(d)$) and Cayley transforms to parameterize and optimize orthogonal/unitary weights has a long and rich history in deep learning, particularly for recurrent neural networks and Stiefel manifold optimization. Key works that should be cited to ground this lineage include:
* **Lezcano-Casado & Martínez-Rubio (2019)**, *"Cheap Orthogonal Constraints in Neural Networks on the Stiefel Manifold"*, which pioneered the use of the exponential map and Cayley transform for optimization on the Stiefel manifold.
* **Helfrich et al. (2018)**, *"Orthogonal Recurrent Neural Networks with Weyl Eleanor Cayley Transform"*, which utilized the Cayley transform to map skew-symmetric matrices to orthogonal matrices.
* **Wisdom et al. (2016)**, *"Full-Capacity Unitary Recurrent Neural Networks"*, which first demonstrated the representational power of unitary and orthogonal constraints in neural networks.
Acknowledging this background prevents the false impression that Lie algebra mapping via Cayley transforms was newly conceived for OFT.

### 2. General Literature on Representational Isotropy and Anisotropy
The paper addresses representational isotropy, citing SAIM (2026), but fails to situate the motivation within the wider literature on representation learning and transformer geometry. Representational anisotropy (the "cone effect") is a well-documented phenomenon in contextualized representation spaces:
* **Ethayarajh (2019)**, *"How Contextual are Contextualized Word Representations? Comparing Geometry with Co-occurrence"*
* **Gao et al. (2019)**, *"Representation Degeneracy Problem in Training Natural Language Generation Models"*
* **Mu & Viswanath (2018)**, *"All-but-the-top: Simple and effective postprocessing for word representations"*
These works establish the fundamental premise of representational isotropy in deep networks and should be referenced to provide a stronger scholarly foundation for why restoring isotropy is desirable.

### 3. Historical Use of Procrustes Alignment in Deep Learning
The Orthogonal Procrustes problem (Gower, 1975) is utilized here to decouple standard weights. It has been extensively used in representation learning for cross-lingual word embedding translation and alignment:
* **Conneau et al. (2017)**, *"Word Translation Without Parallel Data"*
* **Smith et al. (2017)**, *"Offline bilingual word embeddings..."*
Referencing these works would connect the authors' decoupling method to a mature line of research in representation alignment.

### 4. Tangent Space Averaging vs. True Fréchet/Karcher Mean
In Phase 3, the authors average the tangent space matrices directly: $Q_{\text{avg}} = \frac{1}{N} \sum Q_k$. In differential geometry and Riemannian statistics, the standard generalization of the arithmetic mean to a manifold is the **Fréchet (or Karcher) mean**, which minimizes the sum of squared geodesic distances on the manifold:
$$\bar{R} = \arg\min_{R \in \mathrm{O}(d)} \sum_{k=0}^{N-1} d(R, R_k)^2$$
On Lie groups, because there is no general closed-form solution for $d > 2$, the Fréchet mean is typically solved iteratively via a gradient descent algorithm (mapping to tangent spaces, averaging, and taking the exponential map). Direct averaging in the tangent space at the identity is a **first-order, single-step approximation** of the true Fréchet mean. Discussing this distinction and clarifying that tangent-space averaging is a computationally efficient, bi-invariant first-order proxy for the true Fréchet mean would demonstrate a highly sophisticated understanding of the underlying differential geometry.
