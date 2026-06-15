# Peer Review Report

## 1. Summary of the Paper
This paper presents a rigorous theoretical and empirical investigation into the geometric and spectral boundaries of model merging on non-linear curved manifolds. While dominant model-merging techniques (e.g., Task Arithmetic, TIES-Merging, DARE, SAIM) operate in flat Euclidean spaces, they risk representation drift and catastrophic task interference. To preserve representation stability, parameter updates can be constrained to the orthogonal group $\mathrm{O}(d)$ and interpolated inside the tangent Lie algebra $\mathfrak{so}(d)$ via Cayley transforms (as in OrthoMerge). 

The authors evaluate the feasibility of translating Euclidean-based representational isotropy techniques (specifically, SVD-based spectral balancing) to geometric manifolds. They introduce **RIMO** (Riemannian Isometry-respecting Manifold Operations) as a diagnostic environment. Their investigation uncovers two critical phenomena:
1. **The Orthogonality Condition:** Manifold-level merging is highly sensitive to non-orthogonality. Soft orthogonal training keeps parameters close to $\mathrm{O}(d)$ and reduces Euclidean residuals, boosting accuracy from $42.07\%$ to $84.55\%$. Conversely, attempting to bypass orthogonal training via a post-hoc SVD projection onto $\mathrm{O}(d)$ is destructive, collapsing performance to $15.00\%$ and proving the necessity of native manifold-respecting models.
2. **The Tangent Space Spectral Pitfall:** SVD-based spectral balancing in the Lie algebra (RIMO with $t > 1.0$) catastrophically collapses performance to $13.66\%$ (on MLP) and $18.44\%$ (on ViTs). The authors formalize this via two key theorems:
   * **Kernel Distortion Theorem:** Standard numerical SVD solvers introduce non-symmetric coordinate gauges in multi-dimensional null spaces, injecting destructive skew-symmetric noise during projection.
   * **Spectrum Distortion Theorem:** Non-uniform spectral modifications violate the algebraic relation $R \hat{\Sigma} = -\hat{\Sigma} R^T$, causing the subsequent projection operator to distort the active spectrum.
   * Under the non-linear Cayley map, inflating smaller singular values maps to massive, high-dimensional spurious rotations across inactive planes, scrambling representation features.

To resolve these boundaries, the authors propose **RIMO-Pruned** (rank-preserving spectral pruning that zeroes out small singular values, avoiding kernel distortion), which successfully recovers robust accuracies ($91.49\%$ on MLP, $88.16\%$ on ViT). They also introduce **Real Schur Decomposition** as a projection-free, symmetry-preserving alternative, and implement a GPU-parallelizable **Complex Hermitian Solver** (using `torch.linalg.eigh` on $i Q$) that executes in just **7.66 ms** ($12.2\times$ faster than Schur and $8.1\times$ faster than SVD).

---

## 2. Strengths and Weaknesses

### Soundness
* **Strengths:** 
  * The mathematical and theoretical soundness of this paper is exceptional. The derivation of the Kernel Distortion Theorem and Spectrum Distortion Theorem is rigorous, elegant, and provides a clear, mathematically complete explanation for the spectral balancing failure observed in tangent spaces.
  * The systems-level contributions are highly sound and innovative. Using real Schur decomposition to block-diagonalize skew-symmetric matrices into $2\times2$ skew-symmetric blocks directly preserves the Lie algebra structure without requiring a post-hoc projection step. Mapping $Q$ to a complex Hermitian matrix ($i Q$) to enable batched, GPU-parallel complex Hermitian eigen-decomposition is an outstanding and mathematically elegant solution to the $O(d^3)$ sequential complexity bottleneck.
  * The empirical validation is thorough, honest, and multi-faceted. The authors include multi-seed robustness runs, block-diagonal sensitivity sweeps, and scaling analyses showing that flat Euclidean averaging suffers from an $O(1/\sqrt{N})$ representational magnitude decay while manifold merging preserves energy at $O(1)$.
* **Weaknesses:**
  * While the authors honestly discuss the performance gap between manifold merging (e.g., OrthoMerge and RIMO) and flat Euclidean averaging (Task Arithmetic), this gap remains a significant practical limitation of the work. Simple Task Arithmetic consistently outperforms geometric merging (e.g., $94.00\%$ vs $84.55\%$ under soft orthogonal training, and $94.00\%$ vs $72.08\%$ under hard orthogonal constraints). This limit must be presented more prominently in the main text as a primary limitation.
  * The block-diagonal partitioning technique used for rectangular weights is a practical necessity, but it restricts the orthogonal degrees of freedom to localized compartments. This trade-off (lack of global, layer-wide rotation correlations) is not discussed in depth.

### Presentation
* **Strengths:** 
  * The paper is beautifully written, highly structured, and easy to follow.
  * Figure 1 (pipeline flowchart) and Algorithm 1 are outstanding aids that clearly explain the exact execution steps of the proposed pipelines.
  * The authors are highly transparent and detailed in their mathematical notations and experimental details.
* **Weaknesses:**
  * The primary limitation of the work—the performance gap between geometric merging and Task Arithmetic—is largely confined to Section 4 and the appendix. Placing a clear discussion of this gap in a dedicated limitations section in the main text would improve the paper's transparency.

### Significance
* **Strengths:** 
  * This work sets a clear, mathematically proven boundary for the model-merging and representation-learning communities, warning researchers against naively applying Euclidean techniques to non-linear spaces.
  * The proposed mitigations (RIMO-Pruned and the parallel Complex Hermitian Solver) are highly scalable and represent solid systems-level contributions.
  * The scaling analysis ($O(1)$ energy conservation of manifold merging) establishes a powerful, fundamental justification for pursuing geometric model merging in large-scale multi-task serving.
* **Weaknesses:**
  * The primary experiments are evaluated on Split-MNIST and Split-CIFAR-10 with low-complexity models (3-layer MLPs, small custom ViT). Although the mathematical derivations generalize to larger networks, testing the framework on modern production-scale PEFT/LoRA adapters (e.g., LLaMA-7B where $d=4096$) fine-tuned on complex benchmarks (e.g., GLUE or vision-language tasks) would dramatically increase the practical significance of this paper for the broader ML community.

### Originality & Literature Positioning
* **Strengths:** 
  * The paper is highly original in its diagnostic nature. Rather than just presenting another empirical "recipe," it analyzes a fundamental geometric boundary.
  * The application of real Schur and complex Hermitian solvers to Lie-algebraic model merging is highly novel and elegant.
* **Weaknesses:**
  * While the paper is scholarly and cites many classical matrix analysis texts, there are several key areas where the literature positioning and historical context are missing or could be strengthened to provide proper attribution of ideas:
    1. **Pre-OFT Orthogonal Neural Networks:** The paper attributes orthogonal parameterization heavily to Orthogonal Fine-Tuning (OFT; Qiu et al., 2023) and OrthoMerge (Yang et al., 2026). However, parameterizing and optimizing orthogonal weights using Lie algebras ($\mathfrak{so}(d)$) and Cayley transforms has a rich history in deep learning, particularly for recurrent neural networks and Stiefel manifold optimization. The authors should cite pioneering works such as **Lezcano-Casado & Martínez-Rubio (2019)** (*"Cheap Orthogonal Constraints..."*), **Helfrich et al. (2018)** (*"Orthogonal Recurrent Neural Networks..."*), and **Wisdom et al. (2016)** (*"Full-Capacity Unitary Recurrent Neural Networks"*).
    2. **Representational Isotropy & Anisotropy:** The authors cite SAIM (2026), but do not ground the concept of representational isotropy in the extensive literature on the "cone effect" and representation degeneracy in transformers and NLP. Citing foundational works like **Ethayarajh (2019)** (*"How Contextual are Contextualized Word Representations?..."*), **Gao et al. (2019)** (*"Representation Degeneracy Problem..."*), and **Mu & Viswanath (2018)** (*"All-but-the-top..."*) would place the motivation for representational isotropy on a stronger scholarly foundation.
    3. **Procrustes Alignment in NLP:** The Orthogonal Procrustes problem has a long history of use in representation learning for cross-lingual word embedding alignment (e.g., **Conneau et al., 2017**, *"Word Translation Without Parallel Data"*; **Smith et al., 2017**, *"Offline bilingual word embeddings..."*). Connecting their decoupling method to this mature line of alignment research would enrich the paper.
    4. **Fréchet/Karcher Mean Connection:** The paper performs direct averaging of Lie algebra generators: $Q_{\text{avg}} = \frac{1}{N} \sum Q_k$. In Riemannian geometry, the standard definition of the mean of points on a manifold is the **Fréchet (or Karcher) mean**, which minimizes the sum of squared geodesic distances on the manifold. Direct averaging in the tangent space at a single reference point (the identity) is a **first-order, single-step approximation** to the true Fréchet mean. Pointing out this distinction would add exceptional scholarly depth.

---

## 3. Ratings

* **Soundness:** **Excellent**
* **Presentation:** **Excellent**
* **Significance:** **Good**
* **Originality:** **Excellent**

---

## 4. Overall Recommendation
**5: Accept**

**Justification:** This is an exceptionally high-quality, mathematically rigorous, and original paper. It moves beyond typical empirical trial-and-error model-merging literature to perform a deep, diagnostic study of representation geometry. The derivation of the Kernel and Spectrum Distortion Theorems is mathematically elegant and correct, and the systems contributions (the projection-free Real Schur and the GPU-parallel Complex Hermitian solvers) are outstanding. Although there is a persistent performance gap with flat Task Arithmetic and the main experiments rely on lower-complexity datasets, the paper's theoretical robustness, honest analysis of limitations, and systems-level contributions make it a highly valuable addition to the conference.

---

## 5. Detailed Comments, Questions, and Suggestions for the Authors

### 1. Missing Historical Context and Proper Attribution
To elevate the scholarly quality of the paper, please integrate the following citations and discussions into your manuscript:
* **Learning with Orthogonal Constraints:** In Section 2, acknowledge that using Lie algebras and Cayley transforms to optimize orthogonal/unitary parameters has a long history in deep learning. Please cite:
  * *Lezcano-Casado, M., & Martínez-Rubio, S. (2019). Cheap Orthogonal Constraints in Neural Networks on the Stiefel Manifold. ICML.*
  * *Helfrich, K., Ye, D., & Zhou, G. (2018). Orthogonal Recurrent Neural Networks with Weyl Eleanor Cayley Transform. ICML.*
  * *Wisdom, S., Powers, T., Hershey, J., Roux, J. L., & Atlas, L. (2016). Full-Capacity Unitary Recurrent Neural Networks. NeurIPS.*
* **Representational Isotropy Literature:** In Section 2 (Spectral Analysis paragraph), connect the motivation for restoring isotropy to the foundational "cone effect" and anisotropy literature in NLP. Please cite:
  * *Ethayarajh, K. (2019). How Contextual are Contextualized Word Representations? Comparing Geometry with Co-occurrence. EMNLP.*
  * *Gao, J., He, D., Tan, X., Qin, T., Wang, L., & Liu, T. Y. (2019). Representation Degeneracy Problem in Training Natural Language Generation Models. ICLR.*
  * *Mu, J., & Viswanath, P. (2018). All-but-the-top: Simple and effective postprocessing for word representations. ICLR.*
* **Procrustes NLP History:** In Section 3.1, connect the Orthogonal Procrustes decoupling to its historical use in aligning word embedding spaces. Please cite:
  * *Conneau, A., Lample, G., Ranzato, M., Burger, L., & Jégou, H. (2017). Word Translation Without Parallel Data. ICLR.*
  * *Smith, S. L., Turban, D. H., Hamblin, S., & Hammerla, N. Y. (2017). Offline bilingual word embeddings. ICLR.*

### 2. Tangent Space Averaging vs. The Fréchet Mean
In Section 3.3, clarify that direct arithmetic averaging of the skew-symmetric generators $Q_k$ in the tangent space at the identity is a **first-order, single-step approximation** to the true Riemannian **Fréchet (or Karcher) mean**. The true Fréchet mean is defined as the point that minimizes the sum of squared geodesic distances on the manifold, and must be solved iteratively (mapping to the tangent space, averaging, and retracting back in a loop). Explicitly noting that your method is a computationally efficient, bi-invariant first-order proxy for the true Fréchet mean will add a high level of geometric precision.

### 3. Absolute Performance Gap and Limitations Section
Even under hard orthogonal constraints (where residuals are zero), OrthoMerge achieves $72.08\%$ average accuracy, lagging significantly behind simple Euclidean Task Arithmetic ($94.00\%$). Under soft regularization, RIMO-Pruned reduces this gap but still lags by $2.51\%$.
* **Suggestion:** Please add a dedicated **Limitations** section to the main text (e.g., at the end of Section 5 or in Section 4) that prominently discusses this performance gap. Frame this gap as a key open challenge for geometric merging. Discuss how navigating Stiefel manifolds under non-convex loss landscapes creates significant path divergence and loss barriers, making standard flat-space optimizers (like Adam) perform poorly and calling for the development of geodesic-aware optimization techniques.

### 4. Coordinate Constraint Trade-offs in Block-Diagonal Partitioning
For rectangular weights, you partition the matrices into square blocks of size $b \times b$. While this is algebraically clean and preserves orthogonality block-wise, it restricts the orthogonal degrees of freedom to localized compartments, ignoring cross-block coordinate correlations.
* **Question/Suggestion:** Can you elaborate on the representational capacity trade-off of block-diagonal partitioning? Adding a brief paragraph discussing how smaller block sizes ($b$) reduce sequential SVD/Schur latency but limit global rotational expressiveness would be highly instructive.

### 5. Scale and Dataset Generalization
While Appendix C & I show that the spectral pitfall theoretically worsens quadratically with representation size $d$, your main empirical results rely on Split-MNIST with small MLPs.
* **Suggestion:** If possible, include a small-scale empirical validation on pre-trained foundation models (such as RoBERTa-large or CLIP-ViT-B/16) fine-tuned on a standard benchmark (like GLUE or VTAB) using LoRA. Merging LoRA adapters (where $d=4096, k=8$) on the manifold using RIMO-Pruned would make this paper a massive, high-impact landmark for the model-merging community.
