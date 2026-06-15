# Mock Review

## Reviewer Recommendation
* **Overall Recommendation:** 5: Accept *(Technically solid paper, with high impact on at least one sub-area of AI or moderate-to-high impact on more than one area of AI, with good-to-excellent evaluation, resources, reproducibility, and no unaddressed ethical considerations.)*
* **Soundness:** Excellent
* **Presentation:** Excellent
* **Significance:** Excellent
* **Originality:** Excellent

---

## Paper Summary
This paper presents a rigorous theoretical and empirical investigation into the **limits of representational isotropy on curved manifolds**, specifically within the orthogonal group $\mathrm{O}(d)$ and its associated Lie algebra tangent space $\mathfrak{so}(d)$. Under a model-merging context, the authors evaluate the feasibility of translating Euclidean-based spectral alignment methods (like SAIM) to geometric manifolds. 

Through their investigation, the authors uncover and formalize two critical phenomena:
1. **The Orthogonality Condition:** Riemannian model merging (e.g., OrthoMerge) is highly sensitive to non-orthogonality. Incorporating soft orthogonal regularization during training keeping parameters near the manifold reduces Euclidean residuals and boosts merged accuracy from $42.07\%$ to $84.55\%$. Conversely, attempting to bypass orthogonal training via a post-hoc SVD projection of an unconstrained model is functionally destructive, collapsing merged performance to $15.00\%$.
2. **The Tangent Space Spectral Pitfall:** Performing SVD-based isotropic spectral balancing inside the Lie algebra tangent space (by interpolating singular values towards their mean, as in RIMO) catastrophically scrambles representations, collapsing MLP accuracy to $13.66\%$ and Vision Transformer (ViT) accuracy to $18.44\%$. The authors derive the **Kernel Distortion Theorem (Theorem 3.2)** and **Spectrum Distortion Theorem (Theorem 3.3)** to prove that standard SVD solvers introduce non-symmetric coordinate gauges in the multi-dimensional null space, and that the subsequent projection step required to restore skew-symmetry inevitably distorts the active spectrum. Under the non-linear forward Cayley map, inflating smaller singular values maps to massive, spurious high-dimensional rotations across inactive dimensions.

To bypass this pitfall, the authors propose **RIMO-Pruned**, which performs **rank-preserving spectral pruning** instead of isotropic smoothing, keeping inactive dimensions at exactly zero singular value. This achieves a robust merged accuracy of $91.49\%$ on MLP and $88.16\%$ on Vision Transformers. Furthermore, they prove mathematically and empirically that flat-space Euclidean averaging suffers from $O(1/\sqrt{N})$ representational magnitude collapse as the number of tasks $N$ scales up, whereas geometric manifold-level merging preserves representation energy exactly ($\|R_{merged}\|_F = \sqrt{d}$) for any $N$.

---

## Strengths
1. **High Mathematical Rigor:** The paper is exceptionally well-supported by formal propositions and theorems. The proofs for Proposition 3.1, Theorem 3.2, and Theorem 3.3 are mathematically elegant, complete, and correct. The mathematical proof of $O(1/\sqrt{N})$ representational magnitude decay in Euclidean Task Arithmetic is extremely clean and provides a solid foundational reason for geometric merging.
2. **Valuable "Negative" Result Investigation:** Rather than hiding the catastrophic failure of SVD-based spectral balancing in Lie algebra tangent spaces, the authors perform a deep diagnostic study of this failure. They successfully identify its mathematical root causes (kernel gauge asymmetry in standard numerical solvers and violation of the compatibility relation $R\Sigma = - \Sigma R^T$ under projection), which is of immense value to the geometric deep learning community.
3. **Rigorous Empirical Validation:** The authors back up their claims with rich experiments across multiple settings, including a 3-layer MLP and a Vision Transformer (ViT). They sweeps over various hyperparameters ($t$, block size $b$), and provide a multi-seed statistical significance check, ensuring that their findings are highly stable and robust.
4. **Addressing Complexity and Scalability:** To address the cubic complexity $O(d^3)$ of SVD, the authors provide a block size sensitivity analysis (finding $b=128$ as the optimal sweet spot for localized subspace rotations and sequential loop overhead) and run a parallel batched PyTorch SVD latency benchmark. This establishes block-diagonal SVD as a highly parallelizable and hardware-efficient tool for modern LLM layer dimensions ($d = 4096$).
5. **Excellent Presentation and Clarity:** The paper is extremely well-structured, clear, and engaging. The transitions between the theoretical derivations and empirical experiments are flawless, and the tables and heatmaps are clear and high-signal.

---

## Weaknesses & Areas for Improvement
While the paper is technically solid and highly polished, the following minor areas could be addressed or expanded upon:

1. **Large-Scale Real-World NLP Tasks:** The empirical evaluations are conducted on Split-MNIST (using MLP and ViT architectures). Although the authors explicitly address generalizability by providing a formal mathematical proof of quadratic noise scaling with dimension $d$ and parallel SVD benchmarks at modern LLM scale ($d = 4096$), evaluating on full LLMs (such as LLaMA or Mistral fine-tuned with LoRA on standard NLP datasets like GLUE or GSM8K) would make the empirical section even more undeniable.
2. **Empirical Validation of Hard Orthogonal Constraints:** In Section 4.8, the authors note a performance gap of $9.45\%$ between OrthoMerge ($84.55\%$) and Task Arithmetic ($94.00\%$) under orthogonal training, and attribute this to the soft nature of the training constraint ($\lambda_{ortho} = 2.0$) which generates small residuals. While they provide an excellent theoretical discussion and mathematical formulation for hard orthogonal training (Riemannian SGD and geotorch parameters), actually implementing one of these hard-constrained models and showing it closes the gap would enrich the empirical section.
3. **Comparison with Test-Time Adaptive Methods:** SOTA Euclidean model merging includes test-time adaptive methods like AdaMerging and SyMerge. Although the authors cite them, they do not include them in the main quantitative tables. Including a baseline like AdaMerging on the regularized experts could enrich the benchmark.

---

## Questions and Constructive Suggestions for the Authors
1. **LoRA Integration:** Task-specific updates in LLMs are widely parameterized as low-rank LoRA matrices. Can RIMO and RIMO-Pruned be applied directly to LoRA weights? Since LoRA weights are rectangular and low-rank, how would the block-diagonal partitioning and the inverse Cayley transform handle them?
2. **Schur Decomposition Implementation:** In Section 3.4.3, you propose Real Schur Decomposition as a symmetry-preserving spectral decomposition that completely eliminates the need for post-hoc projection. Have you implemented this? Does it resolve the spectral balancing pitfall in practice, and how does its sequential/parallel latency compare with SVD?
3. **Hard Constraints vs. Soft Constraints:** If you run fine-tuning using a hard-constrained orthogonal optimizer (e.g., Riemannian SGD on the Stiefel manifold), does the performance gap between manifold-level merging and Task Arithmetic disappear? Providing a quick pilot experiment on this would be highly appreciated.

---

## Conclusion
This is a top-tier paper that makes outstanding theoretical and empirical contributions to the field of geometric deep learning and model merging. It is exceptionally well-written, mathematically rigorous, and thorough. I strongly recommend its acceptance.
