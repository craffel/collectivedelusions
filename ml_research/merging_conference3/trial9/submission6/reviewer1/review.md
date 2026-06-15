# Peer Review for Conference Submission

## Summary

This paper addresses the challenge of merging or ensembling task-specific projection operators (such as those used in Low-Rank Adaptation, or LoRA) in deep neural networks. The authors observe that linear interpolation of projection matrices (as in traditional parameter- or activation-level ensembling) violates the manifold geometry of orthogonal projections, leading to "eigenvalue shrinkage" and "projected coordinate collapse." 

To resolve this issue, the paper proposes **Continuous Riemannian-Geometric Homotopical Model Merging via Grassmannian Geodesic Blending (C-Lie-MM)**. This framework treats task-specific projection bases as points on the Grassmannian manifold $\mathcal{G}(d, D)$ and blends them along exact geodesics:
1. It computes a static coordinate reference point $Y_0 \in \mathcal{G}(d, D)$ offline as the projection-metric Karcher mean of the expert bases.
2. It maps the experts onto the tangent space of $Y_0$ using a custom Cut-Locus-Aware logarithm map.
3. It performs a sample-wise weighted linear combination of tangent matrices online.
4. It maps the blended tangent matrix back onto the Grassmannian via the exponential map to guarantee that the merged operator remains symmetric, idempotent, and of rank $d$.
5. To bypass the online SVD computation bottleneck during serving, it derives a square-root-free polynomial approximation using low-order Chebyshev polynomials.

The method is evaluated in a 14-layer Analytical Coordinate Sandbox and a Simulated GLUE LoRA Benchmark, demonstrating that it successfully prevents coordinate collapse and outperforms flat ensembling methods (like SABLE) in environments lacking standard architectural buffers.

---

## Strengths

1. **Mathematical Rigor**: The paper is mathematically solid and elegant. The proofs of eigenvalue shrinkage, manifold preservation, differentiability, and the metric distortion bounds on the tangent space are rigorous and complete.
2. **Exemplary Transparency and Scientific Honesty**: The authors are exceptionally candid about their method's assumptions, limitations, and potential gaps. The discussion in Section 4.3 on the "Coordinate Collapse" strawman and how residual connections and LayerNorm act as geometric buffers in real-world architectures is incredibly refreshing and commendable.
3. **Thorough Evaluation**: The paper evaluates the proposed method against 11 baselines in the sandbox and 4 standard merging baselines in the GLUE simulation, providing a comprehensive experimental picture.
4. **Systems-Level Implementation**: The authors go beyond theory by writing a 333-line custom Triton GPU kernel (`triton_kernel.py`) and deriving Chebyshev approximations, showing a strong effort to make the online serving path computationally viable.

---

## Weaknesses

1. **Extreme Over-Engineering and Disproportionate Complexity**: 
   While the mathematical formulation is elegant, the sheer volume of high-level mathematics and systems-level machinery (Grassmannian manifolds, tangent space mapping, Karcher means, SVD sign-alignment tracking, Tikhonov regularization, Chebyshev polynomial approximations, custom Triton GPU kernels) is disproportionate to the actual problem being solved. 
   Complexity in machine learning should only be introduced when absolutely necessary and justified by massive gains. In this paper, the empirical performance gap between the highly complex C-Lie-MM and a simple flat model with slightly tuned parameters is virtually non-existent under realistic, non-orthogonal overlapping manifolds ($70.30\%$ vs. $70.00\%$, which is well within the standard deviation). It is difficult to justify introducing a custom SVD-polynomial Triton serve path to standard neural network pipelines for a marginal $+0.30\%$ gain in accuracy.

2. **The "Coordinate Collapse" Premise is a Strawman**: 
   The core motivation of the paper—that flat linear blending causes a catastrophic exponential representation decay to zero and collapses classification accuracy—is a worst-case theoretical result that only occurs in highly artificial, purely feedforward stacks. 
   As the authors' own self-critical ablation study in Section 4.3 demonstrates, actual neural networks possess standard components (residual connections, LayerNorm, and non-linearities) that act as powerful geometric buffers, completely preventing catastrophic coordinate collapse. When these standard components are present, uniform merging achieves $51.90\%$ and SABLE achieves $64.60\%$ (up from $25.00\%$), while C-Lie-MM achieves $72.30\%$. While a $+7.70\%$ gain is visible, it is far more modest than the catastrophic "collapse to zero" portrayed in the introduction and abstract.

3. **Biased "Simulated" GLUE LoRA Evaluation**: 
   The Simulated GLUE LoRA Benchmark propagates representations through 8 sequential projection layers *without* standard residual connections or LayerNorm. This is a highly artificial and biased setup. Real RoBERTa-Large networks have residual connections and LayerNorm at every single layer. By deliberately stripping these protective components from their simulation, the authors induce an artificial coordinate collapse in the flat baselines (collapsing them to random guessing of $55.0\%$), making their own method appear to deliver an inflated $+42.0\%$ improvement. In a physical RoBERTa-Large model with standard residual paths, the flat methods would not collapse, and the performance gap would be much smaller.

4. **Reference Point Instability and Non-Uniqueness**: 
   Under the boundary condition of perfectly orthogonal tasks, the spectral gap of $P_{\text{avg}}$ collapses to zero, and the Karcher mean reference point $Y_0$ is no longer unique. Although the authors state this does not affect gradient-tracking because $Y_0$ is detached from the gradient graph, it means that minor floating-point perturbations can lead to wildly different coordinate frames. This compromises the stability and consistency of the local tangent space approximation across different machines or backends.

5. **Circular Engineering Design Loop**: 
   The authors introduce high mathematical complexity (non-Euclidean manifold ensembling requiring online SVDs) and then must introduce further engineering complexity (6th-order Chebyshev polynomial expansions and a custom Triton GPU kernel) just to bypass the computational bottleneck of their original choice. A simpler, more direct approach (such as basic gating or temperature tuning in flat Euclidean space) completely avoids the need for SVDs, SVD sign trackers, and polynomial evaluations in the first place.

---

## Detailed Ratings

### Soundness: Good
The theoretical derivations and proofs are correct and mathematically elegant. However, the soundness of the empirical claims is slightly compromised by the artificial and biased simulation setups that exclude standard network components (residuals and LayerNorm) to inflate the baseline collapse.

### Presentation: Excellent
The paper is exceptionally well-written, logically structured, and easy to follow. The figures and tables are informative and highly polished. The authors' honesty and self-critical analysis of their own limitations set an outstanding scientific standard.

### Significance: Fair
The practical significance of this work is low. Deep learning practitioners overwhelmingly favor simple, robust, and easily maintainable methods over complex, delicate non-Euclidean manifold operations, especially when simpler methods (like temperature tuning or gating) achieve virtually identical joint accuracy ($70.00\%$ vs $70.30\%$) without any manifold overhead.

### Originality: Good
The application of Grassmannian geometry and exact Riemannian geodesics to task-specific representation-space projection merging is original and interesting. However, the individual tools (Grassmannian logging/exp maps, Karcher means, Chebyshev approximations) are established techniques, and their combination here represents an incremental, albeit highly formalized, development.

---

## Overall Recommendation

**Rating: 3 (Weak Reject)**

**Justification:**
This is an incredibly well-written, mathematically rigorous, and scientifically honest paper with clear merits. However, the proposed C-Lie-MM framework represents a severe case of over-engineering. It introduces a massive amount of mathematical and systems-level complexity (Grassmannian geometry, offline Karcher mean centroids, logarithm/exponential tangent mappings, SVD sign-alignment tracking protocols, Tikhonov regularization, Chebyshev polynomial approximations, and custom Triton GPU kernels) to solve a problem that is largely mitigated by standard, simpler architectural components (residual connections and LayerNorm) or a simple temperature-tuned flat baseline.

The main experiments are artificially designed (lacking standard residuals and LayerNorm) to force flat baselines to collapse catastrophically, while realistic comparisons under severe task overlap show that simple flat gating performs virtually identically to the highly complex C-Lie-MM ($70.00\%$ vs. $70.30\%$). Because the immense complexity is not justified by substantial, realistic performance gains, the weaknesses outweigh the merits. I encourage the authors to explore simpler, more elegant, and direct ways to align representation spaces that do not require such heavy geometric machinery.
