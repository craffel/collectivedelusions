# Impact and Presentation Quality

## Major Strengths
1. **Rigorous Theoretical Grounding:**
   The paper stands out for its deep mathematical foundation. Decomposing high-dimensional weight-space quantization noise into an in-subspace projected perturbation and an orthogonal complement via the Jacobian (Eq. 5), and relating this to the multi-task loss gap via a second-order Taylor expansion (Eq. 12), is a highly elegant and principled contribution.
2. **Identification of a Physical scale Pathology:**
   Identifying the **task-vector norm scale pathology**—where a 50-fold discrepancy in layer-wise task-vector norms renders standard sharpness-aware optimization blind to sensitive layers like the final layer norm—is a major scientific contribution.
3. **Elegant and Stable Solution (CR-SACM):**
   CR-SACM mathematically resolves scale-blindness by scaling perturbations inversely by $(V_{\text{clipped}, k}^l)^2$, while using clipping ($\beta = 0.10$) to prevent division-by-zero and singular gradient explosion.
4. **Strong Empirical Results and Baseline Coverage:**
   CR-PolySACM consistently outperforms standard PolyMerge across all 6 target schemas, setting a new state of the art in INT4 (19.07% vs 18.10%). Furthermore, HessMerge consistently outperforms AdaMerging, proving that sharpness-aware adaptation is highly effective once scale-blindness is corrected.
5. **Outstanding Scientific Honesty:**
   The authors are exceptionally transparent about the absolute INT4 performance limitation and the expert-to-merge domain gap. This intellectual integrity increases the scientific value of the work.

---

## Constructive Areas for Improvement

While the paper is of high quality, we propose the following theoretical and methodological enhancements to further strengthen the work:

1. **Formalize Generalization in Alternative Subspaces:**
   In Appendix A.3, the authors discuss alternative low-dimensional subspaces (Random Projections and Fourier DCT-based subspaces). It would be highly valuable to formally analyze the metric-space properties of these subspaces. For instance, can we mathematically characterize the preservation of task-vector inner products or Lipschitz constants of the network under these different projection operators? Formalizing why the depth-dependent polynomial subspace is the optimal structural manifold would elevate the theoretical contribution.
2. **Derive a Formal Bound for Implicit Entropy Regularization:**
   In Section 3.5, the authors explain that the unsupervised entropy minimization loss $\mathcal{L}_{\text{entropy}}$ naturally acts as an implicit scale regularizer against parameter inflation and activation drift. To make this claim mathematically rigorous, the authors should attempt to derive a formal upper bound on the parameter scale (or the sum of coefficients $\sum_k \lambda_k^l$) as a function of the prediction entropy. This would formalize the implicit regularization mechanism.
3. **Develop the Percentile-Based Dynamic Clipping Blueprint:**
   The authors propose a percentile-based dynamic clipping blueprint in Section 5 and Appendix A.1 to scale CR-SACM to deeper architectures (such as LLMs). However, they only evaluate it as an empirical proof-of-concept. A more thorough mathematical analysis of how the shape of the task-vector norm distribution (e.g., highly skewed or heavy-tailed distributions in LLMs) dictates the optimal percentile threshold (e.g., 10th percentile) would provide a stronger foundation for this scaling strategy.

---

## Overall Presentation Quality
The presentation is **excellent**. The writing style is professional, direct, and mathematically precise. The structure is highly logical, transitioning seamlessly from the motivation of quantization-operator overfitting to the mathematical derivation of scale pathologies, followed by robust empirical verification.
- Figure 1 (performance vs PTQ formats) and Figure 2 (convergence curves) are exceptionally clean, highly informative, and immediately convey the core ideas.
- Tables are well-formatted, with clear captions and standard deviations.

---

## Potential Impact and Significance
This paper has **high potential impact**. 
Post-hoc model merging is a rapidly expanding field, and deploying merged models on resource-constrained edge hardware under quantization is a critical real-world bottleneck. 
By establishing the mathematical link between weight-space quantization rounding noise, local landscape sharpness, and scale-blindness, this paper sets a new standard for robust model composition. 
The insights and scale-balancing mechanisms introduced in this work are highly likely to influence future research on merging and compressing large-scale foundation models (such as LLMs and vision-language models) where quantization is an absolute necessity.
