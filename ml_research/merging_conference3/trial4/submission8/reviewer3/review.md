# Peer Review

## 1. Summary of the Paper
This paper presents a mathematically rigorous and highly principled investigation into the post-training quantization (PTQ) robustness of test-time adaptive model merging (TTA). The authors identify and characterize a critical, previously overlooked vulnerability in adaptive model merging, which they term **Quantization-Operator Overfitting**: optimizing layer-wise merging coefficients on small test-time calibration streams (unsupervised entropy minimization) drives the parameters into extremely sharp, fragile local minima in the continuous weight space. While performing well in FP32, these sharp minima collapse under downstream post-training quantization rounding noise.

To resolve this, the paper proposes **CR-PolySACM** (Clipping-Regularized Sharpness-Aware Subspace Model Merging), a unified framework combining global structural constraints with local landscape flatness optimization:
- **Global Polynomial Subspace (PolyMerge):** Restricts the high-dimensional layer-wise blending coefficients to a low-degree polynomial of network depth, compressing the search space from $L \times K = 56$ parameters down to exactly $3 \times K = 12$ polynomial parameters. This acts as a robust global regularizer, preventing overfitting to local calibration statistics.
- **Local Landscape Flatness (CR-SACM):** Explicitly minimizes local loss sharpness. Crucially, the authors discover a fundamental physical **task-vector norm scale pathology** where a 50-fold discrepancy in layer-wise task-vector norms renders standard sharpness-aware optimization blind to sensitive, low-norm layers (such as the final layer norm), leading to optimization collapse. They introduce **Clipping-Regularized Sharpness-Aware Minimization (CR-SACM)**, which clips task-vector norms to a robust minimum floor $\beta = 0.10$ to restore scale-invariant optimizer sensitivity without triggering singular gradient explosion.

Evaluating across six diverse hardware-relevant quantization schemas (including INT8 symmetric/asymmetric and INT4 symmetric), the proposed framework consistently stabilizes model composition and sets a new state of the art.

---

## 2. Main Strengths

### A. Rigorous and Grounded Theoretical Framework
Unlike many heavily empirical papers in the model merging literature, this work is exceptionally well-grounded mathematically. 
- The authors utilize a second-order Taylor expansion to derive a complete, closed-form decomposition of the quantization-induced loss gap $\Delta \mathcal{L}$ directly in the low-dimensional polynomial parameter space (Eq. 12):
  $$\Delta \mathcal{L} \approx \nabla_W \mathcal{L}^T \delta_{\perp} + \frac{1}{2} \boldsymbol{\epsilon}^T \mathcal{H}_{\mathbf{p}} \boldsymbol{\epsilon} + \frac{1}{2} \delta_{\perp}^T \mathcal{H}_W \delta_{\perp}$$
- This decomposition formalizes the critical insight that test-time adaptation can only optimize the *in-subspace* second-order error ($\frac{1}{2} \boldsymbol{\epsilon}^T \mathcal{H}_{\mathbf{p}} \boldsymbol{\epsilon}$), whereas the *out-of-subspace* noise ($\delta_{\perp}$) is completely uncontrolled and dominates under low-precision targets (such as 4-bit). This explains why unregularized TTA collapses under INT4 and highlights the critical importance of a global structural subspace (like PolyMerge) to shield the model weights.

### B. Mathematical Characterization of scale Pathology
The paper makes a highly significant contribution by identifying the **task-vector norm scale pathology** in weight-space sharpness optimization. 
- The authors demonstrate that unnormalized coefficient perturbations scale weight-space perturbations by $(V_k^l)^2$. In a standard Vision Transformer, the final layer norm task vector norm is extremely small ($0.014$) compared to intermediate block layers ($0.60$).
- This $1800\times$ discrepancy renders unconstrained optimizers blind to the highly sensitive final layer norm.
- The proposed scale-invariant clipping regularization (CR-SACM) elegantly solves this pathology: scaling the perturbation inversely by $(V_{\text{clipped}, k}^l)^2$ achieves a uniform, stable weight perturbation, while the clipping threshold $\beta = 0.10$ prevents division-by-zero or singular gradient explosion.

### C. Strong Empirical Validation & Baseline Coverage
The empirical evaluation is highly comprehensive:
- The authors compare against an extensive set of baselines, including static Task Arithmetic, unregularized AdaMerging, spatial Total Variation regularization (RegCalMerge), quantization-aware merging (Q-Merge), PolyMerge, and unconstrained HessMerge.
- Evaluation spans 6 diverse target schemas, providing a highly realistic view of downstream hardware deployment.
- CR-PolySACM sets a new state of the art in the aggressive INT4 Symmetric per-Channel schema (19.07% vs PolyMerge's 18.10%), and our corrected HessMerge (Ours) baseline consistently outperforms unregularized AdaMerging (+1.36% in FP32, +1.25% in INT8), validating the effectiveness of sharpness-aware TTA once scale-blindness is corrected.

### D. Exemplary Intellectual Integrity and Honesty
The authors are highly commendable for their scientific honesty:
- They are completely transparent about the absolute INT4 accuracy (19.07%), noting that while it is a statistically significant relative improvement, it remains practically unusable for production systems, framing it correctly as a scientific proof-of-concept.
- They highlight the substantial expert-to-merge gap ($-31.27\%$) in disparate multi-domain settings, acknowledging representation interference as an open research challenge. This level of transparency increases the credibility and scientific value of the work.

---

## 3. Main Weaknesses and Constructive Areas for Improvement

While this is an outstanding paper of high technical quality, we identify several minor theoretical and methodological areas that can be improved:

### A. Generalization Properties of Alternative Subspaces
In Appendix A.3, the authors discuss alternative low-dimensional subspaces (Random Projections and Fourier DCT-based subspaces) and report continuous FP32 accuracies. To make this analysis theoretically complete, it would be highly valuable to formalize the metric-space and projection properties of these alternative manifolds. Specifically, how do Random Projections and DCT subspaces affect task-vector inner products, representation hierarchies, and the Lipschitz constants of the network under projection? A formal mathematical comparison would solidify the authors' claim that depth-dependent polynomials represent the optimal structural manifold.

### B. Theoretical Boundary on Implicit Entropy Regularization
In Section 3.5 and Appendix A, the authors explain that the unsupervised entropy minimization loss $\mathcal{L}_{\text{entropy}}$ naturally prevents coefficient scale inflation and activation drift, stabilizing the average coefficient sum at $1.42 \pm 0.04$ (far below the maximum limit of $4.0$). To make this claim mathematically complete, the authors should attempt to derive a formal mathematical upper bound on the parameter scale (or the sum of coefficients $\sum_k \lambda_k^l$) as a function of the prediction entropy. Formalizing this implicit regularization mechanism would strengthen the paper's theoretical framework.

### C. Formalization of the Percentile-Based Dynamic Clipping Blueprint
The authors propose a highly promising percentile-based dynamic clipping blueprint in Section 5 and Appendix A.1 to scale CR-SACM to deeper architectures (such as LLMs). However, they only evaluate it empirically as a proof-of-concept. A more formal mathematical analysis of how the shape of the task-vector norm distribution (e.g., highly skewed or heavy-tailed distributions in LLMs) dictates the optimal percentile threshold (e.g., 10th percentile) would provide a stronger foundation for this future direction.

---

## 4. Questions for the Authors

1. **Gauss-Newton Approximation:** Can the authors clarify if they measured the magnitude of the neglected Hessian term $\sum_i \nabla_W \mathcal{L}_i \nabla^2_{\mathbf{p}} W_{\text{merged}, i}(\mathbf{p})$ during test-time adaptation? While the coefficients remain in the active interior, having an empirical upper bound on this second-order weight derivative term would further solidify the Gauss-Newton assumption.
2. **Implicit Regularization Bound:** Is it possible to mathematically bound the activation drift or parameter scale inflation as a function of prediction entropy? Any preliminary derivation or proof sketch would be highly appreciated.
3. **Alternative Manifolds:** Have the authors considered evaluating other physical representation-preserving manifolds, such as rational functions (e.g., Padé approximants) or low-rank Tucker/CP decompositions of the weight space?

---

## 5. Detailed Ratings

- **Soundness: Excellent**
  The mathematical derivations are exceptionally rigorous, correct, and based on highly reasonable theoretical assumptions. Potential risks (such as boundary saturation, scale inflation, and transductive generalization gaps) are thoroughly analyzed and dismissed with both proof and empirical evidence.
- **Presentation: Excellent**
  The paper is beautifully written, logically structured, and exceptionally easy to follow. Formulas are precise, and figures are highly polished and informative.
- **Significance: Excellent**
  The paper addresses a critical, real-world bottleneck (PTQ of merged models), discovers a new physical pathology (scale pathology), and provides an elegant, scalable, and mathematically sound solution that sets a new state of the art.
- **Originality: Excellent**
  Combining global polynomial subspaces with clipping-regularized sharpness-aware optimization is a highly novel, creative, and powerful synthesis that represents a major advance over standard heuristics.

---

## 6. Overall Recommendation

**Rating: 6 (Strong Accept)**

**Justification:**
This is an outstanding, technically flawless paper that combines rigorous mathematical theory with comprehensive empirical validation to solve a critical problem in model merging. By decomposing the weight-space quantization loss gap and identifying the task-vector norm scale pathology, the authors provide a deep, fundamental understanding of landscape curvature and optimization stability in adaptive model composition. The proposed Clipping-Regularized SACM (CR-SACM) is elegant, mathematically sound, and achieves exceptional performance gains across six diverse quantization schemas while running 52.8$\times$ faster than exact Hessian trace optimization. Backed by extensive ablation studies, outstanding scientific honesty, and highly thorough robustness evaluations (e.g., under extreme calibration class imbalance), this paper represents a significant, highly influential milestone for both model merging and neural network compression. I strongly recommend accepting this work.
