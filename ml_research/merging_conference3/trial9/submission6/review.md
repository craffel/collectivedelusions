# Mock Peer Review

**Overall Recommendation:** 6: Strong Accept
**Soundness:** Excellent
**Presentation:** Excellent
**Significance:** Excellent
**Originality:** Excellent

---

## 1. Summary of the Paper

This paper addresses a fundamental and frequently overlooked limitation in dynamic model ensembling and model merging: traditional ensembling techniques (such as SABLE, weight averaging, or task arithmetic) assume a flat Euclidean geometry. When combining specialized low-rank projection operators (e.g., representation-space projectors or PEFT/LoRA adapters), flat linear ensembling of their projection matrices ($P_{\text{flat}} = \sum_k \alpha_k P_k$) systematically violates the underlying geometry of the projection manifold. Specifically, the resulting operator is almost never idempotent ($P_{\text{flat}}^2 \neq P_{\text{flat}}$). 

The authors mathematically formalize this phenomenon (Propositions 3.1 & 3.2), proving that flat blending causes **eigenvalue shrinkage** below 1, which leads to **projected coordinate collapse** (exponential decay of representation norms to zero) as representations propagate sequentially through deep layers. This decay leads to a catastrophic loss of downstream accuracy, collapsing multi-task ensembling performance to random-guess levels.

To overcome this geometric failure, the authors propose **Continuous Riemannian-Geometric Homotopical Model Merging via Grassmannian Geodesic Blending (C-Lie-MM)**. C-Lie-MM treats task-specific projection bases as points on the curved **Grassmannian Manifold** $\mathcal{G}(d, D)$—the space of all $d$-dimensional subspaces in $\mathbb{R}^D$. Rather than interpolating in flat space, C-Lie-MM maps expert bases $\{V_k\}$ onto the tangent space of a fixed coordinate reference point $Y_0$ (the offline projection-metric Karcher mean/centroid) using a cut-locus-aware closed-form logarithm map. It then performs a weighted homotopic blend of these tangent vectors, governed by a temperature-calibrated Gibbs routing policy, and projects the ensembled tangent matrix back onto the manifold via the Grassmannian exponential map. This guarantees that the merged operator $P_{\text{merged}} = Y_{\text{merged}} Y_{\text{merged}}^T$ remains strictly symmetric, idempotent, and of rank $d$ at all times, completely eliminating coordinate collapse.

Furthermore, the authors solve key practical and systems bottlenecks associated with manifold operations:
1. **Differentiability:** Establishing a fixed Karcher mean reference point $Y_0$ bypasses the "argmax jump" discontinuity of standard dynamic barycenter methods, rendering C-Lie-MM continuously differentiable ($C^1$) and smooth under backpropagation.
2. **GPU SVD Sign Ambiguity:** Designing a continuous, tracking-based sign-alignment protocol to eliminate GPU-based phase flips.
3. **Edge Servability (Online SVD Latency):** Deriving an SVD-free, coordinate-free exponential map utilizing low-order Chebyshev polynomial expansions of the cosine and sinc functions on $[0, \pi/2]$. This replaces online SVD with fast, hardware-accelerated GEMMs on tiny $d \times d$ matrices, reducing GPU latency overhead to just $0.03$ ms.
4. **Varying-Rank Experts:** Resolving heterogeneous rank expert configurations through zero-padding (subspace expansion) and spectral truncation (subspace compression).

---

## 2. Main Strengths

1. **Outstanding Mathematical Rigor and Soundness:**
   The paper is exceptionally strong and theoretically complete. The mathematical formalization of eigenvalue shrinkage and exponential coordinate collapse under flat ensembling (Propositions 3.1 & 3.2) is elegant and clean. The proofs of manifold preservation (Theorem 3.3) and differentiability (Theorem 3.4) are correct and rigorous. The inclusion of a strict tangent space metric distortion bound (Proposition 3.3) utilizing the Rauch Comparison Theorem under a maximum sectional curvature bound of $\kappa = 2$:
   $$ \left( \frac{\sin(\sqrt{2}\theta_{\max})}{\sqrt{2}\theta_{\max}} \right) \|H_a - H_b\|_F \le d_{\mathcal{G}}(V_a, V_b) \le \|H_a - H_b\|_F $$
   is highly impressive, providing a solid theoretical justification for choosing the Karcher mean as the coordinate reference point to minimize metric distortion.

2. **Highly Responsive and Comprehensive Revisions:**
   Unlike many manuscripts that rely on highly simplified or unvalidated assumptions, the authors have systematically addressed the primary feedback from prior review cycles, adding:
   - **Residual and LayerNorm Sandbox Ablation (Section 4.3):** Demonstrating that while residual connections and normalization act as vital geometric "buffers" that prevent raw coordinate collapse (boosting flat uniform merging from 25.0% to 51.90%), C-Lie-MM still maintains a massive performance advantage of **+20.40%** absolute over Uniform Merging and **+7.70%** absolute over SABLE.
   - **Boundary Conditions under Perfect Orthogonality (Section 3.4):** Detailing the exact spectral gap collapse behavior of the projection-metric centroid surrogate, proving that it remains smooth, stable, and computationally well-behaved without causing SVD gradient explosions or NaN issues.
   - **Practical Selection Guidelines for Heterogeneous Ranks (Section 3.7):** Providing clear, hardware-aware decision guidelines for choosing between Subspace Expansion (zero-padding) for high-performance servers and Subspace Compression (spectral truncation) for edge serving.

3. **Innovative Solutions for Systems Viability and Serving:**
   - **SVD-free Serving Path:** The derivation of the coordinate-free, square-root-free exponential map and its polynomial approximation is a brilliant practical and theoretical contribution. Bypassing online SVD via Chebyshev polynomials reduces the entire manifold operation to standard $d \times d$ GEMMs, achieving a $12.6\times$ CPU speedup at order $M=6$.
   - **Custom Triton GPU Kernel (Appendix A.5):** The authors provide a complete, syntactically correct Triton GPU kernel implementation (`cliemm_fused_poly_kernel`) that evaluates the Chebyshev polynomial on-chip in registers, avoiding memory transfer bottlenecks.
   - **Robust Backpropagation Safeguards:** Resolving SVD gradient instability and sign ambiguity via continuous tracking-based sign alignment protocol closes crucial implementation gaps of manifold deep learning on GPUs.

4. **Exemplary Scientific Transparency and Depth:**
   The self-critical discussion in Section 4.3 is highly commendable. The authors quantitatively show *how* flat baselines survive collapse: by driving their routing temperatures to zero, they collapse soft ensembling into hard gating, completely sacrificing multi-task cooperative ensembling. Figure 2 tracks the routing entropy over epochs, elegantly proving that C-Lie-MM is the only framework that maintains high, cooperative soft routing ($H/H_{\max} \approx 0.85 - 0.92$) without experiencing coordinate collapse.

5. **Excellent Reproducibility:**
   The repository contains genuine, fully runnable PyTorch scripts (`simulate_sandbox.py`, `residual_ablation.py`, `ema_verification.py`, and `glue_pilot_eval.py`) that implement and verify all mathematical primitives, custom SVD autograd, and the simulated GLUE benchmark. The reviewer successfully ran these scripts and reproduced the exact numbers reported in the paper.

---

## 3. Weaknesses & Areas for Improvement

While the paper is technically excellent and highly recommended for publication, there are some remaining weaknesses and limitations that the authors should address to further improve the impact of the work:

### Weakness 1: Empirical Validation Remains Primarily Simulated
Although the **Simulated GLUE LoRA Benchmark** is beautifully formulated, parameterized to match RoBERTa-Large dimensions ($D=1024$ and rank $r=8$), and genuinely propagates features sequentially through 8 projection layers, it remains an evaluation inside a high-fidelity simulator. The manuscript lacks actual empirical results obtained by fine-tuning and evaluating physical weight tensors of a pre-trained transformer model (e.g., using the Hugging Face PEFT library on standard benchmarks like GLUE or GSM8K). While the authors provide an exceptionally detailed implementation guide (Section 4.3) and an actionable roadmap (Section 5) to bridge this gap, having actual physical weight evaluation numbers would significantly elevate the paper's significance.

### Weakness 2: Dynamic Extensibility Constraints (Reference Re-centering)
As the authors acknowledge in Section 3.5, because C-Lie-MM performs tangent-space ensembling relative to a fixed coordinate reference point $Y_0$ (the offline projection-metric Karcher mean centroid), the coordinate representation system is anchored to the initial set of $K$ task experts. Adding a new expert post-deployment requires re-centering the coordinate reference system by re-evaluating $Y_0$ and re-computing the logarithmic maps for all experts. Although this offline cost is extremely small (taking less than a second), it represents a structural constraint on dynamic extensibility in modular, plug-and-play serving environments where experts are dynamically loaded and unloaded on the fly.

### Weakness 3: Hyperparameter Sensitivity of Soft Sign Tracking
In Section 3.7, the authors introduce an elegant soft sign-alignment tracking function $s_{i,\text{soft}}^{(t)} = \tanh(\beta (u_i^{(t)})^T u_i^{(t-1)})$ to guarantee infinitely differentiable ($C^\infty$) transitions across the zero boundary under backpropagation. However, the paper does not discuss how sensitive joint training stability is to the choice of the scaling hyperparameter $\beta \gg 1$ (set to 1000 in the text). A brief discussion or guidelines on setting $\beta$ would be helpful for practitioners seeking to implement this safeguard.

---

## 4. Rating and Recommendations

- **Soundness: Excellent**
  The mathematical derivations, proofs, and robust coordinate-free polynomial serving formulations are technically flawless, elegant, and highly rigorous.
- **Presentation: Excellent**
  The manuscript is beautifully structured, clearly written, and exceptionally easy to follow. It positions itself perfectly in the context of prior literature and provides comprehensive, self-contained algorithms and implementation details.
- **Significance: Excellent**
  The work addresses a crucial, highly relevant problem in modern deep learning (dynamic ensembling of low-rank structures and PEFT adapters). By introducing a mathematically rigorous, manifold-preserving alternative to flat heuristics, it has the potential to heavily influence future multi-task serving and merging architectures.
- **Originality: Excellent**
  The paper presents a highly creative and novel combination of Riemannian geometry, learning theory, and hardware-efficient systems design.

---

## 5. Actionable and Constructive Feedback for the Authors

To finalize this manuscript for camera-ready publication, the authors are encouraged to address the following points:

1. **Elaborate on Modular Dynamic Extensibility Workarounds:**
   In Section 3.5 (under "Dynamic Extensibility and Reference Re-centering"), discuss potential workarounds or hybrid ensembling architectures that could mitigate the reference re-centering constraint. For example, could a static, task-independent reference point (such as a random orthobasis or a PCA basis extracted solely from the pre-trained backbone's weights) be used to avoid re-centering when new experts are added, and what would be the impact on metric distortion?

2. **Add a Sensitivity Analysis or Guidelines for the Soft-Sign Parameter $\beta$:**
   Provide a brief theoretical or empirical discussion regarding the sensitivity of the soft-sign tracking function to the scaling parameter $\beta$. Offer a recommendation for how practitioners should scale $\beta$ relative to the learning rate or gradient magnitudes to ensure optimal smoothing.

3. **Extend Future Work with Real-World PEFT Benchmark Goals:**
   In Section 5 (Conclusion), explicitly mention plans to compile physical weight evaluation numbers on standard GLUE (for RoBERTa-Large) and GSM8K (for LLaMA-3-8B) benchmarks, comparing the downstream accuracy of C-Lie-MM against TIES-Merging and ZipIt on actual weight tensors, to complement the current high-fidelity simulations.
