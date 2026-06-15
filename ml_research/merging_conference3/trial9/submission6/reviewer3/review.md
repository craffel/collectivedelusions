# Peer Review

**Paper Title:** Continuous Riemannian-Geometric Homotopical Model Merging via Grassmannian Geodesic Blending (C-Lie-MM)

---

## 1. Summary of the Paper
This paper proposes **Continuous Riemannian-Geometric Homotopical Model Merging via Grassmannian Geodesic Blending (C-Lie-MM)**, a framework designed to dynamically merge task-specific representation-space projection operators (low-rank projection matrices) during a model's forward pass. 

The authors identify a critical issue in traditional activation-blending and model-merging methods: flat linear interpolation of projection matrices violates the geometry of the projection manifold. They prove that this violation leads to **projected coordinate collapse**—where the eigenvalues of the merged operator shrink strictly below $1$, causing the norms of representation vectors to decay exponentially across sequential layers in deep networks.

To address this, C-Lie-MM treats individual task-specific projection spaces as points on the Grassmannian manifold $\mathcal{G}(d, D)$ and utilizes a fixed reference point $Y_0$ (the projection-metric Karcher mean centroid computed offline) to build a local coordinate system. The ensembling is performed along geodesic paths in the flat tangent space of $Y_0$ via:
1. Mapping task-specific expert bases to the tangent space of $Y_0$ using the Grassmannian logarithm map ($\log_{Y_0}$) offline.
2. Blending the tangent matrices sample-wise using dynamic, temperature-calibrated Gibbs routing weights online.
3. Projecting the blended tangent matrix back onto the Grassmannian manifold using the exponential map ($\exp_{Y_0}$).

This ensures that the merged projection operator is always symmetric, idempotent, and of rank $d$ (strictly on the Grassmannian manifold), completely eliminating coordinate collapse. The authors evaluate C-Lie-MM inside a 14-layer simulated "Analytical Coordinate Sandbox," demonstrating that it completely resolves representation collapse and maintains high performance under both homogeneous and heterogeneous mixed workloads.

---

## 2. Strengths and Weaknesses

### Strengths
- **Mathematical Rigor:** The paper is exceptionally thorough and mathematically sound. It provides complete derivations, formal proofs of manifold preservation (Theorem 3.2), differentiability (Theorem 3.3), and a strict lower bound on tangent-space metric distortion (Proposition 3.3).
- **Outstanding Scientific Transparency:** The authors are highly commendable for their self-critical and transparent discussion in Section 4.3. They openly analyze the ecological validity of the simulated sandbox, the role of residual connections and LayerNorm as protective geometric buffers, and the behavior of learned routing temperatures in flat baselines. This scientific honesty is exemplary.
- **Clear and Structured Presentation:** The writing is professional, precise, and extremely structured. Edge cases such as varying rank capacities ($d_k$), SVD sign-alignment ambiguities, and optimization stabilities (discrete vs. continuous exponential moving average centroid updates) are discussed with outstanding detail.
- **Comprehensive Systems-Level Implementation:** The authors have gone to great lengths to demonstrate systems-level viability, implementing a custom fused Triton GPU kernel in PyTorch to bypass GPU SVD execution overhead and a stabilized backpropagation SVD autograd operator (`StableSVD`).

### Weaknesses
- **Unjustified Mathematical Complexity & Over-Engineering:** 
  The primary critique of this paper lies in its immense, highly complex mathematical pipeline. The authors introduce a dense differential-geometric framework (Grassmannian manifolds, Karcher mean projections, logarithmic/exponential tangent mappings, Rauch comparison theorems, continuous sign-tracking tracking, and 6th-order Chebyshev matrix polynomial expansions) to solve a problem that can be resolved in a far simpler, more direct, and elegant way:
  - Given the task expert bases $\{V_k\}_{k=1}^K$, one could simply compute a flat linear combination $V_{\text{flat}} = \sum_{k=1}^K \alpha_k V_k$.
  - Because $V_{\text{flat}}$ is not necessarily orthogonal, it can be orthonormalized during the forward pass using standard, highly optimized linear algebra: $V_{\text{orth}} = \text{orth}(V_{\text{flat}})$ (via QR decomposition or SVD).
  - The resulting projection matrix $P_{\text{orth}} = V_{\text{orth}} V_{\text{orth}}^T$ is guaranteed to be symmetric, idempotent, and of rank $d$. This completely resolves projected coordinate collapse, is fully differentiable, requires no reference point $Y_0$, no offline coordination, and introduces no local tangent-space metric distortion (unlike C-Lie-MM's tangent space projection, which suffers from up to 55% distortion under orthogonal tasks).
  
  By failing to evaluate or discuss this obvious, highly elegant linear-algebraic alternative, the necessity of introducing C-Lie-MM's extreme mathematical and engineering overhead remains completely unjustified.
- **Lack of Real-World Evaluation:**
  All empirical evaluations are restricted to a synthetic, low-dimensional "Analytical Coordinate Sandbox" ($L=14$, $D=192$, $d=8$). The paper completely lacks physical validation on real-world benchmark datasets (such as GLUE or GSM8K) and actual model weights (such as LLaMA-3 or RoBERTa). Although the authors outline a detailed "integration roadmap" in the conclusion, a conference submission is expected to already possess physical validation to prove its viability under realistic workloads.
- **Complexity Spiraling:**
  The paper exhibits a clear pattern of complexity spiraling: the authors introduce a highly complex curved-manifold method (C-Lie-MM), discover that it is computationally heavy because it requires online sample-wise SVD operations on GPUs, and then layer on *another* level of high engineering complexity—a custom Triton GPU kernel executing 6th-order Chebyshev matrix polynomial expansions—to bypass the SVD. This entire spiral could be avoided by selecting a simpler baseline (such as the linear-algebraic "blend + QR" approach) that naturally parallelizes on modern GPUs.
- **Redundant Sign-Tracking Formulations:**
  The authors spend considerable effort detailing SVD sign ambiguities and proposing a soft-sign $\tanh$ alignment protocol. However, a fundamental property of the Grassmannian exponential map $\exp_{Y_0}(H)$ and the resulting projection matrix $P_{\text{merged}} = Y_{\text{merged}}Y_{\text{merged}}^T$ is that they are mathematically **completely invariant to column-sign flips** of the SVD of $H = U \Theta V^T$ (as both $U$ and $V$ columns are negated simultaneously, preserving the products $u_j v_j^T$ and $v_j v_j^T$). Therefore, the sign ambiguity is physically self-canceling and has zero effect on the forward pass or the mathematical gradients, rendering the extensive tracking protocols mathematically redundant.

---

## 3. Detailed Evaluation of Criteria

### Soundness: Good
The mathematical derivations, proofs, and simulated experiments are technically correct and rigorous. However, the soundness of the empirical claims in a practical deep network is weakened because:
1. The authors do not evaluate the most relevant, direct, and elegant alternative baseline (Linear Blend + QR/SVD Orthonormalization).
2. The ablation study in Section 4.3 shows that in realistic, "cushioned" architectures (equipped with residual identity paths and LayerNorm), raw coordinate collapse does not occur, and Uniform Merging achieves $51.90\%$ accuracy. This directly proves that "catastrophic coordinate collapse" is largely a product of the unbuffered, artificial feedforward setup of the simulator, weakening the practical necessity of the method.

### Presentation: Excellent
The writing, structure, formatting, and clarity of the paper are outstanding. The mathematical concepts are introduced logically, and the self-critical discussion is a masterclass in scientific honesty. SVD backpropagation, varying ranks, and systems-level tradeoffs are analyzed with high precision.

### Significance: Fair
The significance of the paper is heavily constrained by its extreme complexity and lack of real-world validation:
- Practitioners in the industry or ML community are highly unlikely to adopt a framework that requires pre-computing offline projection-centroids, executing online sample-wise SVDs, tracking column sign rotations, and compiling custom-fused Triton GPU kernels, especially when standard parameter-space merging (like TIES-Merging or Task Arithmetic) or a simple "blend + QR" approach can achieve representation blending with negligible complexity.
- Without demonstrating massive performance gains on physical networks and standard benchmarks that justify this extreme overhead, the practical utility of the proposed framework remains limited.

### Originality: Good
The paper provides a highly original combination of Grassmannian differential geometry and representation-space model ensembling. The fixed-reference tangent space formulation to bypass online log-map SVDs is a creative and original contribution.

---

## 4. Overall Recommendation

**Rating:** 3: Weak reject

**Justification:**
This paper has clear merits: it is mathematically rigorous, exceptionally well-written, and addresses an interesting, formal problem (representation-space projection ensembling) with impressive systems-level engineering (Triton kernels, stabilized autograd).

However, the weaknesses overall outweigh the merits. The paper is a classic case of **architectural over-engineering** and **complexity spiraling**. It introduces massive mathematical complexity to solve a coordinate collapse problem that is:
1. Already heavily mitigated by standard architectural buffers in real-world networks (residuals, LayerNorm).
2. Solvable in a vastly simpler, more elegant, and direct way (Linear Blend followed by standard QR/SVD Orthonormalization) which the authors completely omit from their evaluation and discussion.
3. Restored to systems-level viability only by introducing yet *another* layer of complex low-level engineering (fused Triton kernels with 6th-order Chebyshev matrix polynomial expansions) to bypass the online SVDs introduced by the method itself.

To make this paper ready for publication, the authors must simplify their approach, evaluate and compare against the elegant "Linear Blend + QR/SVD" baseline, and demonstrate physical, out-of-simulation performance gains on actual model weights and real-world benchmarks (such as GLUE or GSM8K) to prove that this immense complexity is actually justified by massive practical gains.

---

## 5. Questions and Constructive Suggestions for the Authors

1. **Comparison with Simple Orthonormalization Baselines:**
   Could you please evaluate a baseline where you simply compute $V_{\text{flat}} = \sum_k \alpha_k V_k$ and then orthonormalize the blended basis using standard QR decomposition or SVD ($V_{\text{orth}} = \text{orth}(V_{\text{flat}})$) during the forward pass?
   - How does this simple "Linear Blend + QR/SVD Orthonormalization" baseline compare to C-Lie-MM in terms of accuracy in both the unbuffered and residual/LayerNorm sandbox settings?
   - What is the GPU forward pass latency of this standard QR-based baseline compared to SABLE and your SVD/Chebyshev C-Lie-MM implementations?
2. **Evaluation on Real-World Datasets and Weights:**
   To establish physical viability, please evaluate C-Lie-MM on actual physical weights (e.g., merging task-specific LoRA adapters for RoBERTa-Large or LLaMA-3) on standard benchmarks like GLUE or GSM8K. Without physical validation, it is impossible to know if the synthetic sandbox findings generalize to actual deep representation spaces.
3. **Invariance of Exponential Map and Redundancy of Sign-Tracking:**
   Given that the Grassmannian exponential map $\exp_{Y_0}(H)$ and the projection matrix $P_{\text{merged}} = Y_{\text{merged}} Y_{\text{merged}}^T$ are mathematically invariant to SVD column-sign changes of $H = U \Theta V^T$, could you clarify why the continuous sign-tracking tracking (Eq. 21) and soft-sign tracking (Eq. 22) are necessary? If they are purely to prevent numerical autograd instabilities in specific deep learning libraries, this should be explicitly clarified as an implementation detail rather than presented as a fundamental geometric safeguard.
4. **Simplification of the Framework:**
   In the spirit of simplicity, can the framework be simplified? For instance, does a local projection onto a task-independent PCA reference basis extracted from the pre-trained backbone (as discussed in Section 3.7) perform just as well as the optimized Karcher mean centroid, while completely bypassing offline Karcher mean coordination?
