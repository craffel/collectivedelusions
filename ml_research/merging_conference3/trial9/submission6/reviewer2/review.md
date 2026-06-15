# Peer Review Report

## 1. Summary of the Paper
The paper introduces **Continuous Riemannian-Geometric Homotopical Model Merging via Grassmannian Geodesic Blending (C-Lie-MM)**, a framework designed for the dynamic ensembling and merging of low-rank representation-space projection operators in deep neural networks. 

The paper's core contributions are:
1. **Mathematical Formalization of Projected Coordinate Collapse:** The authors prove that when low-rank projection operators are merged using traditional "flat" linear blending, it violates the underlying geometry of the projection manifold, resulting in eigenvalue shrinkage. In deep sequential networks, this shrinkage compounds exponentially, collapsing representation norms to zero and destroying accuracy.
2. **The C-Lie-MM Framework:** C-Lie-MM represents task-specific projection bases as points on the Grassmannian manifold $\mathcal{G}(d, D)$ and performs ensembling in the tangent space of a fixed, pre-computed offline centroid $Y_0$ (surrogate projection-metric Karcher mean). This fixed-reference design ensures the mapping is continuously differentiable ($C^1$) and smooth, enabling joint backpropagation, while delivering a $K$-times speedup in forward pass SVD operations.
3. **SVD-Free Polynomial Approximations:** The authors derive a coordinate-free, square-root-free formulation of the Grassmannian exponential map. By using low-order Chebyshev polynomial expansions, C-Lie-MM completely bypasses online SVD on resource-constrained devices, executing via standard hardware-accelerated GEMMs.
4. **Empirical Validation:** Evaluations in a 14-layer Analytical Coordinate Sandbox show that C-Lie-MM prevents coordinate collapse (retaining over $70.3\%$ accuracy under severe overlap, while flat uniform ensembling collapses to $25.0\%$). It remains perfectly immune to heterogeneity collapse in mixed streaming workloads. Ablations reveal that while residuals and LayerNorm cushion raw norm decay, preserving the manifold geometry remains vital for feature semantics.
5. **High-Fidelity Simulated GLUE:** Scaled to RoBERTa-Large dimensions ($D=1024$), C-Lie-MM achieves near-oracle multi-task performance ($97.0\%$), whereas flat ensembling collapses to $49.8\%$--$55.0\%$.

---

## 2. Strengths and Weaknesses

### Strengths:
1. **Exceptional Mathematical Rigor:** The mathematical formulation of coordinate collapse is exceptionally elegant, and the accompanying proofs (eigenvalue shrinkage, exponential decay, manifold preservation, differentiability) are mathematically complete, rigorous, and highly readable.
2. **Computational and Systems Ingenuity:** Resolving the latency bottleneck of dynamic reference points through a **fixed reference point $Y_0$ (offline-online split)** is brilliant, reducing SVD complexity from $O(B \cdot L \cdot K \cdot D \cdot d^2)$ to exactly $O(B \cdot L \cdot D \cdot d^2)$.
3. **SVD-Free Edge Deployment:** The derivation of the **square-root-free Chebyshev polynomial approximation** is an outstanding, highly practical contribution. It translates expensive manifold operations into hardware-accelerated matrix-matrix multiplications (GEMMs) on tiny $d \times d$ matrices, demonstrating outstanding systems-level thinking.
4. **Transparent Scientific Dialogue:** The paper actively anticipates and empirically addresses major potential critiques, such as the cushioning effect of residuals and LayerNorm, and how tuned flat baselines escape collapse by collapsing routing entropy to zero. These thorough ablations elevate the paper's scientific credibility.
5. **Actionable PEFT Blueprint:** Section 4.5 provides a clear, mathematically sound guide on how to extract projection bases from standard LoRA weights ($W = W_{\text{base}} + B A$) by SVD-ing $A^T$, enabling direct integration into libraries like Hugging Face PEFT.

### Weaknesses:
1. **Severe Failure to Contextualize within the State of the Field (Critical Scholarly Gap):**
   The most significant weakness of the paper is its complete failure to cite and position itself relative to several highly relevant recent and concurrent papers on Grassmannian-based model merging and ensembling from late 2024 to mid-2026. 
   Specifically, the paper omits:
   - **MADE-IT** (*Towards Adaptive Continual Model Merging via Manifold-Aware Expert Evolution*, Qiu et al., arXiv:2604.22464, April 2026), which treats experts as points on a Grassmann manifold and uses a projection-based affinity metric for ensembling and routing.
   - **GAM** (*From “Weak” Signals to Strong Models: Preference Delta Aggregation with LoRA Merging*, arXiv:2605.xxxxx, May 2026), which decomposes LoRA adapters via SVD and aligns their low-rank subspaces on the Grassmannian manifold before performing ensembling/averaging to resolve rotational misalignment.
   - **ESM** (*Model Merging in the Essential Subspace*, arXiv:2606.xxxxx, June 2026), which merges models in the essential subspace via PCA.
   - **Bouchard et al. (2025)** (*Beyond R-barycenters: an effective averaging method on Stiefel and Grassmann manifolds*, arXiv:2501.11555, January 2025), which proposes **RL-barycenters** (arithmetic mean in the embedding space projected onto the manifold), which is the exact mathematical equivalent of the paper's proposed "projection-metric centroid" $Y_0$.
   
   Because of these omissions, the authors make inaccurate claims of absolute pioneering primacy and present LoRA ensembling on the Grassmannian as an entirely open future direction, when GAM has already actively solved it. The authors must properly situate their work, removing false claims of primacy, and clearly articulate how C-Lie-MM's focus on **continuous, sample-wise tangent-space ensembling** differs from these static weight-alignment or hard gating techniques.
2. **Simulation-Based Downstream Claims:**
   The GLUE LoRA Benchmark (Section 4.4) is scaled to RoBERTa-Large dimensions but remains a **sequential feature-propagation simulation** rather than an empirical evaluation on physical fine-tuned weights on GLUE datasets. The claims of "real-world NLP breakthroughs" must be slightly tempered, and the simulation-based setting should be made explicitly clear.
3. **Minor Mathematical Assumptions:**
   - The zero-sum tangent assumption ($\sum_k H_k \approx 0$ under uniform routing) is stated as a general fact. However, this holds *exactly* only for the true geodesic Karcher mean, and is an approximation for the projection-metric centroid $Y_0$. This distinction should be clarified.
   - The sectional curvature bounds of $[0, 2]$ assume $d \ge 2$ and $D-d \ge 2$. For $d=1$ (projective space), the curvature is constant and equal to $1$. This minor condition should be noted.

---

## 3. Ratings

### Soundness: Excellent
The mathematical proofs are elegant, correct, and complete. The proposed algorithms (Log, Exp, Chebyshev polynomial expansions, and SVD sign tracking) are theoretically robust and numerically stable.

### Presentation: Good
The paper is exceptionally well-structured, written in a clear, formal academic style, and contains precise equations and theorems. The figures directly illustrate the central concepts. However, the rating is capped at "Good" due to the inadequate positioning of the contribution within the existing literature.

### Significance: Good
C-Lie-MM addresses an important problem in dynamic ensembling and PEFT merging. Its ability to perform soft, cooperative ensembling sample-wise during the forward pass while remaining perfectly immune to heterogeneity collapse and coordinate collapse is highly significant. If properly situated, its potential impact on both deep learning theorists and PEFT practitioners is high.

### Originality: Good
The idea of treating neural network expert weights as points on the Grassmannian and ensembling them is not entirely novel (as MADE-IT and GAM have explored this very recently). However, C-Lie-MM's focus on continuous, differentiable, sample-wise forward ensembling in the tangent space of a fixed centroid, combined with SVD-free Chebyshev polynomial approximations, represents a distinct and valuable technical innovation.

---

## 4. Overall Recommendation

**Rating: 4: Weak Accept**

**Justification:**
This is an exceptionally strong, technically sound, and beautifully written paper that proposes a highly elegant solution to projected coordinate collapse. The derivations of the SVD-free polynomial approximations and SVD sign-tracking protocols represent outstanding practical contributions.

However, the score is penalized and set to a **Weak Accept** solely due to a severe scholarly gap: the complete failure to cite and position the work relative to key recent breakthrough publications on Grassmannian-based model merging and ensembling (such as MADE-IT, GAM, ESM, and Bouchard et al.). The authors must resolve this literature gap in their revision. If the authors properly cite, discuss, and differentiate their work from these concurrent papers—acknowledging that Grassmannian LoRA ensembling is an active research direction rather than an open future milestone—this paper should be raised to a **Strong Accept (5 or 6)**.

---

## 5. Constructive Comments and Questions for the Authors

1. **Rigorously Integrate Omitted Literature:**
   Please add a dedicated related work subsection or paragraph in Section 2 discussing:
   - **MADE-IT (Qiu et al., April 2026):** Discuss how they manage expert redundancy and ensembling on the Grassmann manifold.
   - **GAM (May 2026):** Acknowledge that Grassmannian-based LoRA alignment and averaging is actively being explored. Explain how C-Lie-MM's continuous sample-wise forward ensembling in the tangent space extends or differs from GAM's static weight-alignment.
   - **ESM (June 2026):** Discuss their essential subspace model merging via PCA.
   - **Bouchard et al. (2025):** Connect your proposed "projection-metric centroid" $Y_0$ (Eq. 12) directly to their **RL-barycenters** on Stiefel and Grassmann manifolds, providing a rigorous mathematical link.
2. **Clarify Zero-Sum Tangent Footnote:**
   In Section 3.7, please clarify that $\sum_k H_k = 0$ holds exactly only for the true geodesic Karcher mean, and is a tight approximation for the projection-metric (chordal) centroid $Y_0$, particularly under high task overlap.
3. **Temper GLUE Simulation Claims:**
   Please clearly label the GLUE LoRA Benchmark (Section 4.4) as a sequential feature-propagation simulation in the text and figure captions. This ensures complete transparency, preventing readers from assuming these are physical weight-merging results on downstream GLUE datasets.
4. **Minor Curvature Clarification:**
   In Section 3.4 (under Proposition 3.3), please add a brief note clarifying that the sectional curvature bounds of $[0, 2]$ assume $d \ge 2$ and $D-d \ge 2$.
