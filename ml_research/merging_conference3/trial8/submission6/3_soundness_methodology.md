# 3. Soundness of Methodology: LoRA Subspace Projection Routing (LSPR)

## 3.1 Mathematical and Theoretical Foundations
The paper displays an exceptionally high degree of mathematical soundness. Every claim is supported by a clear, closed-form algebraic formulation or a formal mathematical proof.

### A. The Adapter Sensitivity Theorem
The authors prove that the magnitude of an adapter's parameter-efficient update $\Delta y_b = h_b A_k B_k$ is bounded by the projection energy of the entering activation vector $h_b$ onto the orthonormal basis $Q_k$ of the column space of $A_k$:
$$\| \Delta y_b \|_2 \le \| h_b Q_k \|_2 \| R_k B_k \|_{op}$$
- **Proof Correctness:** The proof uses standard properties of QR decomposition ($A_k = Q_k R_k$), matrix/vector product norm inequalities, and the sub-multiplicative property of spectral norms. It is mathematically flawless.
- **Significance:** It formally proves that an adapter is completely inactive ($\Delta y_b = \mathbf{0}$) if the incoming activation vector $h_b$ is orthogonal to the column space of its down-projection matrix $A_k$. This establishes the mathematical rationale for why the projection energy score serves as a highly precise, zero-shot task routing similarity metric.

### B. Joint Training Loss and Representational Alignment
A major contribution of this paper is explaining why standard LoRA fine-tuning fails under geometric routing:
- **Standard LoRA Failure Mode:** Standard training optimizes classification loss ($\mathcal{L}_{\text{classification}}$) alone. Since the up-projection matrix $B_k$ is typically initialized to zero to ensure the adapter is inactive at startup, gradients flowing back to $A_k$ (which are scaled by $B_k$) remain vanishingly small. Consequently, $A_k$'s column space remains random and unaligned with the task activation manifold.
- **The Joint Objective:** The paper introduces a joint loss:
   $$\mathcal{L} = \mathcal{L}_{\text{classification}} + \lambda \mathcal{L}_{\text{reconstruction}}$$
   where $\mathcal{L}_{\text{reconstruction}}$ is a subspace autoencoding objective. Under backpropagation, this reconstruction constraint mathematically forces the columns of $A_k$ to converge directly to the principal components of the task activation distribution, physicalizing the alignment.
- **Overhead Analysis:** The authors honestly analyze the overhead of this joint training loss, noting a modest 15-20% increase in GPU memory and a minor FLOP overhead of approximately $O(B \cdot r \cdot D)$ per step during training, which is a highly favorable trade-off for zero-shot deployment.

---

## 3.2 Key Methodological Details and Systems Aspects

### A. Layer-Wise Freezing Scheme
In multi-layer backbones, LSPR computes ensembling coefficients $\alpha_{k, b}$ only at the first adapter layer (Block 4) and freezes/re-uses them for all downstream layers.
- **Mathematical Soundness:** The authors show that downstream layers are trained with standard classification loss alone, preserving their downstream representation capacity.
- **Empirical Validation:** The paper includes a dedicated empirical validation comparing:
  - **Layer-Wise Freezing (Ours):** Recovers 100% of the Expert Ceiling (74.09% Joint Mean Accuracy).
  - **Layer-Wise Recomputation:** Yields only 51.43% accuracy.
- **Insight:** Since downstream layers are not trained with reconstruction loss, their weight spaces are unaligned. Recomputing routing coefficients on these downstream layers results in random, noisy routing that dilutes activations. Freezing early-layer coefficients is therefore not only computationally efficient but mathematically superior.

### B. Split-Rank LoRA
To mitigate potential downstream capacity dilution when forcing an extremely small bottleneck (e.g., $r=8$) to reconstruct high-dimensional activations, the authors propose a **Split-Rank LoRA** strategy.
- **Mechanism:** The bottleneck rank $r$ is split into $r_{\text{route}}$ columns (which are trained with the joint classification-reconstruction loss) and $r_{\text{task}}$ columns (which are dedicated solely to task adaptation with classification loss only).
- **Mathematical Soundness:** This separates the subspace projection routing basis from the task adaptivity parameters, ensuring zero capacity trade-offs during multi-task adaptation.
- **Empirical Validation:** Split-Rank LoRA ($r=8$) recovers 84.11% accuracy, while maintaining a strong subspace alignment of $0.5447$ on its dedicated routing columns.

### C. Sparse-LSPR for Massive Registries
To avoid the $O(B \cdot K \cdot r \cdot D)$ scaling complexity of parallel ensembling, Sparse-LSPR introduces a Top-$M$ sparse gating mechanism.
- **Mathematical Soundness:** Projecting $h_b \in \mathbb{R}^{1 \times D}$ onto $K$ orthonormal bases $Q_k \in \mathbb{R}^{D \times r}$ requires $K$ independent matrix-vector products, which scales as $O(K \cdot r \cdot D)$ but involves no up-projections, non-linearities, or downstream execution. Once the active set $\mathcal{M}_b$ of size $M$ is selected, the heavy adapter computations scale as $O(B \cdot M \cdot r \cdot D)$.
- **Significance:** Since $M$ is fixed (e.g., $M=2$), the heavy compute cost is completely decoupled from the registry size $K$, allowing constant-time scaling on CPU/edge devices.

### D. Post-Hoc Warm Alignment
For existing public adapters, LSPR proposes a fast fine-tuning step: freezing $B_k$, downstream adapters, and classification heads, and only fine-tuning $A_k$ of Block 4 on the reconstruction loss.
- **Soundness of Constraints:** Because $B_k$ and classification heads are frozen, the downstream classification and representational capabilities of the expert are guaranteed to suffer exactly 0% degradation.
- **The Conceptual Boundary (Honesty):** The authors note a crucial limitation: Warm Alignment *must* be performed using domain-specific or task-aligned queries. If multiple experts are warm-aligned on the same general task-agnostic background dataset, their subspaces will converge to the same principal components, destroying geometric distinction and collapsing test-time routing. This is an incredibly rigorous, honest, and valuable boundary to establish.

---

## 3.3 Anisotropic Threshold Calibration
A common weakness of zero-shot routing is the arbitrary selection of static hyperparameters (such as $\gamma_{\text{OOD}}$). The authors resolve this with extreme mathematical rigor:
1. **Analytical Baseline under Isotropic Assumptions:** They model out-of-distribution (OOD) activations as uniform random vectors on the unit sphere $\mathbb{S}^{D-1}$. Under spherical symmetry, they show that the squared projection score follows a Beta distribution, yielding an expected isotropic projection score of:
   $$\mathbb{E}[u_{\text{OOD}, k}] \approx \sqrt{\frac{r}{D}}$$
   For their sandbox ($D=192, r=8$), this yields $\approx 0.204$.
2. **Correction for Anisotropy (Representation Cone Effect):** In practical models, activations suffer from representation collapse and are restricted to a narrow cone, inflating the OOD score. The authors define the effective dimensionality $d_{\text{dom}} \ll D$ of the dominant subspace $C$ (where cumulative explained variance of the activation covariance matrix $\Sigma$ reaches 95%). This concentrates OOD queries, shifting the expected score upwards to:
   $$\mathbb{E}[u_{\text{OOD}, k}] \approx \sqrt{\frac{r}{d_{\text{dom}}}}$$
   With $d_{\text{dom}} \approx 40$ in their sandbox, the noise floor shifts to $\approx 0.447$.
3. **Hybrid Calibration Strategy:** Based on this, they propose a data-free hybrid calibration: pass a small, task-agnostic set of unlabeled background queries (e.g., random internet text/images) through the early layers of the model to measure the empirical noise floor, and set $\gamma_{\text{OOD}}$ safely above it (e.g., at the 99th percentile). This fully adapts to the anisotropy of different models without requiring target task labeled data.

## 3.4 Summary Assessment of Soundness
The methodology is exceptionally sound. Every design decision—from joint training to layer-wise freezing, warm alignment, split-rank LoRA, and anisotropic threshold selection—is backed by rigorous theoretical proofs, statistical derivations, and empirical confirmation. There are no hand-wavy claims, and the authors are highly transparent about potential failure modes and boundary conditions.
