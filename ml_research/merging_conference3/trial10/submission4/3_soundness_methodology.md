# Soundness and Methodology Check

## 1. Mathematical Soundness of the Proposed Techniques

The mathematical derivations and proofs provided in the paper are theoretically rigorous, with consistent notation and sound logic:

### A. EF-Smooth Noise Shaping
* The paper models the error propagation of EF-Smooth as a first-order finite-impulse response (FIR) noise-shaping filter.
* The sum of discrepancy vectors telescopes when decay factor $\beta = 1.0$:
  $$\sum_{l=l_0}^{l_0+H} \left( \boldsymbol{\alpha}^{(l)} - \tilde{\boldsymbol{\alpha}}^{(l)} \right) = \mathbf{e}^{(l_0+H)} - \mathbf{e}^{(l_0-1)}$$
* Under standard uniform rounding bounds where $|e_k^{(l)}| \leq \frac{s_{\alpha}}{2}$, the cumulative blending error is bounded by:
  $$\left| \sum_{l=l_0}^{l_0+H} \left( \alpha_k^{(l)} - \tilde{\alpha}_k^{(l)} \right) \right| \leq s_{\alpha}$$
* This bound holds **independent of depth $H$**, preventing error variance from growing linearly with network depth ($Var \propto H+1$).

### B. Activation Error Feedback (AEF) and Theorem 3.1
* Let $h^{(l)}_{\text{unquantized}} = \tilde{h}^{(l-1)} + \text{pull}^{(l)} + e^{(l-1)}_{\text{act}}$ and $e^{(l)}_{\text{act}} = h^{(l)}_{\text{unquantized}} - \tilde{h}^{(l)}$.
* Due to telescoping and zero initial error feedback ($e^{(3)}_{\text{act}} = \mathbf{0}$):
  $$\tilde{h}^{(L)} - \tilde{h}^{(3)} = \sum_{l=4}^L \text{pull}^{(l)} - e^{(L)}_{\text{act}}$$
* Bounding the coordinate-wise error $|e_{\text{act}, d}^{(L)}| \leq \frac{s_{\text{act}}}{2}$, we obtain:
  $$\left\| \tilde{h}^{(L)} - \left( \tilde{h}^{(3)} + \sum_{l=4}^L \text{pull}^{(l)} \right) \right\|_2 \leq \frac{s_{\text{act}} \sqrt{D}}{2}$$
* This proof is mathematically solid, elegant, and correct. A major advantage of this proof is that it shows the accumulated error remains strictly bounded by a single-layer quantization step size across arbitrary network depths, eliminating any risk of register overflow on-device.

### C. Intellectual Honesty (Remark 3.2 & Trajectory Divergence)
* The authors acknowledge a critical scientific caveat in **Remark 3.2**: the "ideal accumulated trajectory" defined as $\tilde{h}^{(3)} + \sum_{l=4}^L \text{pull}^{(l)}$ uses the pull vectors calculated using intermediate quantized states, which can mathematically diverge from the true continuous unquantized floating-point trajectory ($h^{(L)}_{\text{float}} = h^{(3)}_{\text{float}} + \sum_{l=4}^L \text{pull}^{(l)}_{\text{float}}$).
* To address this, the authors provide empirical validation showing that this trajectory divergence remains extremely small in practice (mean $\ell_2$ distance of **0.0413** at Layer 14), verifying that the feedback-driven trajectory divergence is benign and does not affect classification decision boundaries. This is highly transparent, rigorous, and scientifically sound.

---

## 2. Mathematical Rigor of Permutation-Invariant Single-Pass Apportionment (PI-SPA)

In Section 3.4, the authors introduce the **Permutation-Invariant Single-Pass Apportionment (PI-SPA)** algorithm to bypass the non-parallelizable $O(K \log K)$ sorting bottleneck of Hamilton's apportionment on specialized vector hardware.

### How PI-SPA Achieves Strict Permutation Invariance:
1. **Deterministic Tie-Breaking:** PI-SPA perturbs the fractional remainders $r_k$ using a tiny deterministic tie-breaker based on their static, unique expert identifier:
   $$r'_k = r_k + \epsilon \cdot \text{ID}_k$$
   where $\text{ID}_k \in \{0, \dots, K-1\}$ is the unique expert index and $\epsilon \ll 1/L$ is a microscopic constant (e.g., $\epsilon = 10^{-5}$ for $L=15$).
   * Since each expert has an immutable ID that does not change when the input tensor is permuted, the perturbed remainders $r'_k$ are strictly unique across all experts.
2. **Threshold-Based Allocation:** It then determines the $S$-th largest perturbed remainder as our allocation threshold:
   $$\theta = \text{Select}\left(\{r'_j\}_{j=1}^K, S\right)$$
   and increments the allocation for the $S$ experts meeting this threshold:
   $$q_k \leftarrow q_k + 1 \quad \text{for } k \text{ where } r'_k \geq \theta$$
   * Because the perturbed remainders $r'_k$ are strictly unique, there are guaranteed to be **exactly $S$ experts** meeting this threshold, ensuring that the shortfall is fully allocated and the ensembling weights strictly sum to 1.0.
   * Since finding the $S$-th largest element can be executed in $O(K)$ time (using parallel selection or a tiny bitonic network), it avoids any branch-heavy sorting.
   * Because the tie-breaker is a static property of the expert, the set of selected experts is **strictly invariant** to any permutation of the expert list in memory, completely eliminating compiler-induced build fragility.
   * Because $\epsilon \ll 1/L$, the ordering is dominated by the remainder magnitude $r_k$; static IDs only resolve ties when remainders are identical, preserving high remainder-magnitude sensitivity.

This is a mathematically brilliant and hardware-elegant solution!

---

## 3. Highly Technical Critiques and Areas of Clarification

While the methodology is exceptionally sound, a rigorous peer-review reveals three areas where the theoretical or implementation details should be further clarified:

### A. Alignment of Logit Scales and Cosine Similarity in Gating
* **Critique:** In Section 3.3, the gating logit is computed using integer matrix multiplication to obtain $z'_{k, b}$ in a 32-bit register. To preserve mathematical equivalence, these integer logits are scaled by $s_z / \tau$ inside the softmax. However, SABLE and SABLE-derived variants (such as QA-Merge SABLE and ChemMerge) combine the parametric routing logits $z_k$ with the scale-invariant cosine similarity gating distance $d_k$ before the softmax: $\boldsymbol{\alpha}_{\text{raw}} \leftarrow \text{softmax}(\mathbf{z} + \mathbf{d})$ (as shown in Algorithm 1, Step 3).
* **Impact:** In an integer-only register pipeline, $z'_{k, b}$ is an unscaled integer, whereas $d_{k, b}$ is a fractional cosine similarity bounded in $[-1, 1]$. If the addition of $\mathbf{z}$ and $\mathbf{d}$ is executed prior to the softmax, their scales must be strictly aligned. Specifically, adding the raw 32-bit integer $z'_{k, b}$ directly to a fractional $d_k$ is mathematically invalid.
* **Recommendation:** The authors should explicitly clarify the register-level scale alignment protocol used to combine the integer-only logits $z'_k$ and the cosine similarities $d_k$ before the softmax. For instance, are the cosine similarities scaled by $1 / s_z$ in fixed-point, or are the parametric logits scaled to floats before the addition?

### B. Out-of-Distribution Calibration with Percentile-Based Scaling
* **Critique:** In Section 3.1, the authors employ a percentile-based calibration strategy (specifically the $99.9$-th percentile of absolute activation values in the calibration set) to determine the activation scale factor $s_{\text{act}}$, combined with hardware clipping to saturate OOD outliers.
* **Impact:** While percentile-based calibration is highly robust compared to maximum-value calibration, extreme OOD inputs could still lead to severe activation clipping or saturation, which could degrade the downstream representation fidelity.
* **Recommendation:** The authors should briefly discuss whether dynamic activation scaling (where $s_{\text{act}}$ is updated on the fly per sample or per mini-batch) would be more suitable than static percentile-based calibration under highly non-stationary out-of-distribution streams.

### C. AEF Integration with Dynamic Layer-wise Scale Factors
* **Critique:** The formulation of Activation Error Feedback (AEF) in Section 3.5 assumes a constant activation scale factor $s_{\text{act}}$ across layers. In practice, dynamic scale factors are often computed layer-by-layer to maximize the INT8 representation range.
* **Impact:** If the scale factor $s_{\text{act}}^{(l)}$ varies layer-by-layer, adding the unaligned activation error $e_{\text{act}}^{(l-1)}$ (which is scaled by $s_{\text{act}}^{(l-1)}$) to the next layer's unquantized state is mathematically invalid.
* **Recommendation:** The authors should add a brief sentence in Section 3.5 referencing Appendix D.1 (which describes the dynamic scale realignment via Helium SIMD instructions) to make the main text self-contained and clear to systems engineers.
