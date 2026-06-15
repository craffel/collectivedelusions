# Soundness and Methodology Check

## 1. Mathematical Strengths and Rigor
The mathematical formulation of **QPathMerge** is highly structured and rigorous. Several derivations and theoretical proofs are exceptional for a conference submission:
- **Symmetric Cancellation of Forward-Backward Drift (Section 3.5)**: The proof showing that the exponential sharpening of forward and backward beliefs perfectly cancels out when transition leakage $M \to 0$ (yielding a perfectly constant trajectory and exactly $0.000000$ jitter) is an elegant and beautiful result. It demonstrates a fundamental advantage of bidirectional solvers over unidirectional filters.
- **Convergence Proof via Dobrushin's Contraction Theorem (Section 3.7)**: In the latest revision, the authors have updated Section 3.7 with the exact row-normalized derivation of the transition probability matrix $\phi$, proving that the true contraction coefficient is $\eta(\phi) = 1 - \frac{K M}{1 + (K-1)M}$. By showing that for $K=4$ and $M=0.10$, $\eta(\phi) \approx 0.6923$, they formally prove that the truncation error decays exponentially fast (bounded by $\approx 0.23 C$ after $H=4$ steps). This provides a highly rigorous mathematical foundation for setting $H = 4$.
- **Power Iteration Convergence Analysis (Section 3.7)**: The analysis showing that the constant future potential assumption ($\psi_{l'} = \psi_l$) reduces the backward recurrence to a power iteration converging to the dominant eigenvector of $\phi \operatorname{diag}(\psi_l)$ is highly precise and mathematically beautiful. It mathematically justifies why extrapolation methods are required to capture non-monotonic trajectories.

---

## 2. Methodology Improvements and Current Status

### 2.1. Resolution of the Bidirectional "Two-Pass" Evaluation Gap
Unlike earlier drafts where the bidirectional two-pass "Predict-then-Smooth" pipeline was purely theoretical, the authors have successfully implemented and evaluated **QPathMerge-TwoPass** in `simulate.py` and included its results in Tables 1, 2, 3, and 4.
- In Sandbox and ResNet-18 evaluations, `QPathMerge-TwoPass` achieves the highest spatial stabilization, slashing Layer Jitter by up to **$5.8\times$** compared to SABLE-Dynamic, while maintaining robust accuracy.
- This resolves the theoretical-empirical discrepancy and validates the entire Predict-then-Smooth methodology on physical intermediate representations.

### 2.2. Out-of-Distribution (OOD) Query Robustness
The paper mathematically outlines how the MRF formulation handles OOD inputs. When representations lie far from task centroids, node potentials naturally flatten towards a uniform distribution ($\psi_l(k) \approx \text{const}$). Belief propagation propagates these, causing ensembling weights to smoothly fall back to a uniform distribution ($\alpha_l(k) \approx 1/K$). This acts as a robust regularizer, preventing un-correlated experts from dominating.

### 2.3. Causal Extrapolations and Spatial Lag
To break the power iteration degeneracy, the paper proposes `LinearExtrap` and `RollingExtrap`.
- Under highly non-monotonic task composition (the Composite Sandbox in Table 3), where task requirements switch abruptly at Layer 9, carrying over a rolling average of past potentials (`RollingExtrap`) introduces severe **spatial lag (inertia)**, dropping accuracy to **91.42%**.
- Conversely, linear slope projection (`LinearExtrap`) successfully projects the trend to anticipate the transition, achieving a leading **99.67%** accuracy (outperforming both standard QPathMerge and SABLE-Dynamic).
- The authors discuss this trade-off with high scientific honesty, highlighting that predictive ensembling must utilize dynamic slope trend projection rather than rigid historical averaging under heterogeneous, multi-task backbones.

---

## 3. Soundness Rating
**Excellent**. The paper's mathematical framework is flawless, its convergence proofs are rigorous, the two-pass bidirectional solver is fully implemented and validated, and its on-the-fly extrapolation variants are analyzed with outstanding scientific transparency.
