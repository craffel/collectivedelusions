# 3_soundness_methodology.md - Detailed Soundness and Methodology Review

## Methodological Strengths
The paper exhibits strong mathematical rigor in several key areas:
1. **Watertight Sample-Splitting Protocol:** Partitioning the calibration set $\mathcal{S}_{\text{cal}}$ into $\mathcal{S}_{\text{prior}}$ (for learning the SVD bases) and $\mathcal{S}_{\text{opt}}$ (for optimizing the temperatures) is a highly sound design. It strictly guarantees that the prior distribution remains label-independent and data-independent of the optimization data, satisfying the core assumptions of PAC-Bayesian theory.
2. **First-Principles Derivation of Representation Interference:** Deriving the entropy-proportional noise model from first principles (expert direction misalignment) is an outstanding theoretical addition. It successfully addresses potential critiques of circularity in the simulation setup (ICS).
3. **Basis-Independence and Scale-Invariance:** Proving that Subspace Energy Projection (SEP) is mathematically invariant under orthogonal basis changes and activation scaling (Proposition 3.1) provides a strong foundation for deploying the method across deep networks of varying scales and dimensionalities.

## Methodological Gaps and Weaknesses

### 1. The Surrogate Loss Gap (Guarantees vs. Reality)
There is a fundamental mathematical gap between the theoretical guarantees of the PAC-Bayesian bound and the physical execution of the model:
* **The Theory:** The PAC-Bayesian bound is formulated and optimized using a **linear calibration surrogate** (Equation 15):
  $$\widehat{\mathcal{L}}_{\text{cal}}(\boldsymbol{\tau}) = \frac{1}{N_{\text{opt}}} \sum_{b=1}^{N_{\text{opt}}} \left( 1 - \sum_{k=1}^K \alpha_{k, b} \cdot p_k(x_b) \right)$$
  which represents the expected error under a stochastic expert-selection policy.
* **The Practice:** In test-time serving, the model executes **activation-space blending** (Equation 4):
  $$h^{(l)} = h^{(l-1)} + \sum_{k=1}^K \alpha_{k, b} \cdot \Delta h_k^{(l)}$$
  which is passed through subsequent non-linear layers and final classification heads.
* **The Gap:** Because downstream layers are non-linear, the actual classification error of the blended activation network is *not* mathematically equal to the linear surrogate loss. Consequently, **the PAC-Bayesian bound does not formally guarantee the generalization of the deployed activation-blending model.** It only guarantees the generalization of the linear proxy. While the authors discuss this computational shortcut in Section 3.5, they gloss over the fact that it breaks the formal connection between the PAC-Bayesian certificate and the physical serving performance.

### 2. Underdetermined SVD Bases under Scaling ($K$)
In Section 3.2, the authors discuss the "Subspace Rank Underdetermined Boundaries under Growing Expert Counts". This is a critical soundness issue:
* If the total calibration budget $N_{\text{prior}}$ is fixed, then as the number of experts $K$ increases, the per-task prior sample size $N_{\text{prior}, k} = N_{\text{prior}}/K$ shrinks.
* Because the rank of each representation matrix $Z_k$ is bounded by $N_{\text{prior}, k}$, the SVD spectrum is truncated.
* If $N_{\text{prior}, k} \le d$ (the target subspace dimension), the SVD becomes underdetermined and spans transductive noise rather than the task manifold.
* While the authors acknowledge this boundary, they do not provide a concrete algorithmic solution (such as Ledoit-Wolf covariance shrinkage or randomized projection regularizers) in their implementation, leaving the system highly vulnerable to failure as the expert registry scales.

### 3. Coarse Discretization and Vacuous Bound Values
The discretization argument used to achieve uniform convergence over the global temperatures $\boldsymbol{\tau}$ relies on a union bound over a grid of size $|\Theta| \le \left(\frac{\tau_{\text{max}} - \tau_{\text{min}}}{\epsilon}\right)^K$. 
* Under small calibration splits (e.g., $N_{\text{opt}} = 24$ as in the real-world experiments), the discretization penalty term $\sqrt{\frac{\ln |\Theta|}{2 N_{\text{opt}}}}$ becomes massive.
* For $K=4$, $\tau_{\text{max}} - \tau_{\text{min}} = 10.0$, and a grid resolution $\epsilon = 0.01$, $|\Theta| \approx 10^{12}$, and $\ln |\Theta| \approx 27.6$. The penalty term alone adds $\approx 0.76$ to the bound.
* This renders the actual certifiable PAC-Bayesian bound values **vacuous** (exceeding 1.0 for a loss bounded in $[0, 1]$).
* The authors drop the constant $\ln |\Theta|$ during optimization (which is standard), but the theoretical "certificate" provided by the bound is practically meaningless at these sample sizes.

### 4. High Sensitivity to Prior Temperature ($\tau_0$)
Dirichlet-PAC stabilizes routing temperatures by penalizing their deviation from a static prior temperature $\tau_0$ (set to $0.20$).
* In highly data-scarce settings (e.g., $N = 24$), the KL penalty completely dominates the optimization. As a result, the learned temperatures $\boldsymbol{\tau}$ barely deviate from the prior (converging to $\approx 0.19$).
* This means Dirichlet-PAC's superior performance over ERM is primarily because the optimization is heavily regularized, forcing it to stick close to the prior temperature.
* This makes the method highly dependent on the manual choice of $\tau_0$. If $\tau_0$ is poorly selected (e.g., $\tau_0 = 0.05$ as shown in Table 4), the ensembling accuracy drops significantly ($76.54\% \to 70.48\%$).
* Therefore, Dirichlet-PAC does not "solve" temperature selection; it merely acts as a heavy anchor to a manually-tuned prior temperature $\tau_0$.
