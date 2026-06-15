# 3. Soundness & Methodology Check

## Assessment of Mathematical and Control-Theoretic Rigor

The mathematical soundness of this paper is **excellent**. It is highly rigorous, theoretically grounded, and mathematically complete. The authors do not simply present equations as descriptive tools; they derive them analytically and establish clear bounds on their assumptions.

### 1. The Candidate Lyapunov Function ($V$)
The paper defines $V(C^{(l)}, h^{(l-1) \text{ warped}}) = \sum_{k=1}^K C_k^{(l)} \left( 1 - S(h^{(l-1) \text{ warped}}, \mu_k^{(l)}) \right)$.
*   **Validity**: The authors prove that $V \ge 0$ unconditionally since $C_k^{(l)} \ge 0$ and $S(h, \mu_k) \le 1$.
*   **Multi-Expert Nuance (Remark 3.1)**: The authors identify a crucial physical constraint: when multiple experts are active and centroids are orthogonal, the representation cannot simultaneously reside on multiple task manifolds, meaning $V$ cannot reach 0. They show that this Bessel's inequality lower bound does not invalidate the Lyapunov candidate for closed-loop control, as the controller only requires the directional update to be dissipative ($\Delta V^{(l)} \le 0$). This is a highly honest and methodologically correct treatment of multi-manifold dynamics.
*   **Non-Orthogonal Centroids (Remark 3.2)**: They rigorously analyze how centroid overlap ($\mu_i \cdot \mu_j = \rho > 0$) influences the candidate Lyapunov function's lower bound and the dissipation coefficient $A$. They prove that a higher centroid overlap naturally scales down the dissipation coefficient, which acts as a desirable self-regulating feature that gates off feedback warping ($\eta \to 0$) under high manifold entanglement, protecting the system from representation corruption.

### 2. Taylor Linearization & Higher-Order Bounds (Theorem 3.5)
The closed-loop controller relies on a first-order Taylor series expansion around $\eta = 0$ to derive the dissipation coefficient $A^{(l)}$. 
*   **The Bound (Theorem 3.5)**: To justify using finite step sizes up to $\eta_{\max} = 0.15$, the authors derive a strict analytical bound on the second derivative of the similarity function, yielding a Lagrange remainder bound of $|R_1(\eta)| \le 0.0169$ (about $2.5\%$ of base similarities).
*   **Transient Stability**: They analyze the worst-case unaligned states (abrupt task transitions where $h \cdot \bar{\mu} \approx 0$). They show that although the remainder error increases to $0.11$, the dissipation coefficient $A^{(l)}$ reaches its absolute maximum, ensuring that the update remains dissipative ($\Delta V^{(l)} < 0$) and rapidly pulls the representation back to the task manifold.

### 3. Layer-Identity Approximation & Error Bounds (Theorem 3.2)
The derivation of the Lyapunov difference makes a "Layer-Identity Approximation" ($S(h^{(l-2) \text{ warped}}, \mu_k^{(l-1)}) \approx S(h^{(l-1)}, \mu_k^{(l)})$).
*   **The Bound (Theorem 3.2)**: They prove that the approximation error is strictly bounded by $2 \|r^{(l-1)}\|_2 + \|\mu_k^{(l)} - \mu_k^{(l-1)}\|_2$, where $r^{(l-1)}$ is the residual layer transformation mapping. Under unified centroids, this simplifies to $2 \|r^{(l-1)}\|_2$.
*   **Assumptions**: They transparently discuss two underlying physical assumptions: (1) constructive residual updates (Pre-LN Transformers), and (2) highly transformative middle layers where large non-linear updates can degrade controller precision.
*   **Empirical Confirmation**: They sweep the residual scale parameter $\gamma$ from $0.1$ to $0.9$. They show that when $\gamma$ is small, the Layer-Identity assumption is highly precise, resulting in statistically significant accuracy gains. When $\gamma$ is large, the feedback gains shrink toward the decoupled baseline, empirically confirming their theoretical bound.

### 4. Unification of ECG-Reset and Lyapunov Stability
The authors prove that ECG-Reset is a mathematically necessary precondition for the validity of the candidate Lyapunov function. Without ECG-Reset under routing failures, concentrations converge to uniform noise ($1/K$). This collapses the Lyapunov function to $V_{\text{collapsed}} = 1 - S(h, \bar{\mu}_{\text{uniform}})$, rendering it unable to represent representation-space error relative to the true active task expert and collapsing the dissipation coefficient $A^{(l)}$ to zero. ECG-Reset freezes the concentration updates, preserving the positive semi-definiteness and mathematical integrity of the Lyapunov state space. This is an exceptionally elegant theoretical unification.

### 5. Representation-Agreement State Correction (RASC)
RASC addresses the circular dependency vulnerability of closed-loop ensembling (where a corrupted router biases concentrations, which biases the Lyapunov function, leading to false positive dissipation and warping, which reinforces the corrupted router's confidence). RASC compares feedforward routing with representation-space coordinate tracking to detect conflicts, overriding corrupted feedforward inputs and breaking the circular positive-feedback loop. This is a highly sound, classic dual-loop control architecture.

## Minor Limitations & Assumptions
1.  **Downstream Non-Linear Threshold Effects**: While they address this in their sensitivity sweeps (confirming robust gains under step-threshold activation mappings), they assume a stylized linear mapping for their main accuracy metric.
2.  **Coordinate Sandbox Scale**: The Coordinate Sandbox uses $D=192$ and $K=4$ tasks. Modern high-dimensional models have $D=4096$ or higher and serve dozens of adapters. While the authors present a very compelling real-world pilot study on LLaMA-3-8B validating that high-dimensional centroids are indeed orthogonal ($\le 0.08$ similarity) and the Dissipation Guard behaves as predicted, the main quantitative results are still evaluated in the sandbox.
