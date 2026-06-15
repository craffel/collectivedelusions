# 3. Soundness and Methodology

This section provides a rigorous theoretical and methodological evaluation of **PID-Merge**, focusing on its clarity, appropriateness of mathematical modeling, potential technical and mathematical flaws, and reproducibility.

## Methodological Strengths
The mathematical formulation of the closed-loop tracking error, the discrete-time velocity update, and the systems-level integrations (such as logit mean-centering, conditional clamping anti-windup, and prefill-locked routing) are highly detailed and clearly written.

---

## Critical Theoretical and Mathematical Flaws

While the paper is well-reasoned, a deep control-theoretic audit reveals several major theoretical and mathematical flaws that undermine its analytical guarantees.

### 1. Incomplete and Flawed Analytical Stability Analysis (Jury's Criterion Omission)
In Appendix B (Section \ref{sec:appendix_stability}), the authors attempt to prove the stability of the closed-loop system using **Jury's Stability Criterion** for a discrete-time third-order characteristic polynomial:
$$A(z) = z^3 + \left[ K_s(K_p + K_i + K_d) - 1 \right] z^2 - K_s(K_p + 2K_d)z + K_s K_d = 0$$
The authors assert that the system is stable if and only if three conditions are satisfied (which they translate into bounds on the gains $K_p$, $K_i$, and $K_d$):
1. $A(1) > 0 \implies K_s K_i > 0$
2. $A(-1) < 0 \implies K_s(2K_p + K_i + 4K_d) < 2$
3. $|a_3| < 1 \implies K_s K_d < 1 \implies K_d < \frac{1}{K_s}$

**The Flaw:** For any third-order polynomial, satisfying these three conditions is **insufficient** to guarantee stability. Jury's criterion for a cubic polynomial requires exactly **four** necessary and sufficient conditions. The authors completely omit the fourth, indispensable condition:
$$|a_0^2 - a_3^2| > |a_0 a_2 - a_3 a_1|$$
Since the polynomial is monic ($a_0 = 1$) and $|a_3| < 1$, this condition simplifies to:
$$1 - a_3^2 > |a_2 - a_3 a_1|$$
Plugging in the paper's defined coefficients, this omitted condition is:
$$1 - (K_s K_d)^2 > \left| -K_s(K_p + 2K_d) - K_s K_d \left( K_s(K_p + K_i + K_d) - 1 \right) \right|$$

**The Counterexample:** We can construct a concrete mathematical counterexample of a cubic polynomial that satisfies all three conditions listed in Appendix B, but is highly unstable.
Let $A(z) = z^3 - 2.95z^2 + 2.725z - 0.75 = 0$.
The roots of this polynomial are:
$$z_1 = 0.5, \quad z_2 = 1.2, \quad z_3 = 1.25$$
Since two of the poles lie outside the unit circle ($|z_2| > 1$ and $|z_3| > 1$), this system is highly unstable and will oscillate/diverge.
However, checking the polynomial against the paper's three conditions:
- **Condition 1 (A(1) > 0):** $A(1) = 1 - 2.95 + 2.725 - 0.75 = 0.025 > 0$ *(Satisfied)*
- **Condition 2 (A(-1) < 0):** $A(-1) = -1 - 2.95 - 2.725 - 0.75 = -7.425 < 0$ *(Satisfied)*
- **Condition 3 (|a_3| < 1):** $|a_3| = |-0.75| = 0.75 < 1$ *(Satisfied)*

All three analytical stability conditions listed in the paper are fully satisfied, yet the system is unstable! Only the omitted fourth condition is violated:
$$1 - a_3^2 = 1 - (-0.75)^2 = 0.4375$$
$$|a_2 - a_3 a_1| = |2.725 - (-0.75 \times -2.95)| = |2.725 - 2.2125| = 0.5125$$
Since $0.4375 < 0.5125$, the fourth condition fails.

Because of this omission, the analytical stability bounds derived in the paper are incorrect, and the corresponding stability penalty $\mathcal{L}_{\text{stab}}$ (Equation 12) implemented in their optimizer is **incomplete**. It is highly probable that the optimizer converges to unstable gain regions under noisy calibration data, causing unpredicted tracking divergence.

### 2. Invalid Linear Time-Invariant (LTI) Assumption (State-Dependent Plant Gain)
To apply discrete-time transfer functions $P(z)$ and $C(z)$ and perform Jury's stability analysis, the paper models the plant gain $K_s$ as a constant.
**The Flaw:** By definition, the plant gain is the derivative of the multi-temperature Softmax:
$$K_s^{(l)} = \frac{\partial \alpha_k^{(l)}}{\partial s_k^{(l)}} = \frac{1}{\tau_k} \alpha_k^{(l)} (1 - \alpha_k^{(l)})$$
Because the active ensembling weights $\alpha_k$ change dynamically layer-by-layer (specifically transitioning from the initial uniform state $1/K$ at Layer 3 to a near-one-hot target at deeper layers), $K_s^{(l)}$ is **state-dependent** and **layer-varying**.
Thus, the closed-loop system is highly non-linear and time-varying. Applying LTI tools (like $z$-transforms, Jury's stability criterion, and transfer functions) is mathematically invalid. It can only serve as a highly local approximation around a static operating point. Establishing true stability guarantees for this system requires non-linear control theory (e.g., Lyapunov's direct method, contraction analysis, or circle/Popov criteria for sector-bounded non-linearities).

### 3. Gradient Vanishing/Explosion in Unconstrained Temperature Parameters
The authors claim that the exponential parameterization $\tau_k = e^{w_k} + \tau_{\min}$ of the task-specific temperatures allows direct unconstrained gradient descent on $w_k \in \mathbb{R}$ without causing gradient explosion or scale-shifting issues.
**The Flaw:** While it prevents divisions by zero, this parameterization is prone to severe gradient vanishing issues:
- **High-Temperature Regime:** If $w_k$ becomes moderately positive, $\tau_k = e^{w_k}$ grows exponentially. This drives the Softmax to become extremely uniform and flat. Since the active inputs to the Softmax are $s_k^{(l)} / \tau_k$, an exponentially large $\tau_k$ drives the activations to zero, and the gradient of the loss with respect to $s_k^{(l)}$ vanishes because of the $\frac{1}{\tau_k}$ scale.
- **Low-Temperature Regime:** If $w_k$ becomes negative, $\tau_k \to \tau_{\min} = 0.01$. This drives the Softmax to behave like a hard argmax. The gradients of an argmax are zero almost everywhere, which also leads to gradient vanishing during backpropagation.
Therefore, the claim that this parameterization "naturally self-bounds" and is free of scale-shifting or gradient issues is theoretically incorrect.

### 4. Interpretability and Non-Monotonicity of Multi-Temperature Softmax
Because the temperatures $\tau_k$ are expert-specific, the mapping from the unnormalized controller state $s_k^{(l)}$ to the final weight $\alpha_k^{(l)}$ is **not strictly monotonic** or rank-preserving across experts.
- An expert with a lower unnormalized tracking state $s_i^{(l)}$ but a lower temperature $\tau_i$ can receive a higher active ensembling weight $\alpha_i^{(l)}$ than an expert with a larger state $s_j^{(l)}$ but higher temperature $\tau_j$.
This non-monotonicity undermines the physical control intuition of the state variables $s_k^{(l)}$, making them uninterpretable. An expert can dominate the ensembling mix even when its tracking error and state trajectory are smaller, which violates basic control design principles.

---

## Reproducibility and Clarity
- **Clarity:** The text is excellently written, structured, and presented. The diagrams and tables are highly professional.
- **Reproducibility:** The paper provides a PyTorch implementation blueprint in Appendix K, listing all hyper-parameters, configurations, and baseline descriptions. However, because of the omission of the fourth Jury condition and the state-dependent nature of the plant, reproducing the "calibrated mode" with the stability penalty $\mathcal{L}_{\text{stab}}$ might lead to unstable controllers under different random seeds or initialization ranges.
