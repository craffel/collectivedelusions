# Peer Review of "PID-Merge: Closed-Loop Discrete-Time PID Control for Stateful Multi-Tenant Adapter Serving"

## 1. Summary of the Paper
This paper proposes **PID-Merge**, a closed-loop, discrete-time Proportional-Integral-Derivative (PID) controlled stateful routing framework designed for dynamic, test-time model ensembling of task-specific expert adapters (e.g., LoRA) in deep multi-tenant networks. 
Dynamic ensembling involves blending expert outputs on-the-fly to handle sequential queries. Stateless dynamic routers (e.g., SABLE) suffer from a "routing jitter paradox," where layer-by-layer representation noise causes weights to oscillate wildly, degrading robustness. Prior stateful routers (e.g., ChemMerge, Momentum-Merge) smooth these trajectories but suffer from "inertial drag" (phase/group delay) during sudden task transitions, which collapses accuracy under highly volatile heterogeneous workloads.
PID-Merge treats the raw early-layer similarity weights as the setpoint and active ensembling coefficients as the plant output. By feeding back the tracking error utilizing Proportional, Integral, and Derivative gains, PID-Merge achieves smooth ensembling trajectories while using the Derivative (D) term to measure tracking error acceleration and eliminate phase lag. The framework includes systems-level enhancements such as:
1. **Scaled Logit Mean-Centering** to prevent absolute value overflow and float overflow while preserving translation invariance under task-specific temperatures.
2. **Conditional Integration (Clamping)** to prevent saturation-induced transition lag in deep networks (integrator windup).
3. **Prefill-Locked Routing** to guarantee key-value (KV) cache coherence and eliminate decoding latency overhead.
The method is evaluated on a simulated sandbox (ICS) and validated physically on a 12-layer GPT-2 Small backbone on an NVIDIA A100 GPU routing three actual adapters.

---

## 2. Key Strengths
- **Original Interdisciplinary Formulation:** Bridging discrete-time classical control theory and parameter-efficient fine-tuning (PEFT) serving is a highly creative, elegant, and original contribution. It replaces fragile chemical metaphors with a rigorous, closed-loop feedback design.
- **Excellent Transition Responsiveness:** By leveraging the Derivative (D) term to measure tracking error acceleration, the controller anticipates task transitions and provides an anticipatory boost that drives weights to the target within 2--3 layers, effectively eliminating "inertial drag" on volatile heterogeneous streams.
- **Impressive Systems Integration and Hardware Validation:** The authors provide a practical, hardware-ready blueprint (such as Prefill-Locked routing and fused Triton kernel designs) and physically validate it on an NVIDIA A100 GPU. They show that PID-Merge is $40\times$ faster than SOTA ChemMerge, adding a negligible latency overhead of just $0.012\text{ ms}$ while slashing depth-wise layer-to-layer jitter by 73%.
- **Clarity of Presentation:** The paper is exceptionally well-structured, logical, and easy to read.

---

## 3. Major Weaknesses and Theoretical Concerns

Despite its clear merits, a rigorous theoretical and control-theoretic analysis reveals several fundamental mathematical and methodological flaws that must be addressed before this work can be considered technically sound:

### A. Critical Mathematical Flaw in the Analytical Stability Analysis (Jury's Criterion Omission)
In Appendix B (Section \ref{sec:appendix_stability}), the authors present a discrete-time linearized stability analysis of the closed-loop PID controller and attempt to prove stability using **Jury's Stability Criterion** for the 3rd-order characteristic polynomial:
$$A(z) = z^3 + \left[ K_s(K_p + K_i + K_d) - 1 \right] z^2 - K_s(K_p + 2K_d)z + K_s K_d = 0$$
The authors state that the system is stable if and only if the following three conditions are met (which they translate into bounds on the gains $K_p$, $K_i$, and $K_d$):
1. $A(1) > 0 \implies K_s K_i > 0$
2. $A(-1) < 0 \implies K_s(2K_p + K_i + 4K_d) < 2$
3. $|a_3| < 1 \implies K_s K_d < 1 \implies K_d < \frac{1}{K_s}$ (since $a_0 = 1$)

**The Flaw:** For a discrete-time third-order characteristic equation, satisfying these three conditions is **insufficient** to guarantee stability. Jury's stability test for a cubic polynomial requires exactly **four** necessary and sufficient conditions. The authors completely omit the fourth, indispensable condition:
$$|a_0^2 - a_3^2| > |a_0 a_2 - a_3 a_1|$$
Since the polynomial is monic ($a_0 = 1$) and $|a_3| < 1$, this condition simplifies to:
$$1 - a_3^2 > |a_2 - a_3 a_1|$$
Plugging in the paper's defined coefficients, the omitted fourth condition is:
$$1 - (K_s K_d)^2 > \left| -K_s(K_p + 2K_d) - K_s K_d \left( K_s(K_p + K_i + K_d) - 1 \right) \right|$$

**The Counterexample:** We can construct a concrete cubic polynomial that satisfies all three stability conditions listed in Appendix B, yet is highly unstable (with multiple poles outside the unit circle).
Let $A(z) = z^3 - 2.95z^2 + 2.725z - 0.75 = 0$.
The roots of this polynomial are:
$$z_1 = 0.5, \quad z_2 = 1.2, \quad z_3 = 1.25$$
Since two of the roots lie outside the unit circle ($|z_2| > 1$ and $|z_3| > 1$), this system is highly unstable.
However, checking the polynomial against the paper's three conditions:
- **Condition 1 (A(1) > 0):** $A(1) = 1 - 2.95 + 2.725 - 0.75 = 0.025 > 0$ *(Satisfied)*
- **Condition 2 (A(-1) < 0):** $A(-1) = -1 - 2.95 - 2.725 - 0.75 = -7.425 < 0$ *(Satisfied)*
- **Condition 3 (|a_3| < 1):** $|a_3| = |-0.75| = 0.75 < 1$ *(Satisfied)*

All three analytical stability conditions listed in Appendix B are fully satisfied, yet the system is unstable! Only the omitted fourth condition is violated:
$$1 - a_3^2 = 1 - (-0.75)^2 = 0.4375$$
$$|a_2 - a_3 a_1| = |2.725 - (-0.75 \times -2.95)| = |2.725 - 2.2125| = 0.5125$$
Since $0.4375 < 0.5125$, the fourth condition fails.

Because of this omission, the analytical stability bounds derived in the paper are incorrect, and the corresponding stability penalty $\mathcal{L}_{\text{stab}}$ (Equation 12) implemented in their optimizer is **incomplete**. It is highly probable that the optimizer converges to unstable gain regions under noisy calibration data, causing unpredicted tracking divergence.

### B. Invalid Linear Time-Invariant (LTI) Assumption (State-Dependent Plant Gain)
To apply discrete-time transfer functions $P(z)$ and $C(z)$ and perform Jury's stability analysis, the paper models the plant gain $K_s$ as a constant.
**The Flaw:** By definition, the plant gain is the derivative of the multi-temperature Softmax:
$$K_s^{(l)} = \frac{\partial \alpha_k^{(l)}}{\partial s_k^{(l)}} = \frac{1}{\tau_k} \alpha_k^{(l)} (1 - \alpha_k^{(l)})$$
Because the active ensembling weights $\alpha_k$ change dynamically layer-by-layer (specifically transitioning from the initial uniform state $1/K$ at Layer 3 to a near-one-hot target at deeper layers), $K_s^{(l)}$ is **state-dependent** and **layer-varying**.
Thus, the closed-loop system is highly non-linear and time-varying. Applying LTI tools (like $z$-transforms, Jury's stability criterion, and transfer functions) is mathematically invalid. It can only serve as a highly local approximation around a static operating point. Establishing true stability guarantees for this system requires non-linear control theory (e.g., Lyapunov's direct method, contraction analysis, or circle/Popov criteria for sector-bounded non-linearities).

### C. Gradient Vanishing in Unconstrained Temperature Parameters
The authors claim that the exponential parameterization $\tau_k = e^{w_k} + \tau_{\min}$ of the task-specific temperatures allows direct unconstrained gradient descent on $w_k \in \mathbb{R}$ without causing gradient explosion or scale-shifting issues.
**The Flaw:** While it prevents divisions by zero, this parameterization is prone to severe gradient vanishing issues:
- **High-Temperature Regime:** If $w_k$ becomes moderately positive, $\tau_k = e^{w_k}$ grows exponentially. This drives the Softmax to become extremely uniform and flat. Since the active inputs to the Softmax are $s_k^{(l)} / \tau_k$, an exponentially large $\tau_k$ drives the activations to zero, and the gradient of the loss with respect to $s_k^{(l)}$ vanishes because of the $\frac{1}{\tau_k}$ scale.
- **Low-Temperature Regime:** If $w_k$ becomes negative, $\tau_k \to \tau_{\min} = 0.01$. This drives the Softmax to behave like a hard argmax. The gradients of an argmax are zero almost everywhere, which also leads to gradient vanishing during backpropagation.
Therefore, the claim that this parameterization "naturally self-bounds" and is free of scale-shifting or gradient issues is theoretically incorrect.

### D. Interpretability and Non-Monotonicity of Multi-Temperature Softmax
Because the temperatures $\tau_k$ are expert-specific, the mapping from the unnormalized controller state $s_k^{(l)}$ to the final weight $\alpha_k^{(l)}$ is **not strictly monotonic** or rank-preserving across experts.
- An expert with a lower unnormalized tracking state $s_i^{(l)}$ but a lower temperature $\tau_i$ can receive a higher active ensembling weight $\alpha_i^{(l)}$ than an expert with a larger state $s_j^{(l)}$ but higher temperature $\tau_j$.
This non-monotonicity undermines the physical control intuition of the state variables $s_k^{(l)}$, making them uninterpretable. An expert can dominate the ensembling mix even when its tracking error and state trajectory are smaller, which violates basic control design principles.

### E. Limitations of Simulated Sandbox and Small-Scale Validation
- **ICS Sandbox Constraints:** As acknowledged, the simulated sandbox injects representation noise only at the initial boundary layer. Consequently, stateless SABLE transitions instantly and does not oscillate in simulation, making the sandbox results appear artificially stable for stateless methods.
- **Lack of Modern Large-Backbone Evaluation:** Physical hardware validation is restricted to GPT-2 Small (124M parameters) routing 3 adapters. Evaluating the method on a larger multi-billion parameter model (e.g., LLaMA-3 8B) would significantly strengthen the empirical claims, as systems-level overhead and integrator windup are much more pronounced at scale.

---

## 4. Questions for the Authors
1. **Omission of the 4th Jury Condition:** Why was the fourth necessary and sufficient condition of Jury's Stability Criterion omitted from the analytical proof in Appendix B? How do you guarantee stability when your current stability penalty $\mathcal{L}_{\text{stab}}$ does not enforce this fourth condition?
2. **State-Dependent and Time-Varying Plant Gain:** Since the plant gain $K_s^{(l)} = \frac{1}{\tau_k} \alpha_k^{(l)} (1 - \alpha_k^{(l)})$ is layer-varying and state-dependent, how do you justify the validity of a linear time-invariant (LTI) stability analysis? Have you considered using non-linear control analysis, such as Lyapunov's direct method or contraction mapping?
3. **Vanishing Gradients in Calibration:** Did you encounter vanishing gradients when optimizing the unconstrained task-specific log-temperatures $w_k$ during calibration, especially when they became very large or very negative?
4. **Rank Reversal and Interpretability:** How do you address the interpretability concerns associated with the non-monotonicity of the multi-temperature Softmax, where an expert with a smaller unnormalized controller state can receive a larger active ensembling weight due to temperature differences?

---

## 5. Detailed Comments/Minor Points
- In Equation 12, the stability penalty is defined based on maximum ambiguity ($K_{s,\max} = \frac{1}{4\tau_k}$). If the stability proof is updated to include the 4th condition, this penalty function should be restructured to incorporate the simplified fourth inequality.
- In Section 4.1, the paper states that the sandbox parameters were calibrated to $\text{sigmas\_scale} = 0.1803$. It would be helpful to provide a brief explanation of how this calibration was performed and why this specific value was chosen.

---

## 6. Overall Recommendation
**Rating: 3: Weak Reject**

**Justification:**
This submission has clear merits, including a highly creative formulation bridging classical control and PEFT serving, exceptional transition responsiveness under heterogeneous workloads, and solid physical GPU validations. 
However, as a control-theoretic framework, its core theoretical contribution is compromised by a fundamental mathematical omission in the stability analysis: the Jury stability proof completely omits the fourth necessary and sufficient condition for 3rd-order discrete systems, and the corresponding optimization penalty $\mathcal{L}_{\text{stab}}$ is mathematically incomplete. Additionally, the LTI assumption for a state-dependent, layer-varying plant gain is theoretically invalid and represents a severe oversimplification. 
Because the paper's central theoretical guarantees require revisions before they can be meaningfully built upon, I recommend a Weak Reject. I strongly encourage the authors to correct the Jury stability proof, expand the stability penalty to include the fourth condition, and honestly address the state-dependent nature of the plant in their revision. Doing so would elevate this highly promising work to a technically flawless, high-impact paper.
