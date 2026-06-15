# 3. Soundness and Methodology

## Clarity of Description and Appropriateness of Methods
The paper is exceptionally well-written, structured, and mathematically dense. Modeling continuous-depth model ensembling as physical kinetics governed by discretized ODEs, and stabilizing the active representation feedback using closed-loop control theory, is highly original and methodologically appropriate. The connections drawn between deep network layers and time-varying dynamical systems are mathematically elegant and natural.

---

## Technical Flaws and Mathematical Limitations (Theorist Critique)

As a theory-minded reviewer, several subtle mathematical assumptions, limitations, and minor technical flaws in the proofs need to be addressed:

### 1. Classical Lyapunov Definition vs. Zero-Error Incompatibility
In classical control theory, a candidate Lyapunov function $V(x)$ must be positive definite with respect to a unique equilibrium point $x^*$ (i.e., $V(x^*) = 0$ and $V(x) > 0$ for all $x \ne x^*$), and its derivative or difference must be negative definite to guarantee asymptotic stability. 
In L-ARC, under multi-expert activation (Eq. 11), there is a physical lower bound:
$$V(C^{(l)}, h^{(l-1) \text{ warped}}) \ge \sum_{k=1}^K C_k^{(l)} - \max_{k} C_k^{(l)} > 0$$
Because of this, the candidate Lyapunov function cannot reach zero. Thus, the closed-loop system cannot be shown to be "asymptotically stable" to a unique zero-error point in the classical sense. Rather, the system is technically **dissipative to a bounded region** or **bounded-input bounded-state (BIBS) stable**. While the paper acknowledges this in the "Zero-Error Incompatibility" Remark (Remark 3.1), calling $V$ a "Lyapunov function" is technically a slight abuse of notation; it behaves more as a *potential function* or *cost function* used to derive a dissipative gradient-like control action. The authors should clarify this distinction to align strictly with classical system theory.

### 2. Unlisted Condition in Theorem 3.2 (Layer-Identity Error Bound)
The proof of Theorem 3.2 (Layer-Identity Error Bound) relies on the step:
$$\| h^{(l-1)} - h_w \|_2 \le 2 \| y - h_w \|_2 = 2 \| r^{(l-1)}(h_w) \|_2$$
This inequality is derived from the projection property:
$$\| u/\|u\|_2 - v \|_2 \le \frac{2 \|u - v\|_2}{\|u\|_2}$$
where $v = h_w$ is unit-norm, and $u = y = h_w + r^{(l-1)}(h_w)$. 
This bound only holds if $\| y \|_2 \ge 1$. The authors justify this in the text by stating that "residual block updates are designed to be constructive or near-orthogonal... meaning the inner product $h_w \cdot r^{(l-1)}(h_w) \ge 0$, which yields $\|y\|_2^2 \ge 1$."
While this is a reasonable heuristic for stable networks, there is no formal guarantee that deep neural network residual blocks are constructive for all inputs, tokens, and layers. In practice, layers can be highly contractive or destructive ($h_w \cdot r(h_w) < -0.5$), which would violate the $\|y\|_2 \ge 1$ condition and break the bound. 
* **Critique:** The constructive update condition $\|h_w + r^{(l-1)}(h_w)\|_2 \ge 1$ must be explicitly listed as a **conditional assumption** in the statement of Theorem 3.2 rather than treated as an unconditional general property of neural networks.

### 3. Parameter Evaluation Error in the Proof of Theorem 3.4
In Section 3.4 (Taylor Linearization Validity), Theorem 3.4 derives a Lagrange remainder bound to prove that Taylor linearization errors are small at finite step sizes ($\eta \le 0.15$). The proof bounds the second derivative $|g''(\xi)| \le 1.50$ under typical parameters $a \approx -0.3, b \approx 0.6$.
In the calculation, the authors substitute:
$$|g''(\xi)| \le 2 \frac{1.0 \cdot 0.21}{0.857} + \frac{0.6 \cdot 1.15}{0.857} + 3 \frac{1.15 \cdot 0.21^2}{0.774}$$
Here, the value $0.21$ is substituted for $|a + b\xi|$. However, since $a \approx -0.3$ and $b \approx 0.6$, the term is:
$$|a + b\xi| = |-0.3 + 0.6\xi|$$
Since $\xi \in [0, 0.15]$, the absolute value of this term ranges from $|-0.3| = 0.3$ (at $\xi=0$) to $|-0.3 + 0.09| = 0.21$ (at $\xi=0.15$). 
The maximum absolute value of this term over the interval is $0.30$, not $0.21$. Evaluating the Lagrange remainder using the endpoint $\xi = 0.15$ rather than the maximum over the entire interval $[0, 0.15]$ is a technical mathematical error.
* **Correction:** If we evaluate $|g''(\xi)|$ using the correct maximum values over the interval:
  - $|a + b\xi|_{\max} = 0.30$
  - $n(\xi)_{\min} \approx 0.961$ (at $\xi=0.15$)
  - $|y(\xi) \cdot \mu_k| \le n(\xi)$
  Re-summing the maximums of the individual terms:
  - First term: $2 \times 1.0 \times 0.30 / 1^3 = 0.60$ (at $\xi=0$)
  - Second term: $0.6 \times 0.961 / 0.961^3 = 0.6 / 0.961^2 \approx 0.65$ (at $\xi=0.15$)
  - Third term: $3 \times n(\xi) (a + b\xi)^2 / n(\xi)^5 \le 3 (a + b\xi)^2 / n(\xi)^4 \approx 3 \times 0.09 / 1^4 = 0.27$ (at $\xi=0$)
  The sum of individual maximums is $0.60 + 0.65 + 0.27 = 1.52$. This slightly violates the $|g''(\xi)| \le 1.50$ bound and yields a remainder bound of $|R_1(\eta)| \le 0.0171$ instead of $0.0169$. The authors must correct this parameter evaluation to maintain absolute mathematical rigor.

### 4. Initialization of Kinetics Concentration States
The paper details the Euler discretization update of the concentrations $C_k^{(l)}$ starting at $l=4$ (Eq. 5), but does not state how the initial concentration state vector $C_k^{(3)}$ is initialized. Are they set to uniform ($1/K = 0.25$), zero, or initialized using early-stage routing collision rates $k_k^{(3)}$? The initialization strategy directly impacts the kinetics propagation lag, and should be explicitly defined.

### 5. Division by Zero Edge Case in Weight Normalization
In Eq. (7), ensembling weights are normalized as:
$$\alpha_k^{(l)} = \frac{C_k^{(l)}}{\sum_{j=1}^K C_j^{(l)}}$$
If all concentration states $C_j^{(l)}$ collapse to $0$ (which can happen under heavy back-reaction decay $k_{\text{decay}}$ combined with $[\cdot]_0^1$ clamping), this formula suffers from a division by zero. A mathematical safety guard (such as $\sum C_j + \epsilon$) should be explicitly defined.

---

## Reproducibility
The paper is highly reproducible. All equations, derivations, and hyperparameters are clearly stated. The authors provide specific details about the 14-layer Coordinate Sandbox (ICS), enabling straightforward reproduction of the results. The inclusion of precise hyperparameter values ($\tau = 0.01$, $\Delta t = 1.5$, $k_{\text{decay}} = 0.3$, $\eta_{\max} = 0.15$, $\gamma = 1.0$, $\theta_G = 0.04$, $\theta_H = 0.95$) is highly commendable and ensures high-fidelity reproducibility.
