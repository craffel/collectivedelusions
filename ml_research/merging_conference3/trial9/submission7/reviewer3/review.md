# Peer Review

## 1. Summary of the Paper
This paper addresses the dual challenges of **routing volatility (jitter)** and **cascading representational drift** in dynamic model serving and ensembling on resource-constrained edge devices. 

Stateful continuous-time ensembling methods (e.g., ChemMerge) smooth routing weight trajectories across depth using discretized ordinary differential equations (ODEs), but they rely on open-loop, heuristic constant feedback step sizes ($\eta$) to warp hidden representations toward pre-computed early-stage centroids. Under mixed, heterogeneous workloads, this open-loop constant feedback pulls highly refined late-layer activations back toward noisy, early-stage coordinates—a phenomenon the authors call **representational backward-shift**—which degrades performance monotonically as $\eta$ increases.

To resolve this, the paper introduces **Lyapunov-Stable Active Representation Coupling (L-ARC)**, a training-free closed-loop control framework that provides formal control-theoretic guarantees for continuous-depth feature propagation. L-ARC models the representation similarity error as a system-level candidate Lyapunov function and analytically derives a local **Dissipation Guard** to compute sample-specific, layer-specific feedback rates ($\eta^{(l)}$) on-the-fly, ensuring that representation warping is strictly error-decreasing (dissipative).

Additionally, the authors introduce:
1. **Entropy-Gated Concentration Gating (ECG-Reset):** A state-space shield that monitors routing entropy to dynamically freeze continuous ODE kinetics during routing dropouts or failures.
2. **Entropy-Triggered Gating (ET-L-ARC):** A control-theoretic optimization that evaluates the Dissipation Guard only under moderate routing uncertainty, collapsing latency overhead under clean workloads to near-zero ($0.06$ ms per sample).
3. **Representation-Agreement State Correction (RASC):** A dual-loop control mechanism that compares feedforward routing selections with feedback representation-space coordinates, overriding corrupted, systematically biased router outputs to resolve "state-locking" failures.

---

## 2. Strengths
* **Rigorous Control-Theoretic Grounding:** Rather than proposing ad-hoc empirical heuristics, the paper is built from the ground up on classical control theory and dynamical systems principles. The formulation of the representation warping step as a closed-loop control action and the derivation of the Dissipation Guard from a system-level candidate Lyapunov function represent a highly sophisticated and mathematically elegant contribution.
* **Brilliant Solutions for Real-World serving Failures:**
  * **ECG-Reset** elegantly resolves kinetics memory corruption under transient dropouts by dynamically setting the integration step size to zero ($\Delta t = 0$), freezing the state space.
  * **RASC** successfully combats systematic router bias by comparing feedforward predictions with unbiased representation-space coordinate tracking, completely neutralizing state-locking failures.
* **Outstanding Scientific Transparency:** The authors report with complete scientific honesty that active representation warping is statistically redundant under clean workloads ($p = 0.0969$) and has a latency overhead. This level of transparency is highly commendable, as it focuses the utility of the active controller strictly where it is mathematically and physically needed (faulty and biased workloads).
* **Hardware-Minded Computational Efficiency:** By introducing Entropy-Triggered Gating (ET-L-ARC), the authors successfully collapse the latency overhead to just $99.85\%$ ($0.06$ ms absolute overhead per sample) under clean serving, demonstrating deep awareness of edge-hardware deployment constraints.
* **Real-World LLM Validation:** The inclusion of a small-scale pilot study on LLaMA-3-8B is excellent, validating the core high-dimensional geometric assumptions (centroid orthogonality, perplexity recoverability, and classification accuracy gains) in a real LLM setting and successfully bridging the gap between sandbox simulations and full-scale architectures.

---

## 3. Weaknesses

While the paper is of exceptionally high caliber, a rigorous analysis of the mathematical proofs and assumptions reveals several subtle nuances and minor technical flaws that must be addressed:

### 1. Classical Lyapunov Definition vs. Zero-Error Incompatibility
In classical control theory, a candidate Lyapunov function $V(x)$ must be positive definite with respect to a unique equilibrium point $x^*$ (meaning $V(x^*) = 0$ and $V(x) > 0$ for all $x \ne x^*$), and its derivative or difference must be negative definite. 
In L-ARC, under multi-expert activation, the authors prove a physical lower bound (Eq. 11):
$$V(C^{(l)}, h^{(l-1) \text{ warped}}) \ge \sum_{k=1}^K C_k^{(l)} - \max_{k} C_k^{(l)} > 0$$
This lower bound prevents the candidate Lyapunov function from reaching zero. Thus, the closed-loop system is not "asymptotically stable" to a unique zero-error point in the classical system-theoretic sense. Rather, the system is technically **dissipative to a bounded region** (or bounded-input bounded-state stable). While the paper briefly mentions this in the "Zero-Error Incompatibility" Remark (Remark 3.1), calling $V$ a "Lyapunov function" is technically a slight abuse of notation; it behaves more as a *potential/cost function* used to derive a dissipative gradient-like control action. This distinction should be clarified to align strictly with classical system theory.

### 2. Unlisted Condition in Theorem 3.2 (Layer-Identity Error Bound)
The proof of Theorem 3.2 (Layer-Identity Error Bound) relies on the step:
$$\| h^{(l-1)} - h_w \|_2 \le 2 \| y - h_w \|_2 = 2 \| r^{(l-1)}(h_w) \|_2$$
This inequality is derived from the standard projection property:
$$\| u/\|u\|_2 - v \|_2 \le \frac{2 \|u - v\|_2}{\|u\|_2}$$
where $v = h_w$ is unit-norm, and $u = y = h_w + r^{(l-1)}(h_w)$. 
This bound only holds if $\| y \|_2 \ge 1$. The authors justify this by stating that "residual block updates are designed to be constructive... meaning the inner product $h_w \cdot r^{(l-1)}(h_w) \ge 0$, which yields $\|y\|_2^2 \ge 1$."
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

## 4. Questions and Actionable Feedback for the Authors
1. **Clarify Lyapunov Framework vs. Dissipative Region:** Please add a brief discussion clarifying that due to multi-expert ensembling lower bounds, $V$ behaves as a potential/cost function rather than a classical Lyapunov function, and that the closed-loop system is dissipative to a bounded region rather than asymptotically stable to a unique point.
2. **Explicitly Conditionalize Theorem 3.2:** Please update the statement of Theorem 3.2 to include the constructive update assumption ($\|h_w + r^{(l-1)}(h_w)\|_2 \ge 1$) as an explicit condition.
3. **Correct Theorem 3.4 Parameter Evaluation:** Please correct the endpoint substitution error in the proof of Theorem 3.4. Evaluate $|a + b\xi|$ using its maximum over the interval ($0.30$) rather than the endpoint ($0.21$). Update the bound to $|g''(\xi)| \le 1.52$ and the Lagrange remainder error to $|R_1(\eta)| \le 0.0171$ accordingly.
4. **Define Initial States:** Please explicitly state how the initial concentration state vector $C_k^{(3)}$ is initialized in the methodology section.
5. **Incorporate Weight Normalization Stabilizer:** Please update Eq. 7 to include a small stabilization constant $\epsilon$ (e.g., $\sum C_j + \epsilon$) to formally guard against division-by-zero failures.
6. **Elaborate on Accuracy vs. Representation Distortion Trade-off:** In Table 2 (Setting C), stateless **SPS-ZCA SOTA** achieves a significantly superior final-layer Semantic Similarity (**0.8270** vs. L-ARC's **0.7813**), despite L-ARC having higher classification accuracy. Please expand the discussion around this trade-off to guide edge practitioners on whether downstream accuracy or activation-space semantic preservation is their primary goal.

---

## 5. Ratings and Overall Recommendation

* **Soundness:** **Excellent** (The mathematical framework is highly rigorous and correct, pending the minor proof corrections and explicit conditional assumptions requested above).
* **Presentation:** **Excellent** (The writing is exceptionally clear, logical, and structured. Figures and tables are of publication-ready quality).
* **Significance:** **Excellent** (The work provides a mathematically certified, training-free approach to stabilizing dynamic serving, which is critical for edge deployments).
* **Originality:** **Excellent** (Formulating active feedback ensembling as a closed-loop control problem is highly novel, and RASC is a beautiful dual-loop self-correction mechanism).

### Overall Recommendation: **5: Accept**
This is an outstanding, exceptionally high-quality paper that bridges classical control theory and deep learning serving with remarkable mathematical rigor and scientific honesty. Formulating active representation coupling as a closed-loop dissipative system represents a significant paradigm shift. The proposed ECG-Reset and RASC mechanisms provide elegant, control-theoretic solutions to real-world edge serving failures (sensor dropouts and persistent router bias). Once the minor mathematical typos in the proofs are corrected and the conditional assumptions are explicitly listed, this paper will be a stellar and highly cited addition to the literature.
