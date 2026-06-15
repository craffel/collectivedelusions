# Evaluation Checklist: Soundness, Theoretical Claims, and Methodology

## 1. Technical & Theoretical Soundness
The mathematical framework presented in the paper is exceptionally solid, elegant, and technically sound. The authors have constructed a rigorous state-space formulation of modular deep network serving, drawing from variational Bayesian inference and systems control engineering.

### A. Mathematical Derivation of Variational Free Energy ($\mathcal{F}_t$)
- Under the Gaussian posterior $q(\mathbf{s}_t) = \mathcal{N}(\mathbf{s}_t; \mathbf{\mu}_t, \mathbf{\Sigma}_t)$ and linear-Gaussian process/observation models, the step-by-step expansion of $\mathcal{F}_t$ (detailed in Appendix A) is mathematically correct.
- **Posterior Mean Substitution:** To avoid propagating high-dimensional uncertainty recursively through time (which is computationally intractable at test-time), the authors substitute the previous state $\mathbf{s}_{t-1}$ with its converged posterior mean estimate $\mathbf{\mu}_{t-1}$. This is a standard and highly practical control-theoretic simplification (similar to the deterministic step in extended Kalman filters).
- **The Resulting Objective:** The derivation correctly isolates terms depending on the optimization variable $\mathbf{\mu}_t$, arriving at:
  $$\mathcal{F}_t(\mathbf{\mu}_t) \propto \frac{1}{2} (\mathbf{e}_t - \mathbf{W}\mathbf{\mu}_t)^T \mathbf{\Pi}_e (\mathbf{e}_t - \mathbf{W}\mathbf{\mu}_t) + \frac{1}{2} (\mathbf{\mu}_t - \mathbf{A}\mathbf{\mu}_{t-1})^T \mathbf{\Pi}_s (\mathbf{\mu}_t - \mathbf{A}\mathbf{\mu}_{t-1})$$
  This is a strictly convex, quadratic objective in $\mathbf{\mu}_t$.

### B. Analytical Closed-Form Solution
- Taking the matrix derivative with respect to $\mathbf{\mu}_t$ and setting it to $\mathbf{0}$ yields:
  $$\mathbf{H}\mathbf{\mu}_t^* = \mathbf{b}_t \implies \mathbf{\mu}_t^* = \mathbf{H}^{-1}\mathbf{b}_t$$
  Where:
  $$\mathbf{H} = \mathbf{W}^T \mathbf{\Pi}_e \mathbf{W} + \mathbf{\Pi}_s$$
  $$\mathbf{b}_t = \mathbf{W}^T \mathbf{\Pi}_e \mathbf{e}_t + \mathbf{\Pi}_s \mathbf{A}\mathbf{\mu}_{t-1}$$
- **Hessian Positive Definiteness:** Since $\mathbf{\Pi}_e > 0$ and $\mathbf{\Pi}_s > 0$ are diagonal precision matrices (and thus positive-definite), and $\mathbf{W}$ is a real matrix, the Hessian $\mathbf{H}$ is guaranteed to be symmetric positive-definite (SPD). This guarantees that $\mathbf{H}$ is invertible and has a unique global minimum, ensuring **100% numerical stability** under any parameter scale.

### C. Systems-Level Cholesky Pre-computation
- Because the generative mapping $\mathbf{W}$ and precision matrices are frozen during test-time, the Hessian $\mathbf{H}$ is constant.
- Pre-computing the Cholesky factorization $\mathbf{H} = \mathbf{L}\mathbf{L}^T$ once upon calibration reduces the test-time serving cost from cubic $\mathcal{O}(K^3)$ matrix inversion to quadratic $\mathcal{O}(K^2)$ forward-backward substitution. This systems-level optimization is mathematically rigorous and highly practical for high-throughput serving engines (like vLLM or S-LoRA).

---

## 2. Examination of Modeling Assumptions and Limitations
The authors are remarkably honest and thorough in discussing and addressing their modeling assumptions and limitations (found under Appendix N / Limitations and Future Work):

1. **Static Variational Covariance matrix ($\mathbf{\Sigma}_t$):**
   - *Limitation:* Treating $\mathbf{\Sigma}_t$ as a static parameter prevents the router from adapting to input-dependent uncertainty (such as highly ambiguous out-of-distribution queries) at test-time.
   - *Soundness of Mitigation:* The authors propose an elegant future roadmap: parameterizing $\mathbf{\Sigma}_t$ as a function of the *lagged* prediction error: $\mathbf{\Sigma}_t = g_{\phi}(\mathbf{e}_{t-1} - \mathbf{W}\mathbf{\mu}_{t-1})$. Because $\mathbf{\Sigma}_t$ depends only on past states, at the current step $t$ it acts as a constant with respect to the optimization variable $\mathbf{\mu}_t$. This preserves the strictly convex quadratic structure of the free energy, and the exact closed-form solution remains $\mathbf{\mu}_t^* = \mathbf{H}^{-1}\mathbf{b}_t$, where only the Hessian varies over time but remains SPD and trivially invertible. This is highly mathematically sound.

2. **Diagonal Transition Prior Matrix ($\mathbf{A}$):**
   - *Limitation:* Treating $\mathbf{A} = \text{diag}(a_k)$ as diagonal assumes independent prior temporal retention of expert beliefs, which fails to capture structured Markovian transitions.
   - *Soundness of Mitigation:* The authors show that their closed-form framework is fully compatible with a dense transition matrix $\mathbf{A} \in (0,1)^{K \times K}$. This preserves the linear prior expectation $\mathbf{A}\mathbf{\mu}_{t-1}$, and the exact closed-form solution is preserved, with only the target vector modified via a standard matrix-vector multiplication ($\mathcal{O}(K^2)$ cost). This represents a direct and elegant extension.

3. **Mathematical Support Mismatch of the Likelihood Model:**
   - *Limitation:* The Gaussian likelihood model $p(\mathbf{e}_t | \mathbf{s}_t)$ defines a probability distribution over the entire real space $\mathbb{R}^K$, while coordinate projection observations are strictly non-negative ($\mathbf{e}_t \in \mathbb{R}_{\ge 0}^K$).
   - *Soundness of Mitigation:* The authors formalize a non-negative observation model using a multivariate Truncated Gaussian likelihood restricted to $\mathbb{R}_{\ge 0}^K$. They show that while the resulting CDF normalization breaks the strictly quadratic structure, a fast single-step linear solve can still be preserved by employing a Laplace approximation (using a second-order Taylor expansion around the prior expectation $\mathbf{A}\mathbf{\mu}_{t-1}$ to construct a local quadratic surrogate of the free energy). This is a standard and highly sound control-theoretic tractability trade-off.

---

## 3. Analysis of Inhibitory Pathways (Ablation Study)
- **Excitatory-Inhibitory Balance:** The unconstrained generative mapping $\mathbf{W}$ allows both positive (excitatory) and negative (inhibitory) weights.
- **The Non-Negative Ablation ($\mathbf{W} \ge 0$):** Constraining $\mathbf{W}$ to be non-negative does not degrade the sequence-averaged accuracy. However, the continuous trajectory analysis (Figure 4) reveals a significant **15-step transient lag** specifically at task switch boundaries.
- **Mechanistic Necessity of Inhibition:** The authors provide a brilliant explanation for this lag: a non-negative generative mapping is incapable of forming negative feedback loops to actively suppress obsolete expert beliefs, relying instead on passive decay. This empirically and theoretically confirms that active task suppression (inhibitory pathways) is mechanically required to prevent transient transition lag in dynamic ensembling streams.

---

## 4. Evidential Decay for Out-of-Distribution Queries
- Under out-of-distribution (OOD) queries, standard PCA projections yield a sensory observation $\mathbf{e}_t \approx \mathbf{0}$.
- The authors mathematically prove that since transition prior eigenvalues of the update matrix $\mathbf{M} = (\mathbf{W}^T\mathbf{\Pi}_e\mathbf{W} + \mathbf{\Pi}_s)^{-1}\mathbf{\Pi}_s\mathbf{A}$ are strictly less than 1, the belief vector $\mathbf{\mu}_t^*$ is guaranteed to exponentially decay towards $\mathbf{0}$.
- This evidential decay naturally drives the Gibbs Softmax ensembling weights to converge to a safe, maximum-entropy uniform ensembling $[1/K, \dots, 1/K]$. This is a beautiful, self-regulating property of active inference that minimizes worst-case prediction error under total uncertainty.

**Conclusion on Soundness:** The methodology is exceptionally sound, mathematically airtight, and theoretically rich. The authors have anticipated and addressed every potential mathematical and structural concern with rigorous control-theoretic derivations and empirical confirmations.
