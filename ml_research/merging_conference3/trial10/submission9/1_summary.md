# Evaluation Checklist: Summary and Overview of the Paper

## 1. Problem Statement & Motivation
- **Context:** Large-scale deep learning serving systems are increasingly modular, deploying multiple parameter-efficient experts (e.g., LoRA adapters) to process diverse, non-stationary query streams. This avoids the high memory overhead of running multiple distinct models.
- **The Core Bottleneck:** Standard routing layers face the acute **Jitter-Lag Trade-Off**:
  1. **Stateless Routers (e.g., SABLE):** Evaluate queries in isolation. They adapt instantly to context switches but exhibit extreme, high-frequency **routing jitter (noise)** under minor representation perturbations or sensory observation noise. This jitter causes systems-level bottlenecks (e.g., hardware cache thrashing from continuously swapping LoRA/PEFT parameters) and representational instability (activation distribution disruptions across deep layers).
  2. **Stateful Routers (e.g., ChemMerge, Momentum-Merge):** Smooth routing trajectories using chemical kinetics ODEs or exponential moving averages (EMA). However, they introduce severe **representational lag (inertial drag)** at task switch boundaries because they rigidly smooth over history, leading to accuracy collapses during transition phases.
- **The Core Objective:** Resolve the Jitter-Lag Trade-Off by designing an adaptive, noise-filtering, yet highly responsive routing layer.

---

## 2. Methodology & Mathematical Formulation
- **Active Inference Routing (AIR):** Models the routing layer as an active cognitive agent performing test-time perception in a dynamic state-space environment.
- **State-Space Model:**
  - Latent belief state: $\mathbf{s}_t \in \mathbb{R}^K$, tracking active task context log-probabilities.
  - Transition prior: $p(\mathbf{s}_t | \mathbf{s}_{t-1}) = \mathcal{N}(\mathbf{s}_t; \mathbf{A}\mathbf{s}_{t-1}, \mathbf{\Sigma}_s)$, with a diagonal state retention matrix $\mathbf{A} = \text{diag}(a_k)$.
  - Observation model: $p(\mathbf{e}_t | \mathbf{s}_t) = \mathcal{N}(\mathbf{e}_t; \mathbf{W}\mathbf{s}_t, \mathbf{\Sigma}_e)$, where $\mathbf{e}_t$ contains the task coordinate projection values, and $\mathbf{W}$ is the unconstrained generative coordinate mapping (allowing excitatory and inhibitory pathways).
- **Variational Free Energy Minimization:**
  - Under Gaussian posterior $q(\mathbf{s}_t) = \mathcal{N}(\mathbf{s}_t; \mathbf{\mu}_t, \mathbf{\Sigma}_t)$, the objective simplifies to precision-weighted sensory and prior prediction errors:
    $$\mathcal{F}_t \propto \frac{1}{2} (\mathbf{e}_t - \mathbf{W}\mathbf{\mu}_t)^T \mathbf{\Pi}_e (\mathbf{e}_t - \mathbf{W}\mathbf{\mu}_t) + \frac{1}{2} (\mathbf{\mu}_t - \mathbf{A}\mathbf{\mu}_{t-1})^T \mathbf{\Pi}_s (\mathbf{\mu}_t - \mathbf{A}\mathbf{\mu}_{t-1})$$
  - Where $\mathbf{\Pi}_e = \mathbf{\Sigma}_e^{-1}$ (sensory precision) and $\mathbf{\Pi}_s = \mathbf{\Sigma}_s^{-1}$ (prior precision).
- **Exact Closed-Form Solution:**
  - Because $\mathcal{F}_t$ is strictly convex and quadratic in $\mathbf{\mu}_t$, taking the derivative and setting it to $\mathbf{0}$ yields the exact belief update:
    $$\mathbf{H}\mathbf{\mu}_t^* = \mathbf{b}_t$$
    $$\mathbf{H} = \mathbf{W}^T \mathbf{\Pi}_e \mathbf{W} + \mathbf{\Pi}_s, \quad \mathbf{b}_t = \mathbf{W}^T \mathbf{\Pi}_e \mathbf{e}_t + \mathbf{\Pi}_s \mathbf{A}\mathbf{\mu}_{t-1}$$
- **Systems-Level Pre-computation (Cholesky Factorization):**
  - Since $\mathbf{H}$ is constant during test-time, the router pre-computes the Cholesky decomposition $\mathbf{H} = \mathbf{L}\mathbf{L}^T$ once upon calibration.
  - Test-time updates are solved in quadratic time $\mathcal{O}(K^2)$ using forward-backward triangular substitution, reducing serving overhead to microsecond scales ($8\text{--}39\,\mu\text{s}$).
- **Action Policy:**
  - Ensembling weights are computed via a Gibbs Softmax policy over the converged belief state:
    $$\alpha_t = \text{Softmax}(\mathbf{\mu}_t^* / \tau)$$

---

## 3. Key Experimental Setup & Results
- **Benchmark Environment:** Analytical Coordinate Sandbox (ACS)—a high-fidelity 14-layer, 192-dimensional simulation environment reproducing non-stationary streaming serving.
- **Key Metric Focus:** Downstream task accuracy (categorical and representation alignment) vs. sequential ensembling weight jitter.
- **Advanced Evaluations and Robustness Checks:**
  1. **Registry Scaling (up to $K=16$):** Demonstrates stable ensembling under large registers. Highlights a parameter-efficient **AIR (Diagonal)** variant that reduces parameter scaling from $\mathcal{O}(K^2)$ to linear $\mathcal{O}(K)$, solving the small-sample calibration bottleneck ($T_{\text{cal}} = 32$).
  2. **Cross-Sequence Calibration Robustness:** Compares parameters calibrated in highly Stable (sparse task switches) vs. Highly Dynamic (continuous task switches) environments. Demonstrates robust generalization with negligible accuracy discrepancies.
  3. **High-Dimensional Nonlinear Manifold Stress Test:** Warps task projection coordinates non-linearly (sinusoidal-quadratic) and injects heavy-tailed Student's $t$ noise ($\nu=3$). Shows that AIR's routing stability directly prevents downstream categorical accuracy collapses (retaining $98.83\%$ vs. SABLE's $93.99\%$ collapse).
  4. **PCA Projection Dimension $d$ Sweep:** Varies $d$ from 1 to 12. Analyzes the Pareto-frontier between low-rank regularizing noise-rejection (small $d$) and detailed semantic representation (large $d$). Empirically validates $d=4$ as the optimal trade-off for $K=4$ experts.

---

## 4. Overall Initial Assessment
- **Strengths:** 
  - Exceptionally solid mathematical formulation bridging systems control engineering (Kalman observers) with cognitive science (Free Energy Principle).
  - Elegant resolution of the Jitter-Lag Trade-Off with an exact closed-form solution that requires no unrolled iterative optimization.
  - Extremely thorough experimental validation, including systems-level micro-benchmarking, registry scaling up to $K=16$, nonlinear stress-testing, and hyperparameter sensitivity sweeps.
- **Weaknesses to probe:**
  - Assumption of a static variational covariance matrix $\mathbf{\Sigma}_t$ (though the authors provide a compelling future path with lagged feedback networks in Appendix N).
  - Reliance on PCA as a linear projection method (partially addressed by the Contractive Autoencoder evaluation in Appendix N).
- **Verdict:** Highly innovative and technically robust paper with extensive evaluation. It offers a mathematically elegant and practical solution to sequential routing bottlenecks.
