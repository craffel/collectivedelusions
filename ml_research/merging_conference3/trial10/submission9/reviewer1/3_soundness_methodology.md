# 3. Soundness & Methodology

## Technical Soundness and Clarity of Description
The mathematical derivation of Active Inference Routing (AIR) is generally presented with a high degree of clarity and structure. Reframing ensembling routing as a state-space optimization problem using the Free Energy Principle is conceptually elegant. However, a rigorous, adversarial examination of the methodology reveals several significant **hidden assumptions, mathematical simplifications, and structural mismatches** that compromise the theoretical purity of the active inference formulation.

---

## Technical Flaws and Hidden Assumptions

### 1. Mathematical Support Mismatch of the Likelihood Model
The observation likelihood model is assumed to be a standard multivariate Gaussian:
$$p(\mathbf{e}_t \mid \mathbf{s}_t) = \mathcal{N}(\mathbf{e}_t; \mathbf{W}\mathbf{s}_t, \mathbf{\Sigma}_e)$$
This defines a probability distribution over the entire real space $\mathbb{R}^K$. However:
* The sensory coordinate observations $\mathbf{e}_t$ are computed as L2 norms of projection coordinates onto PCA subspaces ($e_{k, t} = \|V_k^T \tilde{\mathbf{z}}_t\|_2$). Thus, they are strictly non-negative: $\mathbf{e}_t \in \mathbb{R}_{\ge 0}^K$.
* The generative mapping matrix $\mathbf{W} \in \mathbb{R}^{K \times K}$ is completely unconstrained (which is a core requirement of the active inhibition claim). Thus, the linear predictor $\mathbf{W}\mathbf{s}_t$ can easily predict negative coordinate values.
This creates a fundamental support mismatch between the Gaussian observation model and physical activation observations. 

While the authors acknowledge this in Appendix G (Point 6) and suggest a Truncated Gaussian likelihood, they dismiss it because the resulting cumulative distribution function (CDF) normalization breaks the strictly convex quadratic structure of the free energy, preventing a fast closed-form linear solver and requiring slow, iterative gradient unrolling. They propose a Laplace approximation as a future extension, but do not implement it. Thus, the core implementation relies on a mathematically mismatched likelihood model, which is a notable theoretical flaw.

### 2. Static Variational Covariance Limitation
A primary assumption of the paper is that the variational covariance matrix $\mathbf{\Sigma}_t$ (representing the tracking uncertainty of the belief state) is completely constant (static) during serving:
* This represents a notable modeling limitation: at test-time, the model is unable to represent or adapt to dynamic, input-dependent uncertainty (such as highly ambiguous, out-of-distribution, or corrupted queries).
* If uncertainty were dynamic, the exact closed-form linear solver would be compromised, or the system would have to re-evaluate and re-factorize the Cholesky decomposition of the step-specific Hessian ($\mathbf{H}_t = \mathbf{W}^T \mathbf{\Pi}_{e, t} \mathbf{W} + \mathbf{\Pi}_{s, t}$) at *every single serving step*, which would incur a cubic $\mathcal{O}(K^3)$ latency overhead.
The static variational covariance assumption fundamentally simplifies the FEP to a classical linear Kalman filter with static gains, stripping the system of its capacity to represent dynamic, online confidence.

### 3. Heuristic First-Order Approximation of the Temporal Prior
In Section 3.3, the authors apply a first-order approximation to the temporal prior expectation:
$$\mathbb{E}_{q(\mathbf{s}_t)} \left[ (\mathbf{s}_t - \mathbf{A}\mathbf{s}_{t-1})^T \mathbf{\Pi}_s (\mathbf{s}_t - \mathbf{A}\mathbf{s}_{t-1}) \right]$$
They replace the unobserved previous state $\mathbf{s}_{t-1}$ with its posterior mean estimate $\mathbf{\mu}_{t-1}$ (the "helpful systems-level simplification" in Equation 13). 
* In a mathematically rigorous Kalman Filter or variational filter, the prior transition should integrate over the full posterior distribution $q(\mathbf{s}_{t-1}) = \mathcal{N}(\mathbf{s}_{t-1}; \mathbf{\mu}_{t-1}, \mathbf{\Sigma}_{t-1})$, which propagates uncertainty recursively.
* By ignoring $\mathbf{\Sigma}_{t-1}$ and replacing $\mathbf{s}_{t-1}$ with $\mathbf{\mu}_{t-1}$, they prevent uncertainty from propagating across time. 
While this makes the test-time computational footprint smaller, it is a heuristic simplification that violates standard active inference and Kalman filtering theory. It should be explicitly labeled as a heuristic approximation rather than a mathematically complete derivation of active inference.

### 4. Diagonal Transition Prior Assumption
The transition prior matrix is assumed to be diagonal: $\mathbf{A} = \text{diag}(a_k)$. This assumes that the prior temporal retention of each expert belief is independent. In real-world serving workloads, task transitions exhibit highly structured transition dynamics (e.g., Task A is highly likely to be followed by Task B, but never by Task C). Treating task transition priors as independent diagonal decay factors is a massive simplification that neglects real temporal transition dynamics. While they argue in Appendix G that a dense transition matrix is compatible, it is not implemented or evaluated in the paper.

---

## Reproducibility Analysis
The authors provide detailed descriptions of hyperparameters, algorithmic steps (Algorithm 1), and optimization settings (Appendix B). They also provide details on initialization and learning rates.

However, several critical issues affect reproducibility:
1. **Lack of Codebase Accessibility**: The authors state that the complete source code, sandbox simulation, and scripts are fully available at a public repository (`https://github.com/active-inference-routing/air-serving`). Under a blind review process (which ICML uses), including a public URL that reveals the active project repository or could be a dummy placeholder is a potential violation of double-blind review guidelines.
2. **Synthetic Evaluation only**: Because the entire empirical evaluation is conducted on a synthetic, simulated environment (the Analytical Coordinate Sandbox) rather than on real deep neural networks (e.g., LLaMA, ViT) and real multi-task workloads, it is highly challenging to verify if these results would reproduce in a physical, production-grade serving framework. The "simulation-to-physical gap" remains unbridged in the empirical evaluation, meaning reproducibility on real-world deep networks is not demonstrated.
