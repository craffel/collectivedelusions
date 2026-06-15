# 1. Summary of the Paper

## Main Topic and Approach
The paper addresses the **Jitter-Lag Trade-Off** in dynamic model serving and ensembling within sequential, non-stationary streams. When blending specialized parameter-efficient experts (e.g., via LoRA adapters or Mixture-of-Experts) at test-time, standard stateless routers (e.g., SABLE) suffer from extreme high-frequency **routing jitter** due to sensory noise, while stateful alternatives (e.g., exponential filters or biochemical ODE-based ensembling like ChemMerge) suffer from significant **representational lag (inertial drag)** during task transitions.

To resolve this trade-off, the authors propose **Active Inference Routing (AIR)**, which frames dynamic model ensembling as active perception. Rather than relying on feedforward heuristics or rigid low-pass filters, AIR models the routing layer as an active agent tracking a stateful Gaussian variational belief over latent task contexts. By minimizing Variational Free Energy (which simplifies to a precision-weighted combination of prior and sensory prediction errors), the router dynamically balances top-down prior expectations and bottom-up sensory coordinate projections.

Crucially, the authors derive an **exact closed-form analytical solution** to this variational free energy minimization. Under static variational covariance, this formulation is mathematically equivalent to a classical linear state observer (Kalman filter). By precomputing the Cholesky factorization of the Hessian matrix ($\mathbf{H} = \mathbf{L}\mathbf{L}^T$) at calibration time, the optimal task belief is retrieved at test-time using microsecond-level forward-backward substitution with quadratic $\mathcal{O}(K^2)$ complexity.

---

## Key Findings and Claims
1. **Dynamic Resolution of the Jitter-Lag Trade-Off:** Evaluated on the Analytical Coordinate Sandbox (ACS), AIR matches the accuracy and representation alignment of an omniscient expert oracle under rapid, non-stationary transitions. Meanwhile, under stable but noisy streams, AIR suppresses routing jitter by up to **2.49$\times$** compared to SABLE.
2. **Closed-Form Exact Solver Efficiency:** By formulating the perception step as a quadratic convex optimization, the authors bypass the need for iterative gradient unrolling, adaptive step sizes, or spectral stability penalties. The Cholesky precomputation results in negligible serving latency (less than 0.5% relative overhead).
3. **Necessity of Active Inhibition:** A mechanistic ablation study confirms that allowing negative weights in the generative coordinate mapping matrix ($\mathbf{W} \in \mathbb{R}^{K \times K}$) is essential. Restricting $\mathbf{W} \ge 0$ (non-negative ablation) causes a significant 15-step transient lag at task boundaries, demonstrating that active task suppression (excitatory-inhibitory balance) is required to prevent transition lag.
4. **Generalization and Parameter-Efficiency:** A highly compact linear-complexity variant, **AIR (Diagonal)**, restricts $\mathbf{W}$ to be diagonal (reducing parameter count to $5K$). This variant achieves outstanding accuracy and noise stability under tiny calibration streams ($T_{\text{cal}} = 32$), demonstrating resistance to sequence-slicing overfitting.

---

## Explicitly Claimed Contributions and Supporting Evidence
- **Brain-Inspired Gating Framework:** Reframing model merging as active perception. Supported by Section 3 and Appendix A, which derive the Variational Free Energy equations.
- **Variational Free Energy Formulation:** Showing how VFE simplifies to precision-weighted squared prediction errors, establishing equivalence with a linear Kalman filter. Supported by Appendix A and Section 3.3.
- **Solving the Jitter-Lag Dilemma on ACS:** Demonstrated empirically in Table 1, where AIR maintains oracle-level alignment accuracy ($66.23\%$) on heterogeneous streams with high tracking speed (1.4202 jitter), while suppressing noise ($0.0364$ jitter) on homogeneous streams.
- **Verification of Inhibitory Pathways:** Supported by Figure 4 (trajectory analysis) and Section 4.5, proving that restricting $\mathbf{W}$ to non-negative elements causes a localized 15-step transient lag.
- **Registry Scaling and Calibration Generalization:** Supported by Appendix M and N, which evaluate the framework at $K=16$ scale and perform cross-sequence calibration stress tests.
