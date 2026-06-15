# Comprehensive Summary of the Submission

## Main Topic and Approach
The paper introduces **Unitary Geodesic Routing (UGR)**, a novel geometric paradigm for dynamic test-time model ensembling across heterogeneous, non-stationary task streams. Test-time ensembling blends specialized parameter-efficient fine-tuning (PEFT) expert adapters on-the-fly to adapt a base neural network to sequential input queries. 

Prior stateful methods (e.g., *Momentum-Merge*, *ChemMerge*) perform state updates in unconstrained flat Euclidean spaces and project them onto the probability simplex post-hoc using operators like Softmax. The authors argue that this Euclidean-to-simplex mismatch introduces representational lag (hysteresis), geometric scale distortion of hidden activations, and high-frequency routing jitter.

To resolve these issues, UGR models the ensembling routing state directly on the curved $(K-1)$-dimensional unit hypersphere $\mathbb{S}^{K-1}$. It maps the spherical state vector to the probability simplex via the square-root homeomorphism from Information Geometry (resembling Born's rule in quantum mechanics): $\alpha_{k,t} = (s_{k,t})^2$. Because updates are constrained to the hypersphere, geodesic rotations (spherical interpolations) along the shortest great-circle path map natively to exact, closed-form Fisher-Rao geodesic flows on the probability simplex.

To handle the stability-plasticity trade-off, UGR introduces:
1. **Torque-Driven Adaptive Agility:** A self-regulating mechanism where the angular step size scales with the representational torque (angular distance between the previous state and incoming target activations). This accelerates state transitions during task switches and dampens them during stable periods.
2. **Spatial-Temporal Geodesic Coupling:** A state propagation mechanism across sequential query boundaries ($\mathbf{s}_t^{(L_{\text{frozen}})} = \mathbf{s}_{t-1}^{(L)}$) to enforce temporal coherence across samples.

---

## Explicitly Claimed Contributions and Accompanying Evidence

1. **Curved State-Space Formulation & Born Mapping:**
   * *Claim:* Operating on the unit hypersphere $\mathbb{S}^{K-1}$ maps to the probability simplex natively via the square-root homeomorphism, representing closed-form Fisher-Rao geodesic flows with zero geometric distortion or activation scale-mismatch.
   * *Evidence:* Mathematical proof that $\|\mathbf{s}_t\|_2^2 = 1 \implies \sum_k \alpha_k = 1$ with $\alpha_k \ge 0$, bypasses Softmax layers, and avoids representational boundary compression.

2. **Closed-Form Geodesic Updates:**
   * *Claim:* Bypasses expensive matrix exponentials and numerical ODE integrations, delivering high computational efficiency.
   * *Evidence:* A derived, closed-form Rodrigues-like geodesic rotation operator (spherical linear interpolation / Slerp) requiring only element-wise dot-products, additions, and minimal scalar trigonometric functions. Algorithmic complexity is shown to be $\mathcal{O}(KD)$ per layer, sharing the same asymptotic complexity as standard SABLE or Momentum-Merge but bypassing numerical ODE solvers. Wall-clock latency benchmarks report that UGR adds $<0.07$ ms of latency per query over stateless SABLE.

3. **Torque-Driven Adaptive Agility:**
   * *Claim:* Dynamically scales routing inertia based on representational torque, eliminating representational lag under sudden transitions while suppressing high-frequency jitter.
   * *Evidence:* Qualitative trajectory visualization (Figures 1 and 3) showing rapid, overshoot-free transitions under task switches. Quantitative results (Table 2) showing that standard UGR reduces routing jitter by over 2.10$\times$ inside the query compared to ChemMerge.

4. **Spatial-Temporal Geodesic Coupling:**
   * *Claim:* Smoothly propagates ensembling trajectories across query boundaries to maintain temporal coherence.
   * *Evidence:* Ablations on the 20newsgroups NLP stream comparing Reset and Coupled configurations. Coupled Momentum-Merge experiences a large performance leap (+18.32% absolute accuracy), and UGR (which uses coupling) achieves the highest joint accuracy (92.25%) and low routing jitter (3.68 $\times 10^{-4}$).

5. **Empirical Performance on Benchmarks:**
   * *Claim:* Achieves state-of-the-art accuracy and stability on both synthetic and real-world task streams.
   * *Evidence:* Evaluated on the 14-layer Analytical Coordinate Sandbox (ICS) across 10 random seeds (75.08% classification accuracy, 2.10$\times$ lower routing jitter than ChemMerge) and on a 20newsgroups multi-task MoE classification benchmark across 5 random seeds (92.25% accuracy, 1.63$\times$ lower jitter than Coupled Momentum-Merge).
