# 1. Summary of the Paper

This paper introduces **PID-Merge**, a closed-loop, discrete-time Proportional-Integral-Derivative (PID) controlled stateful routing framework designed for dynamic, test-time model ensembling of multiple expert adapters (such as LoRA) in deep multi-tenant networks.

## Core Problem
When serving multi-task query streams, dynamic routers (such as SABLE) compute expert routing weights sample-by-sample and layer-by-layer. However, these stateless dynamic routers suffer from a severe **routing jitter paradox**: layer-by-layer representation noise causes ensembling weights to oscillate wildly, which corrupts activation manifolds and degrades downstream performance. Conversely, prior stateful routers (e.g., ChemMerge or Momentum-Merge) smooth out these oscillations but introduce a severe **inertial drag** (phase/group delay) during sudden task transitions, which causes significant performance drops under heterogeneous, fast-switching workloads.

## Proposed Solution
To resolve the trade-off between stability (noise filtering) and responsiveness (transition tracking), the authors propose **PID-Merge**. They formulate the layer-wise routing of ensembling weights as a discrete-time control problem:
- **Setpoint (Reference):** Raw early-layer nearest-centroid similarity weights $w_k^{(l)}$.
- **Controlled Variable (Plant Output):** Active ensembling coefficients $\alpha_k^{(l-1)}$ applied at the previous layer.
- **Feedback Loop:** Continuous calculation of tracking error $e_k^{(l)} = w_k^{(l)} - \alpha_k^{(l-1)}$.
- **Velocity PID State Update:** Unnormalized controller states $s_k^{(l)}$ are recursively updated across depth using Proportional (P), Integral (I), and Derivative (D) terms:
  $$\Delta s_k^{(l)} = K_p \Delta e_k^{(l)} + K_i e_k^{(l)} + K_d \Delta^2 e_k^{(l)}$$
  The Proportional (P) term reacts to immediate tracking errors, the Integral (I) term provides smoothing across depth to suppress high-frequency noise, and the Derivative (D) term measures error acceleration to anticipate boundary transitions, allowing rapid convergence within 2--3 layers.
- **Key System Enhancements:** 
  1. **Logit Mean-Centering:** To prevent unnormalized controller logit growth and float overflow, a scaled mean-centering is applied: $\bar{s}_k^{(l)} = \tilde{s}_k^{(l)} - \frac{1}{K}\sum_j \tilde{s}_j^{(l)}$, ensuring numerical stability and translation invariance.
  2. **Anti-Windup Clamping (Conditional Integration):** Prevents saturation-induced transition lag by freezing error integration if weights violate dynamic, $K$-scaled boundaries: $\theta_{\text{high}} = 1 - \epsilon$ or $\theta_{\text{low}} = \frac{\epsilon}{K}$.
  3. **Analytical Stability Bounds:** Derived via discrete-time Jury's Stability Criterion: $K_s(2K_p + K_i + 4K_d) < 2$, with derivative bound $K_d < 1/K_s$. These bounds are enforced during calibrated optimization as a soft penalty.
  4. **Prefill-Locked Routing:** Ensembling weights are computed and locked during the prefill phase, then held static during autoregressive decoding. This guarantees key-value (KV) cache coherence across generation steps and adds zero decoding latency.

## Key Findings and Claims
1. **ICS Evaluation:** Evaluated on the Isolating Coordinate Sandbox (ICS), PID-Merge achieves **94.82%** accuracy on heterogeneous, overlapping streams, outperforming ChemMerge by **+6.40%** and Momentum-Merge by **+8.65%** absolute, nearly reaching the stateless oracle ceiling of $94.93\%$.
2. **Zero-Shot Capability:** PID-Merge in its training-free (zero-shot) configuration ($K_p = 0.5$, $K_i = 0.15$, $K_d = 0.2$) achieves **93.35%** accuracy, providing an immediate and highly practical deployable configuration.
3. **Physical GPT-2 Validation:** Validated physically on a 12-layer GPT-2 Small backbone on an NVIDIA A100 GPU routing three actual adapters. Results show a **73%** reduction in depth-wise layer-to-layer jitter compared to SABLE, with a negligible latency overhead of just **0.012 ms** ($40\times$ faster than ChemMerge).
4. **Generalization:** Shows stable out-of-sample parameter generalization when calibrated on a tiny 32-sample sequence.
