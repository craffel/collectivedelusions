# 2_novelty_check.md: Assessment of Novelty and Delta from Prior Work

## Key Novel Aspects
1. **First Closed-Loop Control-Theoretic Framework for Dynamic Serving:**
   While previous research recognized the necessity of temporal smoothing, prior stateful routers operated in an open-loop fashion, accumulating historical routing weights without feeding back the current system state. PID-Merge introduces a formal closed-loop system where the reference signal (the setpoint) is the raw similarity weight $w_k^{(l)}$ and the plant output is the previous layer's ensembling coefficient $\alpha_k^{(l-1)}$. This closed-loop tracking error feedback ($e_k^{(l)} = w_k^{(l)} - \alpha_k^{(l-1)}$) is entirely novel in the model serving literature.

2. **Derivative (D) Anticipation for Short Horizons:**
   By incorporating the discrete second-order derivative term ($\Delta^2 e_k^{(l)}$), PID-Merge is the first routing framework to actively measure *error tracking acceleration*. This allows the controller to instantly detect the transition at the boundary layer (Layer 3) and provide an anticipatory boost that overcomes the depth-wise transition lag. This enables the controller to converge within 2--3 layers, a critical capability for short adapted network depths (e.g., $9$--$11$ layers).

3. **Dynamic $K$-Scaled Anti-Windup Clamping (Conditional Integration):**
   Prior stateful methods fail to scale to deep topologies due to integrator windup, where unnormalized logits grow unchecked as weights saturate near simplex boundaries. PID-Merge introduces a dynamic, $K$-scaled clamping threshold ($\theta_{\text{high}} = 1 - \epsilon, \theta_{\text{low}} = \epsilon/K$) and freezes the integral accumulator when active weights exceed these thresholds in the direction of integration. This control-theoretic mechanism prevents logit inflation and eliminates saturation-induced transition lag in deep networks.

4. **Scaled Logit Mean-Centering Safeguard:**
   To resolve the lack of translation invariance in Softmax functions under expert-specific temperatures ($\tau_k \neq \tau_j$), the paper introduces *scaled logit mean-centering* ($\bar{s}_k^{(l)} = \tilde{s}_k^{(l)} - \frac{1}{K} \sum_j \tilde{s}_j^{(l)}$ where $\tilde{s}_k^{(l)} = s_k^{(l)}/\tau_k$). This is a mathematically elegant, numerically free safeguard that prevents absolute logit drift and floating-point overflow without distorting active ensembling probabilities.

5. **Prefill-Locked Autoregressive Design:**
   To resolve the severe manifold misalignment and $O(T \cdot D \cdot r)$ KV Cache re-projection bottleneck during decoding, the paper designs a "Prefill-Locked" policy. Weights are computed and locked during the prefill phase, then applied statically during decoding. This guarantees absolute KV cache coherence and slashes decoding latency overhead to zero.

## The 'Delta' from Prior Work
The delta is substantial and clearly delineated:
* **Vs. SABLE (Stateless):** SABLE calculates expert weights independently layer-by-layer and sample-by-sample, which makes it highly vulnerable to representation noise. PID-Merge uses the closed-loop Integral (I) term as a layer-wise low-pass filter to smooth out these high-frequency layer-to-layer oscillations, reducing depth-wise jitter by over **73%** on physical GPT-2, while maintaining the same accuracy level.
* **Vs. ChemMerge & PAC-Kinetics (ODE Kinetics):** ChemMerge simulates continuous-time chemical kinetics using first-order ODEs. This requires adaptive high-order solvers that introduce a massive $0.482$ ms latency overhead, violating real-time edge constraints. PID-Merge runs in discrete-time using an incremental velocity form, requiring only $15$ FLOPs and adding a negligible $0.012$ ms of latency ($40\times$ faster). Furthermore, ChemMerge carries states temporally across different users' queries (creating security/privacy leakage risks), whereas PID-Merge resets states per query, enforcing strict multi-tenant isolation.
* **Vs. Momentum-Merge (Open-Loop EMA):** Momentum-Merge uses a simple, open-loop EMA. This introduces severe inertial drag (phase delay) under rapid task transitions, collapsing accuracy under heterogeneous workloads. PID-Merge utilizes closed-loop error feedback and derivative action to completely eliminate this lag, outperforming Momentum-Merge by **+8.65%** absolute accuracy under overlapping heterogeneous streams.

## Characterization of Novelty
The novelty of this work is **significant and highly robust**. It does not merely repackage existing concepts, but systematically bridges classical control theory and machine learning systems. By identifying the deep layer-to-layer routing problem as a discrete-time tracking control problem, it provides a rigorous, highly interpretable, and computationally elegant solution. The mathematical proofs of discrete-time BIBO stability via Jury's criterion further elevate the theoretical rigor of the work. Rather than relying on metaphorical chemical analogies, it establishes a solid mathematical and physical foundation that addresses real-world systems bottlenecks (latency, memory, security, and KV cache coherence).
