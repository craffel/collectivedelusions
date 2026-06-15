# Paper Summary: Lyapunov-Stable Active Representation Coupling (L-ARC)

## 1. Main Topic and Scope
This paper addresses the problem of dynamic model serving on resource-constrained edge devices using parameter-efficient adapters (such as LoRAs). Specifically, it targets "cascading representational drift" and "routing volatility" in dynamic activation-space ensembling. Stateful continuous-time ensembling methods (e.g., ChemMerge) use active representation coupling (warping intermediate hidden representations toward active task centroids by a constant feedback step size $\eta$) to smooth routing weight trajectories. However, the authors argue that constant feedback step sizes cause a "representational backward-shift" that pulls highly refined late-layer activations back toward noisy, early-stage coordinate spaces, degrading performance. 

To solve this, the paper proposes **Lyapunov-Stable Active Representation Coupling (L-ARC)**, a training-free closed-loop control system that regulates feedback rates on-the-fly to guarantee stable, continuous-depth feature propagation.

---

## 2. Proposed Approach and Core Mechanisms
L-ARC introduces several mechanisms rooted in control theory and physical kinetics:
1. **Continuous-Time Concentration Kinetics (NEKR):** Adapts ChemMerge's discretized reaction-decay ordinary differential equations (ODEs) to track expert concentrations $C_k^{(l)}$ across layers.
2. **Entropy-Gated Concentration Gating (ECG-Reset):** An online filter that monitors the Shannon entropy of routing affinities. If entropy exceeds a safety threshold ($\theta_H = 0.95$), the virtual integration step size is set to zero ($\Delta t = 0$), freezing concentration updates to prevent memory corruption during transient failures.
3. **Lyapunov Feedback Controller & Dissipation Guard:** Models representational error relative to task centroids as a candidate Lyapunov function. Under local linearization, the controller derives a local dissipation coefficient $A_b^{(l)}$ on-the-fly. If $A_b^{(l)} \le \theta_G = 0.04$, feedback warping is gated off ($\eta = 0.0$). If $A_b^{(l)} > \theta_G$, the coupling rate scales adaptively: $\eta = \min(\eta_{\max}, \gamma A_b^{(l)})$.
4. **Entropy-Triggered Lyapunov Gating (ET-L-ARC):** Bypasses the Dissipation Guard calculation under low routing uncertainty ($H < 0.15$) or high failure states ($H > 0.95$) to reduce computational latency.
5. **Representation-Agreement State Correction (RASC):** Under systematic routing bias, RASC compares the feedforward router's prediction with feedback representation-space coordinate tracking. In case of disagreement, RASC overrides the corrupted feedforward rates with the softmax of representation similarities ($p_{\text{sim}}^{(l)}$) to break state-locking failures.

---

## 3. Key Findings and Quantitative Results
The authors evaluate L-ARC in a simulated 14-layer "Analytical Coordinate Sandbox (ICS)" with $D=192$ and $K=4$ task manifolds (MNIST, Fashion-MNIST, CIFAR-10, SVHN) across 10 random seeds under three scenarios:
- **Setting A: Static Centroids (Practical Serving):** Centroids are extracted once at Layer 3. 
  - L-ARC achieves a Joint Mean Accuracy of **74.38% $\pm$ 0.31%** and Semantic Similarity of **0.7937 $\pm$ 0.0059**.
  - This marginally out-performs stateless SABLE (74.06% accuracy, 0.7590 similarity) and decoupled ChemMerge (74.33% accuracy, 0.7881 similarity).
- **Setting B: Layer-Specific Centroids (High-Overhead):** Centroids are extracted for all layers.
  - L-ARC achieves **74.46% $\pm$ 0.31%** accuracy, but is outperformed by SABLE (74.82% $\pm$ 0.32%) and a simple EMA-SABLE heuristic (75.00% $\pm$ 0.33%).
- **Setting C: Transient Routing Failures (20% Router Dropouts):**
  - L-ARC (with ECG-Reset) achieves **73.97% $\pm$ 0.39%** accuracy and **0.7813 $\pm$ 0.0075** similarity.
  - This significantly beats open-loop ChemMerge (68.79% $\pm$ 0.44% accuracy) but is virtually identical to the **L-ARC (ECG-Reset Only, $\eta=0$)** baseline which gets **73.93% $\pm$ 0.41%** accuracy.
- **Setting D: Confident Router Bias (Systematic Bias):**
  - L-ARC (with RASC) achieves **73.59% $\pm$ 0.39%** accuracy, outperforming SABLE (69.56%), decoupled ChemMerge (68.52%), and the Decay-ChemMerge heuristic (68.27%).

---

## 4. Explicitly Claimed Contributions
1. **Stateful Serving Framework (ECG-Kinetics):** Proposal of ECG-Reset to shield ODE kinetics and prevent memory fault propagation under transient dropouts.
2. **Control-Theoretic Feedback Formulation:** Proving representation error convergence using a discrete-time Lyapunov stability framework.
3. **Analytical Dissipation Guard:** An online closed-loop controller that adaptively tunes or gates representation warping on-the-fly.
4. **Feedback-Driven State Correction (RASC):** Dual-loop control to override systematic router bias and resolve state-locking failures.
5. **Rigorous Empirical Benchmarking & Transparency:** Evaluation on the 14-layer Coordinate Sandbox with latency profiling, paired t-tests, and hyperparameter sensitivity sweeps.
