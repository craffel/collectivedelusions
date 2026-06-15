# 1. Summary of the Paper

## Main Topic and Motivation
The paper addresses the challenge of **test-time dynamic model ensembling/routing** under sequential, heterogeneous query streams. When deploying a pre-trained base network with multiple task-specific adapters (such as LoRA), the system must route incoming queries to the correct adapter. Existing "stateless" routers (e.g., SABLE, PAC-ZCA) process each query independently, which leads to high-frequency routing oscillations ("routing jitter paradox") due to query-level noise. Heuristic "stateful" routers (e.g., ChemMerge) reduce jitter by smoothing routing weights but lack formal stability and generalization guarantees, leading to severe accuracy degradation under heterogeneous streams with rapid task switches.

To resolve these limitations, the paper proposes **PAC-Kinetics**, a stateful ensembling framework that:
1. Formulates representation dynamics as a continuous-time non-equilibrium chemical kinetics system, yielding a stable discrete linear recurrence.
2. Proves global asymptotic stability (GAS) and input-to-state stability (ISS) for the router's dynamics.
3. Derives a Catoni-type PAC-Bayesian generalization bound for stationary $\beta$-mixing stochastic processes to optimize kinetics and routing parameters on a short calibration split.
4. Introduces **Adaptive Online Kinetics** to dynamically suppress state retention (routing lag) during rapid task switches, balancing stability and responsiveness.

---

## Claimed Contributions and Evidence

### 1. Unified Mathematical and Control-Theoretic Foundation
* **Claim:** Replacing empirical heuristics in stateful ensembling with provable contractive and Lyapunov stability guarantees (GAS and ISS), alongside a learning-theoretic generalization bound under non-i.i.d. conditions.
* **Evidence:** 
  * Section 3.7 provides proofs of Lipschitz continuity (Lemma 3.2) and global asymptotic stability/input-to-state stability (Lemma 3.3) under zero and bounded inputs, respectively. This is extended to the time-varying Adaptive Online Kinetics system.
  * Section 3.6 derives Theorem 3.1, a Catoni-type PAC-Bayesian bound under stationary $\beta$-mixing stochastic processes using Even/Odd Block Splitting.
  * Section 7.2 provides a detailed trajectory sensitivity bound proving that the trajectory discrepancy under parameter perturbations scales quadratically as $(1-\rho)^{-2}$, justifying the use of the deterministic surrogate mean rather than the randomized posterior.

### 2. High Accuracy and Jitter Reduction in Simulation
* **Claim:** Slashing routing jitter by over $11.2\times$ on orthogonal streams and up to $16.0\times$ on overlapping streams compared to stateless SABLE, while matching or exceeding Oracle accuracy in a simulated Coordinates Sandbox.
* **Evidence:**
  * Tables 1 and 2 report joint serving accuracy and routing jitter in the Analytical Coordinate Sandbox (ICS) over 5 random seeds.
  * On homogeneous orthogonal streams, PAC-Kinetics achieves $95.03\% \pm 0.02\%$ accuracy (matching the Expert Oracle) and a jitter of $0.0062 \pm 0.0000$ ($11.24\times$ reduction from SABLE's $0.0697$).
  * On homogeneous overlapping streams, it achieves $95.07\% \pm 0.02\%$ accuracy and $0.0063 \pm 0.0002$ jitter ($16.25\times$ reduction from SABLE's $0.1024$).

### 3. Stateful Robustness under Rapid Workload Shifts
* **Claim:** Avoiding the accuracy collapse of static stateful routers (e.g., ChemMerge) on heterogeneous streams by optimizing parameters via the PAC-Bayesian bound and incorporating Adaptive Online Kinetics.
* **Evidence:**
  * In Table 1 (Orthogonal Heterogeneous Stream), while ChemMerge collapses to $70.59\% \pm 0.75\%$ accuracy, PAC-Kinetics maintains $92.35\% \pm 0.48\%$ accuracy.
  * In Table 2 (Overlapping Heterogeneous Stream), while ChemMerge drops to $70.49\% \pm 0.93\%$, PAC-Kinetics achieves $92.90\% \pm 0.71\%$ accuracy.
  * Comparison with Stateful ERM (which achieves $87.14\%$ and $83.04\%$ in the respective heterogeneous streams) highlights the regularizing benefit of the PAC-Bayesian Complexity penalty.

### 4. CPU/GPU Efficiency and Physical Validation
* **Claim:** Demonstrating flat, microsecond-scale execution latency on CPU/GPU and validating empirical effectiveness on real-world datasets with a physical PyTorch LoRA network.
* **Evidence:**
  * Section 7.1.3 profiles CPU latency at flatly $\approx 10.4$ microseconds, and Section 7.1.6 profiles vectorized GPU execution at under $3.5$ microseconds.
  * Section 7.1.6 details physical evaluation on MNIST and Fashion-MNIST using a 3-layer MLP. On homogeneous streams, PAC-Kinetics achieves $76.40\% \pm 5.50\%$ classification accuracy (outperforming stateless PAC-ZCA at $71.20\%$ and static Uniform Merging at $54.90\%$). On heterogeneous streams, it achieves $66.30\% \pm 7.79\%$ (outperforming static Uniform Merging at $54.90\%$ and ChemMerge at $58.60\%$).
