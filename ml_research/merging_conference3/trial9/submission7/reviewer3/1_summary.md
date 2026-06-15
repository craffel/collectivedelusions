# 1. Summary of the Paper

## Main Topic and Approach
This paper addresses the challenges of **routing volatility (jitter)** and **cascading representational drift** in dynamic model ensembling for parameter-efficient adapters (such as LoRAs) served on resource-constrained edge devices. 

While stateful continuous-time ensembling methods (e.g., ChemMerge) smooth routing weights across depth using ordinary differential equations (ODEs), they rely on open-loop, heuristic constant feedback step sizes ($\eta$) to warp hidden representations toward pre-computed early-stage centroids. Under mixed or heterogeneous workloads, this constant feedback causes a **representational backward-shift**, pulling highly refined late-layer activations back toward noisy, early-stage coordinates, which degrades serving performance.

To resolve this, the paper introduces **Lyapunov-Stable Active Representation Coupling (L-ARC)**, a training-free closed-loop control framework that provides formal control-theoretic guarantees for continuous-depth feature propagation. L-ARC models the representation similarity error as a system-level candidate Lyapunov function and analytically derives a local **Dissipation Guard** to compute sample-specific, layer-specific feedback rates ($\eta^{(l)}$) on-the-fly, ensuring that representation warping is strictly error-decreasing (dissipative).

The framework also includes:
1. **Entropy-Gated Concentration Gating (ECG-Reset):** A state-space shield that monitors routing entropy to dynamically freeze ODE kinetics during routing dropouts or failures, preventing memory fault propagation.
2. **Entropy-Triggered Gating (ET-L-ARC):** A control-theoretic optimization that evaluates the Dissipation Guard only under moderate routing uncertainty, collapsing computational latency overhead under clean workloads to near-zero.
3. **Representation-Agreement State Correction (RASC):** A dual-loop control mechanism that compares feedforward routing selections with feedback representation-space coordinates, overriding corrupted, systematically biased router outputs to resolve "state-locking" failures.

---

## Key Findings
1. **The Centroid Representational Drift Gap:** Under realistic memory-constrained serving where centroids are extracted only at an early layer (Setting A), stateless approaches like SABLE suffer severe representational degradation, collapse in semantic similarity (down to 0.7590), and routing jitter.
2. **L-ARC Robustness:** In Setting A, full L-ARC (Feedback+ECG-Reset) achieves **74.38% ± 0.31%** accuracy and a semantic similarity of **0.7937 ± 0.0059**, effectively mitigating representational backward-shift and outperforming open-loop models and stateless SABLE.
3. **Redundancy under Clean Serving:** Under clean serving, active representation feedback warping is not statistically significant ($p = 0.0969$) compared to decoupled kinetics ($\eta = 0$), meaning practitioners can set $\eta = 0$ in pristine workloads to eliminate latency.
4. **Resilience to Transient Failures:** Under $20\%$ transient routing dropouts (Setting C), ECG-Reset shields the state space, providing a massive **+5.14%** absolute accuracy gain ($73.97\% \pm 0.39\%$ vs. $68.79\% \pm 0.44\%$ for open-loop ChemMerge).
5. **Mitigation of Confident Router Bias:** Under systematic routing bias (Setting D), open-loop stateful kinetics suffer from "state-locking" (accuracy drops to $\sim 68.5\%$). RASC dynamically overrides biased feedforward updates, steering the system back to the correct task manifold and achieving **73.59% ± 0.39%** accuracy (a **+5.32%** gain over the best stateful heuristic).
6. **Efficiency:** ET-L-ARC collapses L-ARC's latency overhead from $133.31\%$ to just **99.85%** relative overhead ($0.06$ ms absolute overhead per sample) under clean serving, making it highly practical.

---

## Explicitly Claimed Contributions (with Evidence)
1. **Stateful Serving Framework (ECG-Kinetics):** Proposal of ECG-Reset to shield continuous-depth model ensembling. *Evidence:* Table 2 (Setting C results showing +5.14% accuracy gain when ECG-Reset is active).
2. **Control-Theoretic Feedback Formulation:** Constructing a formal Lyapunov stability framework analyzing representation error across depth. *Evidence:* Derivation of the candidate Lyapunov function $V$ (Section 3.3) and Proposition 3.1.
3. **Analytical Dissipation Guard:** Deriving a closed-loop controller that regulates the feedback rate on-the-fly to ensure dissipative updates. *Evidence:* Derivations in Section 3.3, Theorem 3.3 proving unconditional non-negativity of the dissipation coefficient $A$, and Theorem 3.4 providing a Lagrange remainder bound on Taylor linearization errors.
4. **Feedback-Driven State Correction (RASC):** Introducing a dual-loop control mechanism to override systematic router bias. *Evidence:* Section 3.7, and Table 3 (Setting D results showing +5.32% accuracy gain over Decay-ChemMerge, with t-statistic 17.9183, $p = 0.0000$).
5. **Rigorous Empirical Benchmarking & Scientific Transparency:** Evaluating L-ARC in the 14-layer Coordinate Sandbox (ICS) across 10 random seeds, reporting latency profiling, paired t-tests, and gating rates. *Evidence:* Detailed sweeps (Figures 1, 2, 3), paired t-tests (reported p-values), and a small-scale pilot study on LLaMA-3-8B validating the coordinate alignment assumptions on a real-world transformer backbone.
