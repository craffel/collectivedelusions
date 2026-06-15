# 1. Summary of the Paper

## Main Topic and Approach
This paper addresses the problem of dynamic parameter-efficient adapter serving (e.g., serving multiple LoRAs) on resource-constrained edge devices under heterogeneous streaming workloads. To resolve "routing volatility" (jitter) and "cascading representational drift" in existing stateful ensembling methods, the authors propose **Lyapunov-Stable Active Representation Coupling (L-ARC)**. 

L-ARC is a training-free, continuous-time ensembling framework that uses ordinary differential equations (ODEs) to model and smooth routing weights (expert concentrations) across network layers. Its key differentiator is a closed-loop **Lyapunov Feedback Controller** that dynamically warps hidden representations toward active task-specific centroids. It models the representation similarity error as a system-level candidate Lyapunov function, and derives a local **Dissipation Guard** to calculate sample- and layer-specific feedback rates on-the-fly, ensuring that representation warping is strictly error-decreasing (dissipative) under linearization.

The framework also introduces:
1. **Entropy-Gated Concentration Gating (ECG-Reset):** A mechanism that monitors routing Shannon entropy to freeze continuous ODE kinetics during sensor dropouts or transient failures, preventing memory corruption.
2. **Entropy-Triggered Lyapunov Gating (ET-L-ARC):** An optimization that dynamically evaluates the Dissipation Guard only under moderate routing uncertainty ($0.15 \le H \le 0.95$), collapsing absolute latency overhead under clean workloads.
3. **Representation-Agreement State Correction (RASC):** A dual-loop control mechanism that overrides corrupted feedforward router confidence with representation-space coordinate tracking, resolving state-locking failures under systematic router bias.
4. **Mid-Network Recalibration (MNR):** A multi-anchor centroid strategy designed to prevent late-layer representational drift in extremely deep models.

---

## Key Findings
The authors evaluate L-ARC against several baselines (including stateless SABLE SOTA, nearest-centroid SPS-ZCA SOTA, and open-loop ChemMerge) in a simulated **14-layer Analytical Coordinate Sandbox (ICS)** across 10 random seeds. They also report a small-scale real-world pilot study on LLaMA-3-8B with 100 queries. 

The main empirical findings reported are:
* **Under Static Centroids (Setting A):** Full L-ARC achieves a Joint Mean Accuracy of **74.38% $\pm$ 0.31%** and Semantic Similarity of **0.7937 $\pm$ 0.0059**. SABLE SOTA gets **74.06% $\pm$ 0.33%** accuracy and **0.7590 $\pm$ 0.0090** similarity. The authors note that the difference between full L-ARC and Decoupled ChemMerge ($74.33\% \pm 0.34\%$) is **not statistically significant** under clean workloads ($p = 0.0969$), concluding that active feedback warping is practically redundant under unperturbed serving.
* **Under Transient Failures (Setting C):** SABLE SOTA drops to **71.10% $\pm$ 0.52%** accuracy, while open-loop ChemMerge drops to **68.79% $\pm$ 0.44%** due to stateful memory corruption. L-ARC (Ours, Feedback+ECG-Reset) maintains an accuracy of **73.97% $\pm$ 0.39%** and similarity of **0.7813 $\pm$ 0.0075**. This represents a **+5.14%** accuracy gain over open-loop ChemMerge, primarily driven by ECG-Reset (which alone achieves **73.93% $\pm$ 0.41%**). Active feedback under failures is also statistically insignificant ($p = 0.3443$).
* **Under Confident Router Bias (Setting D):** Stateful kinetics suffer from state-locking, dropping to **68.52% $\pm$ 0.68%** accuracy. RASC-equipped L-ARC overcomes this, achieving **73.59% $\pm$ 0.39%** accuracy and **0.7467 $\pm$ 0.0082** similarity, outperforming open-loop ChemMerge by a statistically significant **+5.07%** ($p = 0.0000$).
* **Computational Efficiency:** Наive L-ARC adds significant latency, but with ET-L-ARC optimization, the latency is reduced to **120.50 ms** for a batch of 1000 samples (a **15%** speedup), which the authors frame as an absolute overhead of $0.06$ ms per sample.

---

## Claimed Contributions and Evidence

1. **Stateful Serving Framework (ECG-Kinetics):** Proposal of ECG-Reset to freeze physical kinetics when routing entropy $H^{(l)} > 0.95$.
   * *Evidence:* Under Setting C (Table 2), adding ECG-Reset shields the kinetics tracker from propagating memory faults, boosting accuracy from $68.79\%$ to $73.93\%$ (a $+5.14\%$ absolute gain).
2. **Control-Theoretic Feedback Formulation:** Proving that the similarity distance error relative to layer centroids acts as a valid, positive semi-definite Lyapunov function across layers.
   * *Evidence:* Section 3.3 derives the discrete Lyapunov difference, proving that the active coupling update is strictly error-decreasing (dissipative) under linearization.
3. **Analytical Dissipation Guard:** An online controller that gates off representation warping ($\eta^{(l)} = 0.0$) when the update is non-dissipative ($A_b^{(l)} \le \theta_G = 0.04$) and scales it adaptively otherwise.
   * *Evidence:* Ablation studies in Section 4.3 show that the Dissipation Guard gates off feedback in 30.45% of updates under clean workloads and up to 54.14% under extreme manifold entanglement, showing self-regulating behavior that preserves semantic similarity.
4. **RASC (Representation-Agreement State Correction):** A closed-loop feedback mechanism that overrides corrupted feedforward router rates with representation-space similarity tracking when routing predictions disagree with physical coordinates.
   * *Evidence:* Under Setting D (Table 3), L-ARC with RASC outperforms ChemMerge and other heuristics, boosting accuracy to $73.59\%$ and resolving the state-locking failure.
5. **Rigorous Empirical Benchmarking & Transparency:** Detailed profiling of latency, statistical significance testing (paired t-tests), and gating rates across settings.
   * *Evidence:* Latency profiles (Section 4.4), paired t-tests (Sections 4.3, 4.5, 4.6), and robustness sweeps (Figures 1, 3) are provided.
