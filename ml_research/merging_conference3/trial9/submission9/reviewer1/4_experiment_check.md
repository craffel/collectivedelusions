# 4. Experiment Check

An empirical evaluation must be critical, comprehensive, and honest. This report evaluates the experimental setup, baseline coverage, datasets, and whether the empirical results of **PAC-Kinetics** support its claims.

---

## 1. Experimental Setup and Baselines
The authors construct a highly thorough and systematic experimental design:

* **Analytical Coordinate Sandbox (ICS)**: To precisely evaluate serving trajectories under varying temporal mixing structures, the authors simulate a multi-task serving environment with $K=4$ task experts in a 14-layer, 192-dimensional representation space. They evaluate under two spatial configurations: (1) **Orthogonal Manifolds** (disjoint task signatures) and (2) **Overlapping Manifolds** (overlapping signatures simulating severe representation-space interference and correlated tasks). They test under two sequential streaming patterns: (1) **Homogeneous Stream** (long blocks of same tasks) and (2) **Heterogeneous Stream** (rapid task switches).
* **Comprehensive Baseline Coverage**: The paper compares PAC-Kinetics against seven relevant baselines:
  1. *Expert Oracle*: The theoretical ceiling assuming 100% routing accuracy.
  2. *Uniform Merging (Static)*: Statically averaging all expert weights ($\alpha_k = 1/K$).
  3. *SABLE (Raw) / SPS-ZCA*: Stateless nearest-centroid routing with unregularized coordinates.
  4. *SABLE (SEP)*: Stateless routing incorporating Subspace Energy Projection without unit-normalization.
  5. *Stateless PAC-ZCA*: State-of-the-art temperature-scaled Gibbs routing.
  6. *Heuristic ChemMerge*: Stateful chemical kinetics router with static, heuristic ODE parameters.
  7. *Stateful ERM*: Stateful kinetics router optimized via standard Empirical Risk Minimization without PAC-Bayesian regularizers.
  
  This coverage is excellent, ensuring that both stateless, stateful, heuristic, and unregularized learned baselines are evaluated.

---

## 2. Main Empirical Results and Evidence
The results strongly support the paper's core claims:

* **Massive Jitter Reduction**: In stable Homogeneous streams, PAC-Kinetics achieves a massive **11.2$\times$** reduction in routing jitter on orthogonal streams (from 0.0697 to 0.0062) and up to **16.0$\times$** on overlapping streams relative to stateless SABLE, matching the absolute smoothness of the Oracle and ChemMerge.
* **Accuracy-Stability Pareto Frontier**: Under homogeneous streams, PAC-Kinetics matches the Expert Oracle's performance (achieving **95.03%** on orthogonal and **95.07%** on overlapping manifolds), outperforming static ChemMerge. Under heterogeneous streams, PAC-Kinetics achieves **92.35%** (orthogonal) and **92.90%** (overlapping) accuracy, massively outperforming ChemMerge (which collapses to **70.59%** due to unmitigated lag) and Stateful ERM (which achieves $87.14\%$ and $83.04\%$). This demonstrates the clear benefit of the PAC-Bayesian complexity penalty and the Adaptive Online Kinetics mechanism.
* **Acknowledge the Stateful-Stateless Trade-Off**: The authors honestly discuss that under completely uncorrelated, independent streams, stateful memory is a liability (introducing "inertial drag"). In this setting, stateless PAC-ZCA is optimal ($94.39\%$). PAC-Kinetics is shown to successfully minimize this lag, achieving a high **92.35%** accuracy and providing a robust compromise.

---

## 3. Physical Validation on Real PyTorch Networks and Datasets
To bridge the "simulation gap," the authors perform a physical evaluation using PyTorch and real image datasets (MNIST and Fashion-MNIST):

* **Architecture**: A deep 3-layer Feed-Forward MLP with a shared frozen trunk, a base linear layer blended with two active low-rank LoRA-style adapters, and task-specific classification heads.
* **Homogeneous Results**: PAC-Kinetics achieves **87.61%** representation alignment and **76.40%** classification accuracy, outperforming stateless PAC-ZCA's 71.20% and Uniform Merging's 54.90% (a **+5.20% and +21.50% absolute improvement** respectively). It slashes active routing jitter of PAC-ZCA by **2.59$\times$** (from 0.4891 to 0.1888).
* **Heterogeneous Results**: Under rapid switches, PAC-Kinetics achieves **66.30%** classification accuracy, outperforming Uniform Merging (54.90%) and ChemMerge (58.60%), confirming that the **Adaptive Online Kinetics** successfully suppresses inertial drag in real-world environments.
* **Resolving the Uniform Merging Paradox**: They prove that statically averaging expert weights (Uniform Merging) collapses classification performance to **54.90%** when task representations conflict, whereas dynamic stateful routing is essential.

---

## 4. Verification of Theoretical Discrepancies and Ablations
* **Deterministic Surrogate Gap**: They empirically verify the performance of the randomized router `PAC-Kinetics (Rand)`, which collapses to near-uniform accuracy ($\approx 31\%$-$33\%$). To explain and resolve this, the authors conduct an empirical sweep over smaller perturbation variances ($\sigma_{\text{pert}}^2 \le 0.01$) in Appendix C.4, showing that as the variance decreases, the contractive stability of the linear recurrence is restored, matching the deterministic surrogate.
* **Gated Sequence Models**: They compare against LSTMs and GRUs, proving that high-capacity gated sequence models overfit on short calibration sequences ($T=32$), leading to chaotic internal state trajectories and higher routing jitter (0.4124 and 0.3842 vs. PAC-Kinetics' 0.1384).
* **Systems Scalability**: Latency profiling shows flat **$\approx 10.4$ microseconds** CPU execution latency per sample for fleet sizes $K \le 8$, and vectorized parallel GPU execution latency under **3.5 microseconds** for batch size 128 under $K=8$. This is completely negligible relative to backbone attention layers (typically 10-200ms).
* **Hyperparameter Sensitivity**: They conduct thorough sweeps over prior variance $\sigma_0^2$, calibration length $T$, and fleet size $K$ (up to $K=16$), showing that PAC-Kinetics is highly robust, scales gracefully, and keeps the coupling matrix $W$ well-conditioned (low spectral condition numbers).
* **Non-Negative Constraints Ablation**: They evaluate constraining $W \ge 0$ (literal chemical concentrations), showing it slashes jitter under homogeneous streams but causes a **-8.49% accuracy collapse** on heterogeneous streams. This proves that negative feedback (biochemical inhibition) is essential for low-latency stateful routing.
