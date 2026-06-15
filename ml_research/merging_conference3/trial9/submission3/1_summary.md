# 1. Summary of the Paper

This paper addresses the critical issue of **sequential routing jitter** and **transductive overfitting** in sequential dynamic model ensembling and model merging over deep neural networks. In multi-task model serving scenarios, routing intermediate activation representations dynamically across layers creates a complex non-linear feedback loop. Left unregularized, this feedback loop causes gating coefficients to oscillate violently from layer to layer (routing jitter), degrading downstream classification capabilities and making the router highly vulnerable to transductive overfitting, especially under severe calibration data scarcity (e.g., 16 samples per task).

## Key Challenges Addressed
1. **Sequential Routing Jitter:** The routing decision at layer $l$ is a function of the intermediate feature $h^{(l-1)}$, which itself depends on previous routing decisions. This iterative coupling acts as a chaotic discrete-time dynamical system, yielding violent layer-to-layer oscillations in gating coefficients.
2. **Lack of Mathematical Rigor in Baselines:** Prior efforts (such as ChemMerge) rely on continuous-time chemical kinetics heuristics to empirically smooth trajectories, but they lack formal convergence guarantees, theoretical parameter bounds, or stability proofs.
3. **Severe Transductive Overfitting:** Under extreme data-scarcity, learned parametric routers overfit heavily to local coordinates, causing severe performance degradation.

## Proposed Solutions (CR-Router)
1. **Discrete-Time Dynamical System Formulation:** The authors model feedforward feature-coefficient propagation as an iterative mapping on a Banach space.
2. **Lipschitz Bound on the Joint Mapping:** Under Banach's Fixed-Point Theorem, a novel Lipschitz bound $L_{T_l}$ is derived for the joint layer-wise representation-routing map.
3. **Contraction-Regularized Objective:** The authors design a theoretically sound objective function that regularizes the Frobenius norm of the routing projection matrices (as an analytical upper bound on the spectral norm) and penalizes the inverse routing temperature parameters to enforce a strict contraction mapping ($L_{T_l} < 1$).
4. **Update-Space Quasi-Contraction:** To preserve the pre-trained capacity of frozen backbones (where a strict contraction would require modifying residual paths in a way that degrades performance), the authors formalize a theoretical relaxation ($L_{U_l} < \epsilon$). This stabilizes gating trajectories and prevents routing jitter without altering the frozen base model.
5. **Adaptive Test-Time Temperature Annealing:** A clever inference-time sharpening strategy ($\gamma_{\text{scale}} \le 1.0$) that decouples training-time optimization stability from test-time representation sharpness, successfully resolving the "expert dilution" dilemma.
6. **Centroid-Based Routing Warm-Starting:** An elegant initialization strategy that warm-starts routing parameters using task-specific centroids of the scarce calibration split, guiding early gradient steps directly into stable, task-aligned attraction basins.
7. **Label-Free Online Heuristics:** Three practical online diagnostics (Gating Depth-Variance, Shannon Gating Entropy, and Running Gating Lipschitz Bound) that monitor stability during calibration, allowing hyperparameter tuning without labeled validation data.

## Main Empirical Findings
- **High-Fidelity Sandbox Evaluations (Experiments 1 & 2):** Evaluated inside a 14-layer, 192-dimensional Analytical Coordinate Sandbox across 10 random seeds.
  - *Orthogonal Subspaces (Exp 1):* CR-Router achieves a stellar classification accuracy of **53.35% ± 3.84%** and routing accuracy of **65.10% ± 2.70%** (a **+18.62%** absolute improvement over unregularized routing).
  - *Overlapping Subspaces (Exp 2):* Uniform Merging collapses to 27.48% due to cross-talk, whereas CR-Router achieves **43.48% ± 4.70%** classification accuracy, outperforming unregularized routing by **12.86%** absolute and validating its theoretical predictions.
- **Real-World Vision Embedding Manifolds (Experiment 3):** Evaluated on pre-trained ResNet18 PCA-projected representations of MNIST, Fashion-MNIST, KMNIST, and USPS.
  - Uniform Merging collapses to 7.70% due to cross-talk.
  - CR-Router achieves **53.70% ± 2.37%** classification accuracy and **84.22% ± 3.09%** routing accuracy, outperforming the heuristic **L2-Fixed Router** by **+6.37%** absolute classification accuracy.
- **Adaptive Test-Time Annealing:** Decoupling stability and sharpness yields a massive performance boost, improving CR-Router's classification accuracy to **62.45% ± 2.98%** (+8.90% absolute) at $\gamma_{\text{scale}} = 0.10$.
- **Serving Efficiency Benchmarks:** CR-Router reduces forward-pass latency on CPU to **25.34 ms** (a **33.7% latency reduction** over non-parametric SABLE) and increases throughput to **15,785.1 samples/s** (a **1.51x speedup** over SABLE and **1.58x speedup** over ChemMerge), validating its computational practicality.
