# 4. Experimental Design and Validation Check

## Quality of Experimental Design & Baselines
While the paper is theoretically strong, the experimental validation is heavily constrained by an artificial and simplified evaluation setup:

### 1. The "Projected Representation Sandbox" is Highly Artificial
The authors evaluate their method inside a simulated sandbox:
- **Tiny Architecture:** The 14-layer residual MLP with width 64 (approx. 63,000 active parameters) is extremely small by modern deep learning standards. Real-world model merging is typically applied to models with millions or billions of parameters (e.g., ResNets, ViTs, LLMs).
- **Manifold Degradation via JL Projections:** Raw images (e.g., CIFAR-10, SVHN) are flattened and projected into a 192-dimensional space using random Johnson-Lindenstrauss matrices. This random projection severely degrades the underlying spatial structures and features of the image manifolds.
- **Low Baseline Performance:** Due to the degraded features and the tiny MLP network, the performance of the individual expert models is extremely poor:
  - MNIST Expert: **78.81%** (standard CNNs easily achieve >99%)
  - FashionMNIST Expert: **73.39%** (standard models easily achieve >90%)
  - CIFAR-10 Expert: **24.72%** (random guess is 10%; standard CNNs achieve >90%)
  - SVHN Expert: **18.24%** (random guess is 10%; standard CNNs achieve >95%)
  - Joint Expert Mean: **48.79%**
- **Generalization Limitation:** It is highly questionable whether the empirical findings (e.g., the flatness of the loss landscape, the relative performance of isotropic vs. non-isotropic priors, and the SWA equivalence) generalize to real-world, high-performance vision backbones or LLMs. Although Appendix A provides a "scaling blueprint," the paper contains no actual experiments on large-scale models.

### 2. Inappropriate Baselines for Tiny Architectures
The paper compares against standard post-hoc merging baselines such as **Ties-Merge** and **DARE-Merge**:
- Ties-Merge and DARE-Merge are designed for large-scale, high-capacity networks where dropping or pruning parameters (e.g., 80% sign pruning or 10% dropout) does not catastrophically disrupt representation learning.
- In a tiny width-64 MLP, sequential representation coordinates are dense and highly fragile. Applying Ties-Merge or DARE-Merge to such a small network results in extreme representation collapse (Ties-Merge drops to **29.68%**, and DARE-Merge achieves **33.24%**, both underperforming the Static Uniform baseline of **33.57%**).
- Comparing a continuous optimization technique to baselines that catastrophically fail on the chosen architecture represents a weak baseline comparison, artificially inflating the relative advantages of the proposed method.

---

## Critical Discrepancies and Inconsistencies

By analyzing the underlying code and comparing the results in the paper, we identified two critical discrepancies between the main experiments (Table 1) and the Few-Shot Calibration Scarcity Sweep (Section 4.3):

### 1. Regularization Hyperparameter Inconsistency
- In the main experiment (Table 1), the regularization coefficient is fixed at a default value of **$\lambda_{\text{PAC}} = 0.010$**.
- In the Few-Shot Calibration Scarcity Sweep (Section 4.3), the regularization coefficient is dynamically scaled as **$\lambda_{\text{PAC}} = 0.120 / M$**, where $M$ is the number of samples per task.
- When $M = 10$, this dynamic coefficient is **$0.012$** (not $0.010$).
- This hyperparameter inconsistency means that the $M = 10$ results in the scarcity sweep are run with a different regularization pressure than the corresponding Table 1 results, preventing a direct and consistent comparison.

### 2. Sequential Data Drawing & Sample Drift
The `RealWorldSandbox` uses sequential pointer tracking (`drawn_indices`) to draw disjoint subsets of samples from the consolidated dataset pool.
- In `run_experiments.py`, the scarcity sweep is executed *after* the main experiments have finished.
- Calling `sandbox.generate_data(M)` inside the scarcity sweep draws a **completely different, disjoint subset of calibration samples** than the main Table 1 experiment's calibration split (even when $M = 10$).
- Since $M = 10$ is extremely small (only 40 samples total across 4 tasks), this data drift/sample discrepancy introduces non-trivial statistical variations.
- This explains why the results reported for $M=10$ in Section 4.3 and `scarcity_results.json` differ from the Table 1 results in `results.json`:
  - **Offline Unconstrained:** **36.09 $\pm$ 2.53%** (Table 1) vs. **36.17 $\pm$ 2.44%** (Section 4.3)
  - **Isotropic PAC-Bayes:** **36.09 $\pm$ 2.23%** (Table 1, Randomized Ensemble) vs. **36.69 $\pm$ 2.65%** (Section 4.3)
  - **FIM PAC-Bayes:** **36.07 $\pm$ 2.17%** (Table 1, Randomized Ensemble) vs. **36.62 $\pm$ 2.64%** (Section 4.3)

These discrepancies reveal a minor lack of experimental hygiene, as the results in the scarcity sweep are evaluated on different data splits and with different hyperparameters than the main table results.
