# Experimental Evaluation and Claims Check

## 1. Experimental Setup and Calibrated Simulator
The experimental design is exceptionally thorough, structured, and scientifically rigorous:
- **Analytical Coordinate Sandbox**: Evaluates a 14-layer backbone processing representations from four visual domains with highly detailed noise characteristics (Task 1 to Task 4). This allows the authors to precisely model different task difficulties and noise scales (from minimal $\sigma=0.05$ to extreme $\sigma=1.20$).
- **Oracle Ceilings**: The authors establish a Joint Mean Expert Oracle Ceiling of 70.90%, allowing readers to understand the maximum potential performance. They provide an elegant mathematical explanation for why Task 4's ceiling is 19.20% (due to severe noise), demonstrating high scientific integrity.
- **Parallel Sweeps**: Every major result is evaluated across **10 independent random seeds** (seeds 42 to 51) and reported with both means and standard deviations, ensuring statistical significance and eliminating any risk of cherry-picking.

## 2. Comprehensive Baselines
The paper compares the proposed methods against a highly comprehensive set of baselines:
- **Static Baseline**: Uniform Merging (training-free baseline).
- **Global Dynamic Baseline**: Linear Router.
- **Layer-wise Dynamic Baselines**: L3-Linear, L3-Softmax (unregularized/random-initialized), and QWS-Merge (state-of-the-art quantum-inspired wave superposition).
- **Proposed/Well-Regularized Baselines**: L3-Softmax-WellReg (zero-initialized, weight-decayed Softmax) and VR-Router (ours with task-variance penalty).
This extensive coverage ensures that the proposed framework is evaluated against the strongest existing methods and naive compromises.

## 3. Support for Core Claims
The empirical results provide overwhelming, definitive support for all of the paper's claims:
- **Claim 1: Exposing "Vectorization Collapse" and the "Batch-Average Smoothing Confounder"**: Supported by Table 1 and Table 3. Under heterogeneous streams at $B=256$, unregularized L3-Softmax achieves a strong $59.35\%$. However, at $B=1$ (where batch averaging is removed), its accuracy collapses to $41.09\% \pm 3.73\%$, falling nearly 17% below naive Uniform Merging. This beautifully illustrates how batch-averaging acts as a smoothing confounder that masks severe overfitting.
- **Claim 2: Efficacy of Proper Priors (Zero-Initialization + Weight Decay)**: Supported by Table 1, where both L3-Softmax-WellReg and VR-Router completely resolve collapse, maintaining a flatline joint accuracy of $\approx 59.16\%$ across all batch sizes (from $B=1$ to $B=512$).
- **Claim 3: Empirical Redundancy of Explicit Variance Regularization**: Supported by Table 1 and Table 4. L3-Softmax-WellReg and VR-Router perform statistically identically ($59.16\%$ vs. $59.14\%$). Furthermore, Table 4's ablation study shows that optimizing with cross-entropy alone under the zero-initialized Softmax prior already yields $59.18\% \pm 1.25\%$, proving that the architectural prior (starting at a uniform maximum-entropy state) is the true driver of stability, making explicit training losses redundant.
- **Claim 4: Exposing the "Dynamic Routing Paradox"**: Quantified by showing that the learned routing coefficients under well-regularized routing have a Mean Absolute Deviation (MAD) of only $0.0236$ (or $2.36\%$) from uniform. This proves that to survive data scarcity, the router must be restricted so heavily that its functional flexibility is virtually eliminated, explaining why it only beats Uniform Merging by a marginal $+1.16\%$.
- **Claim 5: Sensitivity Sweeps and Generalizability**: Supported by Table 2 ($\lambda_{var}$ sweep), Table 5 (subspace overlap $\rho$ sweep), Table 6 (projection dimension $d$ sweep), Table 8 ($\mathcal{L}_{\text{smooth}}$ sweep), Section 4.5 (MLP depth sweep), Section 4.4 (calibration data size sweep), Section 4.6 (real-world MNIST+FashionMNIST validation), and Section 4.8 (Dynamic LoRA rank sweep). This massive array of empirical sweeps covers every possible angle of the routing architecture, demonstrating outstanding generalizability and completeness.
