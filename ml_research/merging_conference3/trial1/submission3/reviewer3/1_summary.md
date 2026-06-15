# 1. Summary of the Paper

## Main Topic and Motivation
The paper addresses the challenge of **test-time model merging**, which aims to unify multiple task-specific expert models into a single multi-task foundation model during inference using unlabeled target task samples. Traditional merging approaches rely on static, training-free heuristics (e.g., Task Arithmetic, Ties-Merging), which often suffer from severe parameter interference and task conflict. Recent test-time adaptive merging frameworks (e.g., AdaMerging, SyMerge) optimize a self-labeling proxy loss at test-time to find optimal merging coefficients and task-specific parameters.

The authors identify a critical bottleneck in existing test-time adaptation methods: they rely on standard deterministic optimization schemes (like SGD or Adam). The authors argue that the joint multi-task proxy loss landscape of test-time model merging is highly non-convex, featuring high-frequency ripples and sharp, sub-optimal local basins that trap deterministic optimizers. Consequently, deterministic adaptation is highly sensitive to initialization and struggles with poor out-of-distribution (OOD) generalization.

## Proposed Approach: ThermoMerge
To overcome these limitations, the paper introduces **ThermoMerge** (Thermodynamic Test-Time Diffusion for Synergistic Model Merging), which reframes test-time model merging as a thermodynamic crystallization process. The parameter configuration is treated as a physical system transitioning from a disordered, high-entropy state (chaotic independent experts) to a highly ordered, synergistic crystalline multi-task state (optimal multi-task fusion).

The key technical components of ThermoMerge are:
1. **Stochastic Gradient Langevin Dynamics (SGLD)**: Injects temperature-scaled Gaussian noise into the updates of both the merging coefficients and task-specific parameters to enable escape from local sub-optimal basins.
2. **Simulated Annealing Cooling Schedule**: Governs the Langevin temperature using an exponential decay schedule ($T_t = T_0 \cdot \gamma^t$) to transition from global exploration ("hot phase") to precise local convergence ("cold phase").
3. **Dimensionality-Scaled Langevin Noise (DSLN)**: Scales the coordinate-wise Langevin noise standard deviation inversely with the square root of the parameter group's dimension ($d_j$):
   $$\sigma_j = \sqrt{\frac{2 \eta_j T_t}{d_j}}$$
   This ensures that the total expected thermal kinetic energy remains invariant to parameter dimension, preventing high-dimensional classification heads from "boiling" and destroying pre-trained features, while allowing low-dimensional merging coefficients to explore aggressively.
4. **Curvature-Aware Preconditioning (Adam-SGLD)**: Integrates SGLD updates with Adam preconditioning to scale the gradient steps and noise by local landscape geometry.

## Key Findings and Claims
* **Synthetic 1D Simulation**: On a designed non-convex 1D physical simulation landscape, ThermoMerge is claimed to consistently escape a sharp local trap (around $\Lambda = 0.2$) and converge to a wide global minimum (around $\Lambda = 0.6$), achieving a **56.7% reduction in final proxy loss** and a **65.0% reduction in generalization variance** compared to SyMerge.
* **Phase Transition Signature**: By numerically integrating the Boltzmann distribution over the 1D landscape, the authors identify a sharp peak in the Specific Heat capacity ($C_v$) at a critical temperature $T_c \approx 0.02$, which they claim represents the physical signature of parameter crystallization.
* **Deep Learning Class-Splitting Validation (MLP & LoRA)**: On actual neural networks trained in a class-splitting setup across three datasets (MNIST, FashionMNIST, KMNIST), the authors claim that ThermoMerge with DSLN stabilizes joint adaptation and achieves highly stable and competitive multi-task accuracies (e.g., $84.46\% \pm 0.59\%$ on FashionMNIST and $80.37\% \pm 0.24\%$ on KMNIST), matching or closely matching state-of-the-art deterministic joint adaptation.
* **Out-of-Distribution (OOD) Robustness**: Under severe Gaussian noise corruption ($\sigma=0.25$), ThermoMerge is claimed to achieve competitive accuracies with lower generalization variance than SyMerge, showing robustness to domain shifts.

## Explicitly Claimed Contributions (with Evidence in Paper)
1. **A Novel Thermodynamic Framing**: Casts model merging and task alignment as the minimization of physical free energy. (Evidence: Section 3.1--3.3, 4.5).
2. **The ThermoMerge Algorithmic Framework**: Combines preconditioned SGLD with a Simulated Annealing exponential cooling schedule. (Evidence: Section 3.3, 3.6, Algorithm 1).
3. **Dimensionality-Scaled Langevin Noise (DSLN)**: Scales noise variance inversely with parameter dimensions to stabilize joint adaptation across multi-scale parameter groups. (Evidence: Section 3.4, 4.6).
4. **Rigorous Multi-Dataset Validation**: Evaluates ThermoMerge on both a 1D synthetic landscape and deep neural networks (MLP & LoRA) across MNIST, FashionMNIST, and KMNIST. (Evidence: Section 4.2, 4.7, 4.8).
5. **Comprehensive Ablation and Phase Transition Analysis**: Maps hyperparameters and analyzes the Boltzmann distribution to find Specific Heat capacity peaks and critical crystallization temperature $T_c \approx 0.02$. (Evidence: Section 4.3, 4.5).
