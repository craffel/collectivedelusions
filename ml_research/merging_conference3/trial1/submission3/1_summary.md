# 1. Summary of the Paper

## Overview and Problem Statement
The paper addresses the challenge of test-time model merging, where multiple fine-tuned task-specific expert models are consolidated into a single unified multi-task model at test-time without requiring expensive retraining. Current state-of-the-art adaptive merging methods, such as AdaMerging and SyMerge, rely on deterministic gradient-based optimization (like SGD or Adam) to optimize layer-wise merging coefficients and task-specific classification heads.

The authors identify a fundamental bottleneck: under severe task interference and parameter conflicts, the joint test-time proxy loss landscape is highly non-convex, characterized by high-frequency ripples and sharp, sub-optimal local basins. Deterministic optimizers are mathematically guaranteed to get trapped in these local minima if initialized near them, leading to poor multi-task trade-offs and weak out-of-distribution (OOD) generalization.

## Proposed Methodology (ThermoMerge)
To overcome this limitation, the paper proposes **ThermoMerge** (Thermodynamic Test-Time Diffusion for Synergistic Model Merging), which reframes test-time model adaptation as a physical thermodynamic crystallization process. The parameters are modeled as transitioning from a disordered, high-entropy state (chaotic independent experts) to a highly ordered, low-entropy crystalline state (globally aligned, synergistic multi-task model).

ThermoMerge's core components include:
1. **Stochastic Gradient Langevin Dynamics (SGLD)**: Injects temperature-scaled Gaussian noise into gradient updates to allow the parameters to hop over energy barriers and escape sharp local traps.
2. **Simulated Annealing Exponential Cooling Schedule**: Automatically decays the temperature from a high value (the "Hot Phase" for global exploration) to zero (the "Cold Phase" for deterministic convergence).
3. **Dimensionality-Scaled Langevin Noise (DSLN)**: Scales the coordinate-wise Langevin noise standard deviation inversely with the square root of each parameter group's dimension ($1/\sqrt{d_j}$). This prevents high-dimensional parameters (like classifiers) from experiencing "noise catastrophe" and representational vaporization, while allowing low-dimensional parameters (like merging coefficients) to maintain high exploratory energy.
4. **Layer-wise Functional Parameter-Group Scaling**: Groups weights and biases of a functional layer together into a single parameter group to resolve thermodynamic imbalance (since biases have much smaller dimensions and would otherwise be perturbed too heavily).
5. **Predictive Agreement and Entropy Safeguard**: An unsupervised early-stage safety valve that monitors prediction entropy and teacher agreement on test-time streaming batches to automatically trigger "Emergency Quenching" (halving temperature scale) if representational collapse is detected.

## Key Empirical Findings
1. **1D Non-Convex Physical Simulation**: On a simulated 1D rugged loss landscape representing task interference, ThermoMerge achieves a **56.7% reduction in final proxy loss** and a **65.0% reduction in generalization variance** compared to deterministic optimization (SyMerge), successfully escaping the sharp local trap.
2. **Deep Learning Validation (MNIST, FashionMNIST, KMNIST)**:
   - On MLPs, ThermoMerge achieves highly stable and competitive multi-task accuracies (e.g., $84.46\% \pm 0.59\%$ on FashionMNIST and $80.37\% \pm 0.24\%$ on KMNIST), matching or slightly outperforming SyMerge and outperforming standard active flat-minima baselines like SAM and SWA.
   - On out-of-distribution (OOD) domain shifts (Gaussian noise corruption), ThermoMerge exhibits outstanding robustness and extremely low generalization variance compared to deterministic and sharpness-aware baselines.
3. **PEFT/LoRA Model Merging**: On low-rank parameter-efficient fine-tuning spaces, ThermoMerge delivers a significant boost (e.g., **+1.11% OOD accuracy improvement** on FashionMNIST over SyMerge) and prevents representation collapse, demonstrating that SGLD's global exploration is particularly vital when optimization is constrained to narrow low-dimensional manifolds.
4. **Thermodynamic Phase Transition Signature**: Numerical integration of the Boltzmann distribution over the loss landscape reveals a sharp specific heat capacity peak ($C_v$) at a critical temperature $T_c \approx 0.02$, providing physical evidence of a genuine parameter crystallization phase transition.
5. **Computational Complexity & Latency**: Detailed GPU wall-clock latency profiling confirms that ThermoMerge adds negligible overhead ($\approx 1.5\% - 4.9\%$) compared to SyMerge, while requiring half the F-Evals/G-Evals of SAM.
