# 1. Summary of the Paper

## Main Topic and Approach
This paper introduces **ThermoMerge** (Thermodynamic Test-Time Diffusion for Synergistic Model Merging), a physics-inspired optimization framework for unsupervised test-time model merging. Traditional model merging combines specialized fine-tuned expert checkpoints into a single multi-task model without expensive retraining. Recent test-time adaptive methods (such as AdaMerging and SyMerge) optimize merging coefficients ($\Lambda$) and task-specific classification heads ($\Theta^{tr}$) using unlabeled downstream samples. 

The authors argue that these existing adaptive methods are severely bottlenecked by their reliance on **deterministic optimizers** (e.g., Adam or SGD). They claim that under task interference and parameter conflicts, the joint test-time proxy loss landscape is highly non-convex, characterized by severe task interference, high-frequency ripples, and sharp, sub-optimal local basins that permanently trap deterministic optimizers. 

To overcome this, ThermoMerge models the parameter configuration as a physical system undergoing **thermodynamic crystallization** (transitioning from a high-entropy disordered state of chaotic independent experts to a highly ordered, low-energy crystalline multi-task state). This physical process is implemented using **Stochastic Gradient Langevin Dynamics (SGLD)** guided by an **exponential Simulated Annealing cooling schedule**. During the early "hot phase," temperature-scaled Gaussian noise is injected into the updates to allow parameters to hop over energy barriers. As the temperature exponentially cools, the system enters a "cold phase," settling and "freezing" into a flat global minimum basin.

To address the severe dimensionality mismatch between low-dimensional merging coefficients (e.g., $d_{\Lambda} \approx 10^1$) and high-dimensional classification heads (e.g., $d_{\Theta} \approx 10^5$), the paper introduces **Dimensionality-Scaled Langevin Noise (DSLN)**. DSLN scales the coordinate-wise Langevin noise standard deviation inversely with the square root of the parameter group's dimension ($1/\sqrt{d_j}$). This scale factor ensures that the expected total thermodynamic kinetic energy injected into each parameter group is invariant to its dimensionality, preventing high-dimensional classification features from being destroyed by excessive thermal agitation (referred to as high-dimensional noise catastrophe).

---

## Key Findings
1. **Synthetic 1D Simulation:** On a hand-crafted 1D non-convex loss landscape explicitly designed with sinusoidal ripples and a sub-optimal local trap, ThermoMerge successfully escapes the local basin across 10 random seeds. It achieves a **56.7% reduction in final proxy loss** and a **65.0% reduction in generalization variance** (indicating a flatter and more robust minimum) compared to deterministic joint optimization (SyMerge).
2. **Phase Transition Signature:** By numerically integrating the Boltzmann distribution over the 1D landscape, the authors identify a sharp peak in the specific heat capacity ($C_v$) at a critical temperature $T_c \approx 0.02$, which they claim serves as the definitive thermodynamic signature of physical parameter crystallization.
3. **DSLN Validation:** In a multi-dimensional simulation, the proposed DSLN formulation successfully prevents high-dimensional noise catastrophe. As the classifier dimension scales up to $100,000$, unscaled SGLD collapses due to feature destruction, while DSLN maintains low classifier loss and allows the merging coefficients to explore and escape local traps.
4. **Deep Learning Validation (MLP and LoRA):** The framework is evaluated on lightweight Multi-Layer Perceptrons (MLPs) and low-rank adapters (LoRA PEFT) on three image classification datasets: MNIST, FashionMNIST, and KMNIST. The authors report that our Dimensionality-Scaled Langevin Noise successfully stabilizes SGLD, achieving competitive average accuracies (e.g., $84.46\% \pm 0.59\%$ on FashionMNIST and $80.37\% \pm 0.24\%$ on KMNIST for MLP; $78.41\% \pm 1.67\%$ on FashionMNIST for LoRA). Under OOD Gaussian noise corruption ($\sigma=0.25$), ThermoMerge shows competitive robustness.

---

## Explicitly Claimed Contributions and Accompanying Evidence
*   **A Novel Thermodynamic Framing:** Casts test-time model merging as a physical crystallization process and task alignment as the minimization of free energy.
    *   *Evidence:* Qualitative trajectory plots on a 1D non-convex loss landscape (Figure 1).
*   **The ThermoMerge Algorithmic Framework:** Combines preconditioned SGLD (Adam-SGLD) with Simulated Annealing exponential cooling.
    *   *Evidence:* Algorithm 1, qualitative trajectories (Figure 1), and step-by-step MNIST loss curves demonstrating the "hot" and "cold" phases (Figure 4).
*   **Dimensionality-Scaled Langevin Noise (DSLN):** Scaled noise variance inversely proportional to dimension ($1/d_j$) to prevent feature vaporization.
    *   *Evidence:* Comparative simulation results across varying classifier dimensions up to $100,000$ (Table 5) and ablation of weight-bias joint group scaling vs. separate tensor scaling.
*   **Rigorous Multi-Dataset Validation:** Evaluation on MLP and LoRA setups across MNIST, FashionMNIST, and KMNIST under clean and OOD corrupted test-time streaming.
    *   *Evidence:* Quantitative accuracy tables (Table 7, 8, 10, 11) comparing ThermoMerge against training-free and test-time adaptive baselines.
*   **Comprehensive Ablation & Phase Transition Analysis:** Systematic sweep of hyperparameters ($T_0, \gamma$) and numerical integration of Boltzmann thermodynamic variables.
    *   *Evidence:* Hyperparameter grid search table (Table 3), Specific Heat capacity peak plotting (Figure 3), and empirical sensitivity analysis of the scaling factor $\alpha$ (Table 4).
