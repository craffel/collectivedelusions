# Evaluation Task 2: Novelty and Technical Delta Check

## Delta from Prior Work
The proposed method, **ThermoMerge**, is positioned at the intersection of test-time model merging and stochastic optimization. To critically assess its novelty, we compare it against its closest predecessors:

1. **AdaMerging (ICLR 2024):** Introduces unsupervised test-time adaptation for learning merging coefficients ($\lambda$) by minimizing entropy on unlabelled test data.
2. **SyMerge (ICML 2026):** Addresses the stability issues of entropy minimization by introducing an **Expert-Guided Self-Labeling** proxy objective. SyMerge jointly optimizes the merging coefficients and a single task-specific classifier layer using standard deterministic optimization (SGD/Adam).

### The Technical Delta of ThermoMerge:
The exact optimization formulation, the expert-guided self-labeling proxy objective, and the parameter parameterization (optimizing a task-specific head along with the merging coefficients) are **identical to SyMerge**. 

The actual technical delta of ThermoMerge consists of:
* Replacing SyMerge's deterministic gradient-based optimizer (SGD/Adam) with **Stochastic Gradient Langevin Dynamics (SGLD)**.
* Introducing an **exponential Simulated Annealing schedule** to cool down the Langevin temperature over adaptation steps.
* Proposing **Dimensionality-Scaled Langevin Noise (DSLN)** to scale the added coordinate-wise noise standard deviation by $1/\sqrt{d_j}$, where $d_j$ is the dimension of the parameter group $j$.
* Engineering preconditioning and noise buffer pre-allocation heuristics to reduce SGLD overhead.

---

## Analysis of Core Concepts
* **Simulated Annealing and Langevin Diffusion:** Using Langevin noise (SGLD) and simulated annealing (temperature cooling schedules) to escape local minima in non-convex neural network loss landscapes is a classic, well-established technique dating back to the 1980s and 1990s. Its application in deep learning optimization, Bayesian neural networks, and Langevin Monte Carlo is standard. Transferring these existing, off-the-shelf optimization tools to the loss landscape of test-time model merging represents a straightforward application of classic optimization rather than a conceptual breakthrough in machine learning theory.
* **Dimensionality-Scaled Langevin Noise (DSLN):** The paper’s most specific adaptation is DSLN. The authors observe that adding unscaled isotropic noise to high-dimensional parameter groups (like a classification head with $d \approx 10^5$) leads to representational collapse ("thermal destruction"), whereas low-dimensional merging coefficients ($d \approx 3$) require aggressive exploration. Scaling noise variance inversely with dimension $d_j$ ensures that the total injected kinetic energy (Euclidean norm of the noise vector) remains invariant across layers. While this is a highly sensible and useful engineering heuristic for multi-scale parameters, scaling noise or learning rates based on dimensionality or gradient variance has strong precedents in adaptive optimization (e.g., RMSprop, Adam, and preconditioned SGLD). Thus, DSLN represents an incremental, though practical, engineering refinement.

---

## Characterization of Novelty
The novelty of this paper is best characterized as **incremental and application-driven**. 

* **Conceptual Novelty (Low):** The theoretical framing of model merging as "crystallization" is primarily metaphorical; the underlying mathematics are standard SGLD and Simulated Annealing applied directly to the pre-existing SyMerge objective.
* **Algorithmic Novelty (Moderate-Low):** Replaces a deterministic optimizer with a stochastic one. The only bespoke algorithmic addition is DSLN, which is a straightforward scaling heuristic to resolve a known issue with high-dimensional noise.
* **Practical Novelty from a Practitioner’s Viewpoint:** For a practitioner, introducing SGLD, temperature scheduling, and dimensionality scaling adds considerable implementation complexity, more hyperparameters to tune ($T_0$, $\gamma$, annealing steps), and computational steps (generating and scaling random noise at every iteration). In practice, adding this complexity is only justified if there is a substantial, reproducible performance delta on real-world large-scale models. As detailed in the experiment check, this delta is missing on deep neural networks.
