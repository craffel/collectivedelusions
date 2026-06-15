# 3. Soundness and Methodology Evaluation

This file evaluates the technical soundness, appropriateness of methods, clarity, potential mathematical/notational flaws, and reproducibility of the paper.

## Mathematical Formulation and Soundness
The core mathematical formulation of **FlatMerge** is sound and well-grounded:
1. **Subspace Constraint:** The polynomial parameterization of layer-wise blending coefficients $\lambda^l_k(\mathbf{w}_k) = \sum_{j=0}^d w_{k, j} \cdot \left(\frac{l-1}{L-1}\right)^j$ is mathematically clean. It reduces the optimization space from $L \times K$ to $(d+1) \times K$.
2. **Smoothed Objective:** The flatness-aware objective $\mathcal{L}_{\text{smooth}}(\mathbf{W}; X) = \mathbb{E}_{\mathbf{U} \sim \mathcal{S}_D} [\mathcal{L}_{\text{ent}}(\mathbf{W} + \sigma \mathbf{U}; X)]$ is a correct application of randomized smoothing to seek flat valleys in the low-dimensional coefficient space.
3. **Zeroth-Order Estimator:** The central-difference ZO gradient estimator $\hat{\nabla}_{\mathbf{W}}^{\text{ZO}} \mathcal{L}_{\text{smooth}}(\mathbf{W}; X)$ (Eq. 7) is correct and mathematically standard.

## Technical Flaws and Notational Inconsistencies
We identified several notable notational and technical inconsistencies that need to be addressed:
1. **Notational Inconsistency ($\sigma$ vs. $\rho$):** 
   In Section 3.3, the perturbation scale (perturbation radius) is defined and denoted using the symbol **$\sigma$** (e.g., in Eq. 5, Eq. 7, and Algorithm 1). However, in Section 4 (Experiments), the authors repeatedly refer to the perturbation radius using the symbol **$\rho$** (e.g., L164: *"the fixed perturbation radius $\rho = 0.05$"*; L319: *"perturbation radius ($\rho$) on Model II"*; L332: *"sweeps FlatMerge's perturbation radius $\rho \in [0.001, 0.2]$"*). The authors must unify this notation throughout the paper (preferring one symbol, e.g., $\sigma$ or $\rho$, consistently).
2. **Initialization Details of Polynomial Parameters:**
   In Algorithm 1, Line 2, the authors state: *"Initialize: Polynomial parameters $\mathbf{W} = \{\mathbf{w}_k\}_{k=1}^K$ such that blending coefficients $\lambda^l_k = 0.3$ uniformly."* To achieve a uniform coefficient of $0.3$ across all layer depths, the polynomial parameter vector $\mathbf{w}_k = [w_{k,0}, w_{k,1}, \dots, w_{k,d}]^\top$ must be initialized to $[0.3, 0, \dots, 0]^\top$. Explicitly stating this mathematical initialization would improve the clarity and reproducibility of the algorithm.
3. **Hardware Profiling Metrics - Static Memory Confusion:**
   In Section 3.5, under "Memory-Bandwidth and Weight-Reconstruction Bottlenecks", the authors state: *"FlatMerge requires a static allocation of $2040.42\text{ MB}$ (to store the base model, 4 task vectors, and active merged weights) but requires **exactly $0.00\text{ MB}$ of activation caching**."* Since the base model size is $340.11\text{ MB}$ and there are 4 task vectors plus the active merged weights, the static allocation is indeed $6 \times 340.11\text{ MB} = 2040.66\text{ MB}$. This is correct. However, they compare it to Weight-space TTA which requires $1360.28\text{ MB}$ of static memory plus activation caching. While FlatMerge completely eliminates activation caching (which can easily exceed 2.0 GB), it actually increases the static weight-storage memory on-device by requiring all 4 task vectors and the base model to be kept in SRAM/DRAM simultaneously, whereas standard TTA only needs the active merged weights in memory. The authors should clarify this trade-off explicitly: FlatMerge trades off a higher static model-storage footprint for a complete elimination of the highly dynamic, batch-size-dependent activation memory cache.

## Clarity and Reproducibility
* **Clarity:** The paper is exceptionally well-written, and the methodology is presented in a very structured and understandable manner. The inclusion of Algorithm 1 makes the optimization loop clear.
* **Reproducibility:** The exact optimization hyperparameters ($\sigma = 0.05$, $B_{\text{zo}} = 10$, $d=2$, learning rate $\eta$, steps $T=100$ or $200$) are clearly provided. However, the exact learning rate ($\eta$) used in the simulated or physical experiments is not explicitly specified in the text (e.g., in Section 4.1 or Algorithm 1), which is a minor reproducibility gap.
