# 1. Summary of the Paper

## Main Topic and Motivation
This paper addresses the problem of **spatial weight-space interference** in weight-space model merging. When multiple task-specific expert models, fine-tuned from a shared pre-trained initialization, are merged via standard linear interpolation (Task Arithmetic) or sign-consistent averaging (TIES-Merging), conflicting updates from orthogonal domains cancel each other out, leading to catastrophic representation collapse. 

To mitigate this, the paper proposes **Exclusive Parameter Merging (EPM)**, a training-free parameter routing framework. EPM's key goal is to resolve weight conflicts at the coordinate level during composition, thereby reducing the dependency on highly parameterized post-hoc test-time adaptation (TTA) methods (like AdaMerging or ZipMerge) which are prone to overfitting or optimization failure.

## Proposed Methodology
EPM is composed of two primary techniques:

1. **Soft Exclusive Parameter Allocation (Soft-EPA):**
   * **Task Vector Standardization:** To prevent "rich" tasks (with naturally larger weight updates due to high-entropy datasets, e.g., SVHN/CIFAR-10) from dominating coordinate routing, the task vectors are standardized globally (or layer-wise) by their standard deviation ($\sigma_k$).
   * **Soft Routing:** At each coordinate $j$, the dominant expert $k^*(j)$ is elected based on the maximum scaled absolute magnitude. The update for $k^*(j)$ is kept at full strength ($\lambda_k \tau_{k, j}$), while non-dominant expert updates are attenuated by a coherence retention factor $\gamma = 0.2$ (dense baseline).
   * **Mathematical Formulation:** The paper demonstrates that Soft-EPA is mathematically equivalent to a convex combination of hard coordinate-wise exclusivity ($\gamma=0$) and standard Task Arithmetic ($\gamma=1$).
   * **Dynamic Coherence Scheduling (DCS):** Under sparse merging, $\gamma$ is scaled dynamically using a quadratic schedule: $\gamma(p) = \gamma_0 + (1-\gamma_0) \cdot p^2$, where $p$ is the target sparsity. This prevents representational starvation under extreme pruning.
   * **Scale Decoupling:** The standardization scale is used exclusively as a decision-making filter for routing and saliency ranking, whereas the unstandardized physical updates are integrated into the network to preserve pre-trained knowledge scales.

2. **Task-Level Coefficient Tuning (TLC-Tune):**
   * Rather than optimizing hundreds of layer-wise coefficients, TLC-Tune restricts the search space to just $K$ global scaling factors (one per expert).
   * These $K$ scaling factors are optimized using a zero-order (1+1) Evolution Strategy on a small, offline validation split (128 samples per task) to maximize a minimax multi-task accuracy objective.

## Key Findings and Claims
* **Resolution of Representational Collapse:** Under severe domain conflicts (MNIST, FashionMNIST, CIFAR-10, SVHN) on a ViT-Tiny (5.7M parameters) backbone, standard average-based merging methods collapse to near-random accuracy on grayscale tasks. EPM with TLC-Tune elevates MNIST and FashionMNIST to $\sim$48% and $\sim$46% respectively, achieving a joint mean of **46.19%** (dense).
* **Robustness vs. Optimization Failure:** SOTA layer-group-wise tuning methods (AdaMerging and ZipMerge) are evaluated under the same zero-order ES setup across a 500-step study. They remain completely stuck at their initial state, demonstrating "absolute optimization failure." In contrast, TLC-Tune's 4-dimensional space converges in under 40 steps, proving that low-dimensional scale parameterization is highly robust to zero-order search and transductive noise.
* **The Exclusivity Contradiction and Shielding:** The paper acknowledges that standard average blending can achieve higher joint averages under 50% sparsity (as the probability of coordinate collisions decays). However, it shows that Soft-EPA acts as a crucial spatial shield, raising the worst-case performance floor (MNIST accuracy) by over 43% absolute compared to average blending with pruning.
* **Sparsity Resilience:** Under extreme pruning (80%), DCS rescues EPM from capacity starvation, achieving **26.41%** joint mean accuracy, although it is outperformed by DARE (**40.90%**) which uses expectation-value rescaling.

## Explicitly Claimed Contributions (with Evidence)
1. **Soft Exclusive Parameter Allocation (Soft-EPA):** A training-free coordinate-level routing operator that mitigates spatial weight-space interference. (Evidence: Section 3.1, Tables 1 & 2 showing accuracy improvements over standard averaging).
2. **Task Vector Standardization:** Mitigates the "Rich Task" Dominance Trap by normalizing update scales before routing. (Evidence: Section 3.1, and Section 4.3 analysis showing a 13.8% "scale override" rate that protects grayscale tasks from being erased).
3. **Task-Level Coefficient Tuning (TLC-Tune):** A minimalist $K$-dimensional scale optimization that converges rapidly and avoids the Overfitting-Optimizer Paradox. (Evidence: Section 4.1 & 4.3, Figure 2 showing validation and test trajectories over 500 steps).
4. **Dynamic Coherence Scheduling (DCS):** A quadratic scaling mechanism for $\gamma(p)$ to prevent representational collapse under extreme sparsity. (Evidence: Table 3 showing DCS outperforming a fixed $\gamma=0.2$ at 80% sparsity).
5. **LLM Scalability Blueprints:** Direct pseudocode implementation details for decoder-only autoregressive Transformers. (Evidence: Algorithm 1).
