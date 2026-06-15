# 1. Summary of the Paper

## Main Topic and Motivation
The paper addresses the challenge of deploying multiple task-specialized neural network experts under tight resource constraints (e.g., edge devices) without incurring the linear growth in memory/storage footprints or the high computational cost of joint multi-task retraining. The authors target the field of **model merging**, specifically focusing on the limitations of **static model merging** (which causes parameter collisions and representational interference across different tasks) and **dynamic model merging** (which often suffers from high parameter overhead leading to overfitting on small calibration sets, or relies on simple flat linear projections that fail to capture non-linear, multi-scale representational boundaries).

## Proposed Approach
The authors propose **ChaosMerge** (Chaos-Theoretic Attractor Merging), which introduces a non-linear dynamical systems framework to model-space fusion:
1. **Gated Coupled Map Lattice (G-CML):** The sequence of a network's layers ($L$ layer groups) is conceptualized as discrete time-steps of a chaotic Coupled Map Lattice (CML) driven by a Logistic Map $f(u) = 4u(1-u)$. To prevent the exponential gradient explosion ($4^{14} \approx 2.68 \times 10^8$ for a 14-layer network) typical of deep recurrence with chaotic maps, the authors introduce learned layer-wise gating coefficients ($\lambda_l \in [0, 1]$) acting as residual skip-connections.
2. **Task-Specific Dynamic Routing (via Task-Level Centroids):** To prevent task-specific trajectories from being washed out in heterogeneous batches, the model computes merging coefficients using a task-level feature centroid $\psi(x)_j$ in a sphere-projected 4-dimensional phase space. Weights are assembled once per task/batch, balancing representational customization and inference speed.
3. **Extremely Compact Parameter Footprint:** ChaosMerge requires exactly **384 parameters** total (comprising initializer weight/bias, layer-wise coupling, scaling amplitudes, attractor keys/biases, and layer-wise gating).
4. **Annealed Chaos-to-Order Merging (Extension):** An annealing scheme that dynamically interpolates the transition from active chaos (Logistic Map) early in training to stable order (Tanh Gated Map) near convergence, bridging exploration and exploitation.

## Key Findings and Claims
- **Gradient Stabilization:** The learned gating factor $\lambda_l \approx 0.12$ successfully tames gradient explosion, boosting average merging accuracy by **+18.60%** absolute over the ungated chaotic baseline (from 55.20% to 73.80%).
- **State-of-the-Art Annealing Performance:** The proposed Annealed Chaos-to-Order Merging achieves **78.12%** average accuracy, which outperforms pure G-CML (72.90%), pure Tanh Gated (75.45%), and even over-parameterized dynamic routers like the Linear Router (77.10%) and QWS-Merge (77.05%) under the Task-Specific evaluation setup.
- **Resilience to Overfitting:** In low-resource scenarios (B=16 samples), G-CML maintains **73.50%** average accuracy, demonstrating robustness against the Overfitting-Optimizer Paradox that plagues larger unconstrained routers in tiny calibration regimes.
- **Unsupervised Clustering Bottlenecks:** On-the-fly unsupervised $K$-means clustering of heterogeneous batches in the 4D projected sphere space is highly fragile due to spatial overlap (achieving only 45.31% clustering purity, leading to a catastrophic drop in classification accuracy to 45.31%).

## Explicitly Claimed Contributions (with Evidence)
1. **Gated Chaotic Merging Paradigm (G-CML):** Successfully tames gradient explosion. Evidence: Lyapunov exponents show a transition from positive (average $\lambda_{\text{Lyapunov}} = +0.3420$, active chaos) to negative (average $\lambda_{\text{Lyapunov}} = -0.2964$, stable basins) as optimization proceeds (Figure 2).
2. **Task-Specific Dynamic Routing:** Prevents individual task signatures from being washed out in mixed batches. Evidence: Task-Specific routing achieves higher accuracy (73.80%) compared to Task-Averaged routing (71.20%) for G-CML (Table 1).
3. **Extremely Compact Parameter Footprint:** Utilizes only 384 parameters. Evidence: Parameter counting shows exactly 384 parameters, which is nearly $30\times$ fewer than the Linear Router baseline (10,808 parameters).
4. **Outstanding Empirical Results:** Outperforms static baselines. Evidence: ChaosMerge (G-CML) under task-specific routing scores 73.80% compared to Uniform Merging (54.75%) and AdaMerging (70.85%). Under the annealed scheme, it achieves 78.12% (Table 2).
