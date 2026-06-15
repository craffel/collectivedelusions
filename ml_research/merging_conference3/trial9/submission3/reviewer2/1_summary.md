# 1. Summary of the Paper

## Main Topic and Approach
The paper addresses the problem of **sequential dynamic model ensembling and model merging** in deep neural networks, focusing on multi-task model serving (e.g., routing specialized Low-Rank Adapters or expert modules at runtime). 
Specifically, the authors focus on a major phenomenon they term **sequential routing jitter**, where layer-wise dynamic routing coefficients undergo high-frequency, violent oscillations across the depth of a deep network. This jitter leads to representational instability, functional shifts, and severe transductive overfitting under extreme data scarcity (e.g., only 16 calibration samples per task).

To address this, the paper approaches sequential deep ensembling through the lens of **discrete-time dynamical systems**. By applying **Banach's Fixed-Point Theorem**, the authors derive a novel Lipschitz bound on the joint layer-wise representation-routing map. To enforce contraction properties and mathematically guarantee convergence to a unique, stable representation-routing trajectory, they propose the **Contraction-Regularized Router (CR-Router)**. CR-Router incorporates a joint regularized objective that penalizes both the Frobenius norm of the routing projection matrices (acting as an upper bound on the spectral norm) and the inverse routing temperatures. 

Additionally, the authors propose:
1. **Update-Space Quasi-Contraction**: A theoretical relaxation for pre-trained, frozen residual models (where a strict contraction of absolute representations is impossible without modifying the backbone).
2. **Centroid-Based Routing Warm-Starting**: An elegant initialization strategy using calibration-split centroids to guide optimization into stable, task-aligned basins.
3. **Adaptive Test-Time Temperature Annealing**: A post-hoc inference sharpening method that scales down routing temperatures to mitigate "expert dilution" while preserving optimization stability.
4. **Three Label-Free Tuning Heuristics**: Metrics (Gating Depth-Variance, Shannon Gating Entropy, and Running Gating Lipschitz Bound) to monitor and tune regularization parameters online under extreme data scarcity.

## Key Findings and Claims
1. **Theoretical Convergence**: The joint layer-wise mapping can be constrained to be a strict contraction ($L_{T_l} < 1$) under specific, mathematically derived bounds on the routing projection matrix spectral norm $\|W_{\text{route}}^{(l)}\|_2$ and inverse temperature $1/\tau_l$.
2. **Mitigation of Routing Jitter**: The unregularized Linear Router exhibits extreme routing jitter (high-frequency layer-wise coefficient changes), while CR-Router converges smoothly to a unique fixed-point trajectory across depth.
3. **Substantial Empirical Performance Gains**:
   - In **orthogonal task subspaces** (Experiment 1), CR-Router achieves **53.35% $\pm$ 3.84%** classification accuracy, outperforming the unregularized router by **18.62%** absolute.
   - In **overlapping task subspaces** (Experiment 2), static Uniform Merging collapses to **27.48%** due to representation cross-talk, whereas CR-Router achieves **43.48% $\pm$ 4.70%** (an absolute improvement of **16.00%** over Uniform Merging and **12.86%** over the unregularized router).
   - Under a **hierarchical multi-task depth-heterogeneous setting**, CR-Router significantly outperforms the Shared Router and the L2-Fixed Router (by **+14.37%** in Exp 1 and **+8.25%** in Exp 2).
   - On **real-world vision embedding manifolds** (Experiment 3: MNIST, Fashion-MNIST, KMNIST, USPS embeddings), CR-Router achieves **53.70% $\pm$ 2.37%** classification accuracy, outperforming L2-Fixed by **+6.37%** and the unregularized router by **+14.00%** absolute.
4. **Post-Hoc Sharpening via Annealing**: Reducing the temperature scale factor $\gamma_{\text{scale}}$ to 0.10 during test-time inference increases CR-Router's classification accuracy on real-world manifolds from 53.55% to a stellar **62.45% $\pm$ 2.98%**, effectively resolving the "expert dilution" dilemma.
5. **Efficiency**: Parametric routing with CR-Router achieves a massive serving throughput increase over non-parametric distance-based baselines (SABLE, ChemMerge) because it avoids costly nearest-neighbor computations.

## Explicit Contributions
- **Dynamical Systems Formulation**: Translating sequential dynamic ensembling into an iterative discrete-time dynamical systems framework over Banach spaces.
- **Theoretical Generalization and Lipschitz Bounds**: Proving Theorem 3.1 and Theorem 3.2, which derive Lipschitz bounds for the joint representation-routing mapping under standard and interpolative coordinate systems (including a soft-alignment formulation).
- **CR-Router Design & Objective**: Proposing the joint regularized objective (Frobenius norm on routing heads + inverse temperature penalty) to guarantee contraction properties.
- **Centroid-Based Routing Warm-Starting**: Resolving seed sensitivity under extreme data scarcity through centroid-based initialization of routing weights.
- **Adaptive Test-Time Temperature Annealing**: Decoupling training-time optimization stability from test-time representation sharpness, unlocking a +8.90% absolute gain.
- **Label-Free Heuristics**: Proposing and validating three online, label-free metrics to monitor and tune regularization in the absence of validation sets.
- **Exhaustive Empirical Evaluation**: Evaluating CR-Router across orthogonal, overlapping, and real-world embedding manifolds, demonstrating significant gains over competitive baselines.
