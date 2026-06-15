# Summary of the Paper

## Context and Overview
This paper addresses the problem of dynamic model merging (test-time ensembling) for multi-task sequential workloads using parameter-efficient adapters like Low-Rank Adapters (LoRAs). Existing stateful routing frameworks (such as ChemMerge and PAC-Kinetics) use continuous-time chemical kinetics or state space models to resolve the **routing jitter paradox** (the high-frequency oscillation of ensembling weights caused by stateless activation-space projections). However, these frameworks apply a **single, global ensembling coefficient vector** uniformly across all network depths, enforcing *spatial homogeneity*. 

The authors propose **Layer-Decoupled Stateful Kinetics (LDS-Kinetics)**, which challenges this spatial homogeneity assumption. LDS-Kinetics partitions network layers into $M$ disjoint blocks (or individual layers), maintaining independent concentration states for each block. These states evolve according to block-specific parameters, allowing different layers to learn distinct temporal tempos (adaptation speeds vs. decision stability).

To manage the overparameterization and risk of transductive overfitting on short calibration streams introduced by decoupling parameters, the authors formulate a unified **PAC-Bayesian complexity penalty** based on Catoni's $\beta$-mixing PAC bound.

## Methodology Key Components
1. **Coordinate Projection:** Normalizes intermediate representations at an early layer ($l_{\text{route}} = 3$) and projects them onto task-specific PCA subspaces to obtain task affinity coordinate signals.
2. **Decoupled Stateful Recurrence:** Tracks temporal task affinity with separate state vectors $s^{(m)}_t$ per block, updated using dynamic block-wise retention rates $a^{(m)}_{k, t}$ scaled by consecutive coordinate cosine similarity ($Sim_t$).
3. **Multi-Temperature Gibbs Policy:** Maps each block's concentration state to a probability distribution (ensembling weights) via block-and-task-specific learnable temperatures and retention rates.
4. **PAC-Bayesian Regularization:** Extends Catoni's PAC bound to regularize the total parameter vector, shrinking parameters toward safe, default SABLE-grounded priors.

## Key Findings and Experimental Results
- Evaluated on a **14-layer Analytical Coordinate Sandbox** (orthogonal and overlapping task manifolds, homogeneous and heterogeneous query streams, across 5 independent seeds).
- **Temporal-Spatial "Tempo-Gradient":** Deep ablation sweeps reveal that early blocks learn high decay (low memory) to adapt rapidly to task transitions, while late blocks learn low decay (high memory/inertia) to act as stable low-pass filters that suppress logit jitter.
- **Adam lockstep symmetry pathology:** The authors identify that standard unregularized Decoupled ERM falls into textbook weight symmetry due to Adam's sign-based updates. Small perturbations break this symmetry, but unregularized models suffer from severe overfitting. The PAC-Bayesian complexity penalty resolves both the optimization pathology (via the KL gradient) and the generalization gap.
- **Non-linear Sandbox (GELU + LN):** Shows that stateful ensembling completely bridges the "stateful accuracy penalty" under non-linear activation propagation, outperforming SABLE and static baselines by up to 0.7% (absolute) in accuracy.
- **Physical Validation:** Confirms the benefits on a physical 6-layer sequence model, showing a 46.6% routing jitter reduction over SABLE and a 6.1% reduction over global stateful ensembling, with negligible (<1%) systems overhead using parallelized batched tensor products.
