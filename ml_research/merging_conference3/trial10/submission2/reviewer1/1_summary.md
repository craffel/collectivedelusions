# Paper Summary - Layer-Decoupled Stateful Kinetics (LDS-Kinetics)

## 1. Main Topic and Scope
This paper addresses the problem of **dynamic model merging (or test-time ensembling)** in multi-task sequential serving workloads. The goal is to dynamically blend task-specific low-rank adapters (LoRAs) on-the-fly within a shared backbone network to serve sequentially arriving queries from different tasks. 

## 2. Proposed Approach: LDS-Kinetics
Existing stateful routers assume *spatial homogeneity*, maintaining a single global ensembling weight vector applied uniformly across all network depths. This paper introduces **Layer-Decoupled Stateful Kinetics (LDS-Kinetics)**, which decouples these ensembling dynamics along the depth of the network.
- **Block-wise Partitioning:** The ensembling layers are partitioned into $M$ disjoint blocks.
- **Independent Recurrences:** Each block $m$ maintains its own independent stateful concentration vector $s_t^{(m)}$:
  $$s_t^{(m)} = \mathbf{A}_t^{(m)} s_{t-1}^{(m)} + W^{(m)} \mathbf{e}_t$$
  where the decay/retention rates $\mathbf{A}_t^{(m)}$ are block-specific, and $W^{(m)}$ represents an unconstrained coordinate injection matrix.
- **Gibbs Softmax Policy:** Converts the states into ensembling weights $\alpha_t^{(m)}$ using block-specific, task-specific learnable temperatures.
- **PAC-Bayesian Regularization:** To prevent transductive overfitting and optimization degeneracies of the expanded parameter space (scaling as $M \times (2K + K^2)$) on short calibration streams, the authors formulate a unified complexity penalty derived from Catoni's $\beta$-mixing PAC bound.

## 3. Explicitly Claimed Contributions and Evidence
1. **Formulation of LDS-Kinetics:** The authors define a generalized state-space recurrence model decoupled along $M$ network blocks.
2. **Unified PAC-Bayesian Complexity Penalty:** Formulates a learning-theoretic isotropic $L_2$ weight-decay regularizer centered around default SABLE-grounded parameters to stabilize learning under data-scarce calibration regimes ($T=32$).
3. **Empirical Deconstruction of Spatial-Temporal Dynamics:** Shows that deeper layers learn higher state retention (lower decay) to serve as stable low-pass filters, while early layers learn higher decay to track rapid transitions.
4. **Exhaustive Sandbox & Physical Backbone Evaluation:**
   - Evaluated on a 14-layer coordinate sandbox across 5 independent seeds.
   - Evaluated on a non-linear sandbox (GELU + LN).
   - Validated on a physical 6-layer sequence model in PyTorch with pre-trained LoRA experts.

## 4. Key Findings
- On highly dynamic (heterogeneous) workloads, LDS-Kinetics ($M=11$) achieves marginal accuracy improvements (e.g., $66.84\%$ vs $66.81\%$ for Global PAC-Kinetics under overlapping manifolds) while maintaining low routing jitter compared to stateless methods.
- Unregularized decoupled training fails due to an optimization symmetry pathology under Adam, where all blocks update in lockstep. The PAC-Bayesian complexity penalty resolves this by breaking the symmetry and guiding correct specialization.
- Under non-linear propagation (GELU + LN), stateful smoothing completely out-performs stateless baselines (by up to $0.70\%$ in accuracy) because it prevents compounding ensembling weight noise across depths.
- In large-scale expert pools (up to $K=16$), the accuracy of LDS-Kinetics and Global PAC-Kinetics converges due to tight regularization, but LDS-Kinetics provides significantly better routing jitter reduction ($8\%$ to $12.8\%$).
- By utilizing a batched tensor state formulation, the physical execution latency of the $M=2$ decoupled router is virtually identical to the global baseline, eliminating latency overhead concerns.
