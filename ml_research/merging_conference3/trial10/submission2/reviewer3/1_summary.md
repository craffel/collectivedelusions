# Technical Summary: Layer-Decoupled Stateful Kinetics (LDS-Kinetics)

This document provides a comprehensive summary of the conference submission titled **"Layer-Decoupled Stateful Kinetics (LDS-Kinetics) for Dynamic Model Merging"**.

---

## 1. Main Topic and Scope
The submission addresses **dynamic model merging (test-time ensembling)** of low-rank adapters (LoRAs) in multi-task sequential workloads. Rather than routing queries to isolated full-parameter models, dynamic model merging blends task-specific expert adapters on-the-fly sample-by-sample using a single shared backbone model. 

The paper identifies a critical, unaddressed assumption in current stateful dynamic model merging methods (such as ChemMerge and PAC-Kinetics): **spatial homogeneity**. SOTA stateful routers apply a single, global ensembling coefficient vector across all depths of the network. LDS-Kinetics challenges this by treating network depth as an active variable, decoupling the stateful ensembling kinetics across different network layers or blocks of layers.

---

## 2. Methodology and Approach
LDS-Kinetics partitions the dynamic ensembling layers of a network (e.g., Layers 4 to 14 in a 14-layer backbone) into $M$ disjoint blocks. 

1. **Activation-Space Routing:** Input activations are extracted at an early layer ($l_{\text{route}} = 3$), normalized, and projected onto task-specific principal component analysis (PCA) subspaces to generate task coordinate affinity signals $\mathbf{e}_t \in [0, 1]^K$.
2. **Decoupled State Recurrence:** Each of the $M$ blocks maintains an independent temporal state vector $s^{(m)}_t \in \mathbb{R}^K$ representing task concentration. The update equation is:
   $$s^{(m)}_t = \mathbf{A}^{(m)}_t s^{(m)}_{t-1} + W^{(m)} \mathbf{e}_t$$
   where $\mathbf{A}^{(m)}_t = \text{diag}(a^{(m)}_{1, t}, \dots, a^{(m)}_{K, t})$ represents the dynamic block-wise state-retention, which is scaled by the online cosine similarity $Sim_t$ of incoming coordinates to flush memory during task switches.
3. **Gibbs Softmax Policy:** Raw concentration states are mapped to ensembling weights via a multi-temperature Gibbs policy:
   $$\alpha^{(m)}_{k, t} = \frac{\exp(s^{(m)}_{k, t} / \tau^{(m)}_k)}{\sum_{j=1}^K \exp(s^{(m)}_{j, t} / \tau^{(m)}_j)}$$
   where the temperatures $\tau^{(m)}_k$ are learnable block-wise parameters.
4. **Learning-Theoretic PAC-Bayesian Regularization:** To prevent transductive overfitting on short calibration sequences ($T=32$), the authors derive a unified complexity penalty from Catoni's $\beta$-mixing PAC bound. Under a Gaussian posterior and isotropic prior centered around SABLE-grounded neutral defaults, this simplifies to an isotropic block-wise $L_2$ weight decay that regularizes learnable state-retention and temperature parameters.

---

## 3. Key Findings
* **Temporal-Spatial "Tempo-Gradient" (Ablation Findings):**
  * **Early Blocks (Layers 4–7):** Learn high temperatures ($\tau \approx 0.18$) and high decay/low retention rates ($a \approx 0.32$). This enables short-term memory, making early layers highly dynamic and responsive to rapid task switches, aligning representation spaces immediately.
  * **Late Blocks (Layers 12–14):** Learn low temperatures ($\tau \approx 0.04$, leading to sharp, deterministic choices) and low decay/high retention ($a \approx 0.94$). This acts as a robust low-pass filter, smoothing out logit fluctuations and suppressing serving jitter.
* **Optimization-Level Lockstep Pathology:** Unregularized Empirical Risk Minimization (ERM) fails because of a sign-symmetry pathology under the Adam optimizer. Gradients share the same sign across blocks at initialization, causing Adam's sign-based updates to lock block parameters into identical values. The PAC-Bayesian penalty solves this optimization pathology by introducing a KL gradient bias that breaks this starting sign-symmetry naturally, in addition to bounding statistical complexity.
* **Non-linear Coordinate Sandbox (GELU + LN):** Stateful ensembling outperforms stateless methods by up to $0.70\%$ in absolute accuracy under non-linear activation propagation, as temporal smoothing prevents high-frequency ensembling weight fluctuations from compounding into extreme representational drift.
* **Scalability to Large Expert Pools ($K=16$):** At scale, LDS-Kinetics suppresses heterogeneous jitter by $8.0\%$ and homogeneous jitter by $12.8\%$ over the global baseline, exhibiting sub-linear step latency scaling ($\sim 345\ \mu$s for $M=11$ on CPU) due to parallelized matrix-vector formulations.
* **Physical Backbone Validation:** On a pre-trained physical 6-layer Transformer, LDS-Kinetics ($M=2$) achieves a $46.6\%$ jitter reduction over SABLE and $6.1\%$ over Global PAC-Kinetics, with negligible systems-level latency overhead when state recurrences are packed into a single batched tensor operation.

---

## 4. Explicitly Claimed Contributions (with Evidence)
1. **The LDS-Kinetics Framework:** Formulation of a depth-decoupled stateful dynamic model merging framework generalized to arbitrary block scales $M$ (theoretically detailed in Section 3 and verified empirically in Section 4).
2. **Unified PAC-Bayesian Complexity Penalty:** Derivation of a principled learning-theoretic regularizer from Catoni's bound to stabilize decoupled parameter optimization in low-data calibration regimes (detailed in Section 3.5, with empirical ablation sweeps showing it prevents transductive overfitting in Figures 2 and 3).
3. **Deconstruction of Depth-Dependent Kinetics:** Empirically validating that deep neural networks naturally organize temporal ensembling scales along their depth, forming a responsive-to-stable "tempo-gradient" (elaborated in Section 4.3.4, with schematic illustrations in Figure 1).
