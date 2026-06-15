# Paper Outline: Layer-Decoupled Stateful Kinetics (LDS-Kinetics) for Dynamic Model Merging

## Title
Layer-Decoupled Stateful Kinetics (LDS-Kinetics) for Dynamic Model Merging

## Authors & Affiliation
Dr. Evelyn Vance & Dr. Marcus Thorne
Department of Computer Science, Stanford University, Stanford, CA, USA
{evance, mthorne}@stanford.edu

## 1. Abstract
- **Context:** Dynamic model merging (or test-time model ensembling) allows serving multi-task queries using low-rank adapters (LoRAs) without storing full task-specific models or incurring high inference routing latency.
- **Challenge:** Prior state-of-the-art stateful routing methods (like PAC-Kinetics) apply a single global routing coefficient across all network depths. This assumes spatial homogeneity across layers, ignoring that early, middle, and late layers of deep networks process fundamentally different semantic representations (e.g., local features vs. task-specific logits).
- **Proposed Solution:** We introduce **Layer-Decoupled Stateful Kinetics (LDS-Kinetics)**. By partitioning layers into distinct blocks (or individual layers) and maintaining decoupled concentration states with block-specific retention and coupling parameters, we allow different depths to stabilize at different temporal scales.
- **Key Contributions:** 
  1. We generalize continuous-time kinetics state space routing to block/layer-wise levels.
  2. We introduce learning-theoretic PAC-Bayesian regularization to simultaneously constrain all decoupled blocks.
  3. We conduct a massive empirical evaluation (5 seeds, orthogonal/overlapping task manifolds, homogeneous/heterogeneous workloads) comparing $M=1$ (global), $M=3$ (tri-block), and $M=11$ (fully decoupled) configurations.
- **Results:** LDS-Kinetics matches or outperforms prior stateful ensembling baselines while providing the first empirical deconstruction of stateful routing depth dynamics, showing that late layers learn higher inertia to act as stable filters while early layers adapt rapidly.

## 2. Introduction
- **Serving Multi-task Streams:** High-throughput serving of sequential heterogeneous query streams (e.g., MNIST, CIFAR-10, SVHN, etc.) in real-time.
- **Evolution of Dynamic Routing:** From raw, stateless projections (SABLE, PAC-ZCA) that suffer from high routing jitter under coordinate noise, to stateful models (ChemMerge, Momentum-Merge, PAC-Kinetics) that model routing as continuous-time chemical kinetics to smooth ensembling trajectories.
- **The "Spatial Homogeneity" Assumption:** Current stateful routers apply a single global ensembling coefficient $\alpha_t$ across all layers. Is this optimal?
- **Our Hypothesis:** Layers at different depths have different processing tempos. Early layers align intermediate representations, middle layers integrate semantics, and late layers refine output logits. A decoupled state space model is mathematically and empirically superior.
- **Empiricist Contribution:** We do not just claim this; we run extensive empirical sweeps. We evaluate Tri-Block ($M=3$) and Fully Decoupled ($M=11$) setups, analyze the learned parameters, and demonstrate that PAC-Bayesian complexity regularizers are essential to prevent transductive overfitting of high-dimensional parameter spaces on short calibration streams.

## 3. Related Work
- **Static vs. Dynamic Model Merging:** Weight arithmetic vs. input-dependent ensembling.
- **Stateful Routing in Model Merging:** ChemMerge (biochemical kinetics), Momentum-Merge (layer-wise EMA), PAC-Kinetics (learning-theoretic stateful).
- **Depth and Representation Dynamics:** Layer-specific processing characteristics of deep models. How early vs. deep layers behave differently.

## 4. Methodology (LDS-Kinetics)
- **Subspace Projections:** Standard PCA projection of normalized Layer 3 activations to obtain task affinity coordinates $\mathbf{e}_t$.
- **Decoupled Stateful Recurrence:** Formulating block-specific concentration state vectors $s^{(m)}_t = \mathbf{A}^{(m)}_t s^{(m)}_{t-1} + W^{(m)} \mathbf{e}_t$ with dynamic retention scaling $Sim_t$.
- **Gibbs Policy:** Multi-temperature block-specific Gibbs softmax to convert states into ensembling weights $\alpha^{(m)}_t$.
- **Dynamic Activation Blending:** Applying $\alpha^{(m(l))}_t$ to blend LoRAs at layer $l$.
- **PAC-Bayesian Complexity Penalty:** Formulating the Catoni $\beta$-mixing PAC-Bayesian complexity bound over the combined parameter vector $\Theta \in \mathbb{R}^{M \times (2K + K^2)}$.

## 5. Experimental Evaluation (The Empiricist's Sweeps)
- **Coordinate Sandbox:** 14-layer backbone, 192 dimensions, 4 task experts.
- **Workloads:** Homogeneous vs. Heterogeneous task sequences.
- **Manifolds:** Orthogonal vs. Overlapping task representations.
- **Baselines:** Oracle, Static Uniform, SABLE (Raw/SEP), Stateless PAC-ZCA, ChemMerge, Global PAC-Kinetics, Decoupled ERM.
- **Main Quantitative Results:** 
  - Joint Accuracy and Jitter tables for Orthogonal and Overlapping manifolds.
  - Deep analysis of LDS-Kinetics ($M=3$, $M=11$) vs. Global ($M=1$) and unregularized Decoupled ERM.
- **Empirical Ablations & Findings:**
  1. *The Overfitting Hazard of Decoupling:* Unregularized Decoupled ERM matches global routing but fails to extract depth-specific benefits because it overfits the calibration data.
  2. *Learned Parameter Trajectories:* Visualizing the learned retention rates and temperatures as a function of depth block.
  3. *Workload Robustness:* Sweeping noise levels $\sigma$ and showing how LDS-Kinetics maintains stable routing.

## 6. Conclusion & Discussion
- Summary of findings.
- Future work: Scaling to LLMs and vision-language models.
