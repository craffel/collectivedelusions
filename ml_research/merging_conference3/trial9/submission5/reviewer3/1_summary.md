# Intermediate Evaluation 1: Summary of the Paper

## Main Topic and Scope
This paper presents a methodological audit and experimental deconstruction of dynamic model merging (specifically activation-space ensembling/blending of task-specific Low-Rank Adaptation (LoRA) experts). It examines the widely held consensus that classical parametric routers (e.g., linear gating heads optimized via gradient descent) catastrophically fail under low-data constraints. The scope covers both synthetic simulation (using a 14-layer Analytical Coordinate Sandbox) and real-world validation (using a BERT-Tiny model on SST-2 and QQP classification tasks).

## Approach and Methodology
The authors critically evaluate State-of-the-Art (SOTA) dynamic merging techniques—such as SABLE (stateless nearest-centroid routing) and ChemMerge (stateful continuous-time routing governed by chemical kinetics ODEs)—against classical parametric linear routers (Softmax and Sigmoid). 
To address potential optimization confounding factors in the classical routers, the authors introduce:
1. **Maximum-Entropy Zero-Initialization**: Setting weights and biases to zero, which maps to a uniform prior (maximum entropy) to avoid initial random routing bias.
2. **Proper L2 Regularized Calibration**: Applying L2 weight decay and sweeping the regularization hyperparameter $\lambda$ to mitigate overfitting on tiny calibration sets.
3. **Anisotropy Stress Test**: Injecting Toeplitz-structured covariance ($\rho \in [0.0, 0.5]$) into the activation spaces to simulate the representation "cones" observed in foundation models.

The paper evaluates these models across two data regimes:
- **Small-Sample Constraint Regime** ($N_{\text{cal}} = 64$ samples)
- **Large-Sample Generalization Regime** ($N_{\text{cal}} = 4000$ samples)

Additionally, the authors conduct:
- **Initialization and Gating Ablations**: Comparing Zero-Initialization vs. Random Initialization and Softmax vs. Sigmoid gating.
- **Layer-Wise vs. Layer-Invariant Ablation**: Testing 11 separate layer-wise parametric gating heads vs. a single layer-invariant head.
- **Open-Loop Smoothing Baseline (EMA-SABLE)**: Appending Exponential Moving Average smoothing to SABLE.
- **Real-World Transfer Validation**: Evaluating a BERT-Tiny model on SST-2 and QQP datasets with $N_{\text{cal}} \in \{32, 500\}$.

## Key Findings and Revelations
1. **The Small-Sample Overfitting Bottleneck**: At $N_{\text{cal}} = 64$, classical parametric routers overfit severely because learning $768$ parameters from $64$ samples is mathematically under-determined. Under-regularized linear routers collapse, while training-free methods (SABLE, ChemMerge) perform well because their cosine-similarity formulations act as highly effective, parameter-free inductive geometric priors.
2. **Large-Sample Generalization Recovery**: At $N_{\text{cal}} = 4000$, classical routers recover completely. The unregularized Softmax router achieves $76.22\%$ accuracy, significantly outperforming SABLE ($73.76\%$, $p < 0.01$) and closely approaching ChemMerge ($76.90\%$).
3. **The Bias-Variance Trade-off**: Strong regularization ($\lambda = 10^{-2}$) is necessary in small-sample regimes but acts as a performance-limiting constraint bias in large-sample regimes, capping accuracy at $74.10\%$. Scaling the regularization parameter down ($\lambda = 10^{-4}$) under data abundance achieves a near-optimal $75.70\%$.
4. **Closed-Loop Feedback Stabilization (Representational Lag)**: Tracking intermediate representation quality reveals that ChemMerge's stateful dynamics introduce a "representational lag" (cosine similarity to prototypes is lower in intermediate layers). However, this lag acts as a beneficial closed-loop low-pass filter that stabilizes ensembling trajectories under heavy activation noise, explaining ChemMerge's performance ceiling.
5. **No Catastrophic Jitter**: Layer-wise parametric routers do not suffer from routing weight oscillations (jitter), maintaining very smooth trajectories (Jitter: $0.0068 - 0.0458$) compared to ChemMerge ($0.0368$).
6. **Task Subspace Separability**: On real BERT-Tiny validation, classical routers do not collapse even at $N_{\text{cal}}=32$, because the underlying tasks (sentiment vs. duplicate questions) are semantically disjoint, mapping to highly separated subspaces where finding a separating hyperplane is trivial.

## Claimed Contributions and Supporting Evidence
- **Methodological Deconstruction**: Proving that reported failures of classical parametric routers are artifacts of poor initialization and lack of regularization, rather than representational collapse. *Evidence*: Quantitative tables showing complete recovery of parametric routers under proper regularization or sufficient data.
- **Information-Theoretic Initialization Framing**: Validating that zero-initialization provides an unbiased, uniform routing prior that acts as a safe fallback. *Evidence*: Ablation showing Zero-Init consistently outperforms Random Init by $+0.56\%$ to $+0.72\%$ in the low-data regime.
- **Mechanistic Characterization of ChemMerge**: Exposing that ChemMerge is effectively a closed-loop feedback controller whose representational lag functions as a temporal low-pass filter. *Evidence*: Layer-wise cosine-similarity tracking showing hysteresis in ChemMerge, along with Euler discretization step analyses and EMA-SABLE comparisons.
- **Actionable Deployment Guidelines**: A decision matrix mapping data budgets and noise profiles to the optimal ensembling strategy, grounded in a quantitative serving-time complexity analysis. *Evidence*: Detailed complexity table and sample-complexity/temperature sweeps.
