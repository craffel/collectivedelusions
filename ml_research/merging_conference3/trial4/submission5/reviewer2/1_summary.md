# 1. Summary of the Submission

## Main Topic and Approach
This submission focuses on the problem of **weight-space model merging**—specifically, how to fuse multiple task-specific fine-tuned neural network experts (derived from a shared pre-trained base model) into a single, unified, multi-task model without additional training. The authors target the major challenge of **catastrophic representational collision** (or parameter interference) in weight space, which occurs when unregularized, task-specific parameter updates (task vectors) are directly added or averaged.

To resolve this issue, the authors propose **Sparsity-Guided Task Arithmetic (SG-TA)**, a post-hoc, deterministic weight-space regularization framework. SG-TA decouples the sparsification and merging phases by applying magnitude-based binary masking to individual task-specific update vectors before they are linearly combined. 

The submission explores:
1. **Global Quantile (GQ) Masking**: Computes a single magnitude threshold globally across the entire task vector, allowing different layers to retain varying densities of active updates.
2. **Layer-wise Quantile (LQ) Masking**: Enforces a homogeneous parameter keep-ratio across all layers by calculating independent magnitude thresholds for each layer.
3. **Task Vector Magnitude Normalization (TV-Norm)**: Scales each task vector by the inverse of its mean absolute magnitude prior to masking to prevent tasks with larger update scales from dominating the merged weight space.
4. **Sigmoid-Gated Soft Masking (SG-TA-Soft)**: A continuous, differentiable soft-gating mechanism using a sigmoid function as a smooth surrogate to hard binary masking.
5. **Non-Uniform Hyperparameter Calibration (Random Search and Coordinate Search)**: Explores task-specific keep-ratios $k_i$ and scaling factors $\alpha_i$ to bypass the exponential complexity of grid search.

To select optimal merging hyperparameters (such as keep-ratios and scaling factors) without overfitting, the authors leverage **Offline Few-Shot Validation Tuning (OFS-Tune)** using only 10 samples per task.

## Key Findings
- **Magnitude Masking as a Spatial Regularizer**: Applying deterministic magnitude-based binary masking to task vectors significantly reduces weight-space interference and mitigates representational collapse.
- **Global Budget Flexibility is Crucial**: GQ masking consistently and substantially outperforms LQ masking (e.g., Joint Mean Accuracy of 61.40% vs. 57.81%). This indicates that enforcing uniform budgets across layers is suboptimal, as task specialization is concentrated in specific transformer blocks.
- **Outperforming Complex Heuristics**: SG-TA GQ achieves higher or comparable joint mean accuracy compared to more complex stochastic (DARE-Merging, 58.44%) and multi-stage consensus protocols (TIES-Merging, 60.64%).
- **Addressing Magnitude Imbalance**: Incorporating TV-Norm balances multi-task performance (MNIST accuracy rises from 36.74% to 53.70%), though it introduces calibration sensitivity that is stabilized by scaling the validation pool size to 20 or more samples.
- **Continuous Landscape Stabilization**: SG-TA-Soft (GQ-Soft) achieves a comparable Joint Mean Accuracy of 61.06% but significantly reduces hyperparameter selection variance (standard deviation of 0.75% vs. 1.39% for hard masking).
- **Coordinate Search Scalability**: Task-specific Coordinate Search optimizes hyperparameters in linear time $\mathcal{O}(T)$ while achieving a balanced multi-task model (MNIST accuracy improves by +13.64% over uniform search).
- **Absolute Performance Collapse**: Despite substantial relative improvements over naive merging, there remains a massive 34.51% performance gap between the merged model (61.40%) and the joint expert ceiling (95.91%).

## Explicitly Claimed Contributions and Evidence
1. **Validation of the Spatial Regularization Hypothesis**: Supported by a comprehensive grid sweep across keep-ratios $k \in [0.1, 1.0]$ and scaling factors $\alpha$, demonstrating that unregularized task updates introduce significant noise.
2. **Analysis of Masking Scopes (GQ vs. LQ)**: Supported by empirical results showing that GQ masking outperforms LQ, and a crossover analysis demonstrating LQ's superiority only at very high keep-ratios ($k \ge 0.7$).
3. **Task-Vector Magnitude Normalization and Validation Pool Scaling**: Supported by empirical results demonstrating the resolution of task dominance and a physical sweep of validation sizes $N_{\text{val}} \in [10, 20, 50, 100]$ to stabilize calibration.
4. **Non-Uniform Calibration via Coordinate Search**: Supported by comparing Coordinate Search against Random Search, showing that CS achieves better-balanced accuracy with linear-time complexity.
5. **Sigmoid-Gated Soft Masking**: Supported by evaluations of SG-TA-Soft demonstrating reduced calibration variance across random seeds.
