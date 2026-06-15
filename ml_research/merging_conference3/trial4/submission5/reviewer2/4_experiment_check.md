# 4. Experiment Check

## Evaluation of Experimental Setup and Datasets
The empirical evaluation in this submission is exceptionally thorough, structured, and statistically rigorous, though limited in architectural and dataset scale:
- **Datasets**: The benchmark comprises **MNIST**, **FashionMNIST**, **CIFAR-10**, and **SVHN**. While these datasets simulate a high-conflict visual classification scenario, they are classical, low-resolution, and simple benchmarks. They represent a highly controlled "sandbox" but do not capture the complexity of real-world multi-task learning or modern language/multimodal tasks.
- **Backbone**: The choice of a compact Vision Transformer (\texttt{vit\_tiny\_patch16\_224}, 5.7M parameters) allows the authors to perform exhaustive parallel sweeps over random seeds, keep-ratios, and baseline parameters. However, as noted in the methodology review, the low capacity of this model behaves differently from over-parameterized foundation models.
- **Statistical Rigour**: The authors evaluate all methods across **5 random calibration seeds**, reporting means and standard deviations. This is a commendable practice that is often missing in model-merging literature, where single-run results are frequently presented.

---

## Evaluation of Baselines
The selection of baselines is highly comprehensive and represents a major strength of the paper:
1. **Naive Uniform TA and Optimized TA**: Establish clear reference points for unregularized weight addition and global scaling.
2. **TIES-Merging and DARE-Merging**: Represent the current state-of-the-art in weight-space consolidation.
3. **Decoupled Prune-then-Merge (P-then-M)**: Directly tests whether the decoupled sequence matches layer-wise masking.
4. **Layer-Group Scaling (L-Scale)**: Proves that layer-wise scaling alone (without sparsification) is highly suboptimal (scoring only 32.44% Joint Mean Accuracy), validating that magnitude-based pruning is indeed the primary driver of regularization.
5. **Fisher-Weighted Averaging**: A strong first-order curvature-based baseline, which the proposed zero-order SG-TA GQ outperforms by a large margin (61.40% vs. 37.85%).
6. **Joint Multi-Task Learning (MTL) and Dense Experts (Ceiling)**: Establish the true multi-task training upper bound (95.55%) and individual expert limits (95.91%). Including these ceilings is highly honest and highlights the massive absolute performance gap.

---

## Do the Results Support the Claims?
Yes, the empirical findings strongly support the authors' claims, and the paper is written with high scientific honesty:

1. **SG-TA (GQ) Outperforming Complex Baselines**:
   - **Claim**: SG-TA (GQ) achieves 61.40% Joint Mean Accuracy, outperforming TIES-Merging (60.64%) and DARE-Merging (58.44%).
   - **Support**: Table 1 clearly shows these numbers.
   - **Honesty**: The authors explicitly state that "because of overlapping standard deviations, our method's superiority over TIES-Merging is not statistically significant." This level of empirical honesty is exemplary and prevents over-claiming.

2. **Crucial Role of Global Budget Flexibility**:
   - **Claim**: Global Quantile (GQ) masking consistently and substantially outperforms Layer-wise Quantile (LQ) masking.
   - **Support**: Supported by Table 1 (61.40% vs. 57.81% at best configurations) and Table 2 (which shows GQ outperforming LQ across most keep-ratios $k \le 0.5$).
   - **Crossover Phenomenon**: The sensitivity analysis (Section 4.3) identifies a highly interesting crossover at $k \ge 0.7$, where LQ outperforms GQ. The authors provide a logical structural explanation: when the keep-ratio is high, enforcing a homogeneous budget prevents GQ from over-pruning certain layers and creating communication bottlenecks in the transformer's information flow. This thoroughness strengthens the paper.

3. **Efficacy of Task Vector Magnitude Normalization (TV-Norm)**:
   - **Claim**: TV-Norm resolves magnitude imbalance and balances multi-task performance.
   - **Support**: Table 1 shows that adding TV-Norm (SG-TA GQ-Norm) dramatically increases MNIST accuracy from 36.74% to 53.70% (a huge +16.96% absolute boost) and slightly increases CIFAR-10 accuracy, while regularizing the dominant SVHN accuracy from 85.35% to 70.18%.
   - **Calibration Sensitivity Sweep**: The authors physical sweep of validation sizes $N_{\text{val}} \in [10, 20, 50, 100]$ successfully supports their hypothesis that small-sample noise causes calibration volatility under TV-Norm, and that doubling the pool size to a highly manageable $N_{\text{val}}=20$ completely stabilizes the calibration standard deviation (reducing it by over 4x to $\pm 1.10\%$).

4. **Sigmoid-Gated Soft Masking (SG-TA-Soft) for Landscape Stabilization**:
   - **Claim**: Continuous soft-gating stabilizes hyperparameter calibration.
   - **Support**: Table 1 shows that SG-TA (GQ-Soft) achieves 61.06% Joint Mean Accuracy with a standard deviation of only **$\pm$ 0.75%** (compared to $\pm 1.39\%$ for the hard masking variant). This empirically validates that continuous gating smooths the calibration loss landscape and stabilizes hyperparameter selection.

5. **Coordinate Search (CS) Scalability**:
   - **Claim**: Coordinate Search is highly scalable and rebalances representations.
   - **Support**: Table 3 shows that CS completes calibration in 43.61s (vs. 110.07s for Random Search) and rebalances the model (MNIST accuracy improves by +13.64% over uniform search, and CIFAR-10 by +7.22%), by dynamically allocating highly sparse budgets to simple domains to prevent their interference with complex ones.
