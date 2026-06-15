# 1. Summary of the Paper

## Main Topic
The paper explores the optimization dynamics and potential pitfalls of adaptive, unsupervised model merging at test time. Specifically, it focuses on **AdaMerging**, a state-of-the-art framework that uses unsupervised prediction entropy minimization on a small, unlabeled test-time batch to dynamically learn layer-wise or task-wise merging coefficients for multi-task learning.

## Approach
To dissect the performance of AdaMerging, the authors deconstruct its learned scaling coefficients through a critical, minimalist lens. They introduce two diagnostic controls to evaluate layer-wise coefficients:
1. **Intra-Task Layer Shuffling**: Randomly permuting the learned layer-wise coefficients of each task across the 12 Transformer layers. This is used to test if the coefficients are structurally specialized to the network's hierarchy (e.g., early vs. late layers).
2. **Spatial Averaging (Spatial Mean)**: Computing the average (mean) of the optimized layer-wise coefficients for each task and applying this single flat scalar across all layers of the network. This reduces the degrees of freedom from hundreds (layer-wise) back to one scalar per task (flat).

Additionally, the paper evaluates direct task-wise optimization (Task-wise AdaMerging) and proposes a **Calibrated Prediction Entropy** objective to mitigate gradient imbalance by normalizing each task's loss by its value at initialization.

## Key Findings
1. **Overfitting-Optimizer Paradox**: 
   - Layer-wise AdaMerging (Adam GD) achieves a high average test accuracy of $88.05\%$.
   - **Intra-Task Layer Shuffling** causes accuracy to collapse to $78.61\%$, indicating that the learned coefficients are not random noise but are structurally specialized to their layer positions (respecting the network's representational hierarchy of early general features vs. late task-specific features).
   - **Spatial Averaging** acts as a spatial low-pass filter. By replacing layer-wise coefficients with their spatial mean, it smooths away high-frequency transductive test-time overfitting, achieving $84.96\%$ accuracy, which still outperforms the static Task Arithmetic baseline ($84.64\%$) despite losing layer-specific routing.

2. **Spatial Averaging Paradox**: 
   - While post-hoc **Spatial Averaging** of layer-wise optimized coefficients achieves $84.96\%$, **direct** test-time optimization of flat task-wise scales (Task-wise AdaMerging) fails spectacularly, collapsing performance to $81.19\%$ (well below the uniform $84.64\%$ initialization).
   - This paradox is explained by **multi-task gradient imbalance** under uncalibrated prediction entropy and low-dimensional bottlenecks. Simple tasks (like MNIST/FashionMNIST) have naturally sharp logit distributions and highly responsive entropy. In a low-dimensional flat bottleneck (one scalar per task), the optimizer is drawn into scaling up easy-task coefficients to drive joint entropy down, which causes massive parameter interference and collapses performance on harder tasks (CIFAR-10 and SVHN).
   - In contrast, high-dimensional layer-wise optimization has enough local degrees of freedom to minimize entropy locally (e.g., in task-specific late layers) without forcing global scaling trade-offs.

3. **Incompatibility of Global Bottlenecks with Joint Entropy Minimization**:
   - The authors' proposed remedy, **Calibrated Prediction Entropy**, fails to restore direct flat-coefficient adaptation ($80.59\%$). This proves that the bottleneck itself, combined with the uncalibrated nature of prediction entropy (which drives overconfident misclassifications by artificially inflating logit magnitudes without task labels), is fundamentally incompatible with joint weight-space optimization across shared layers.

4. **Landscape and Representation Analysis**:
   - Noise sweeps show that the optimization landscapes are flatter and highly robust to coefficient noise.
   - Linear CKA similarity sweeps across all 12 layers show that early layers remain highly aligned with the expert ($CKA > 0.995$) across all merging schemes, while late layers exhibit distinct specialization to task-specific representations.

## Explicitly Claimed Contributions
- **Contribution 1**: Deconstructing the Overfitting-Optimizer Paradox, proving that learned layer-wise coefficients capture structural architectural hierarchy but are prone to test-time overfitting.
- **Contribution 2**: Discovering and mathematically explaining the Spatial Averaging Paradox, highlighting the multi-task gradient imbalance caused by uncalibrated prediction entropy objectives in low-dimensional weight-space bottlenecks.
- **Contribution 3**: Demonstrating that post-hoc Spatial Averaging acts as a powerful regularizer and spatial low-pass filter, outperforming static Task Arithmetic without requiring ground-truth labels for a grid search.
- **Contribution 4**: Comprehensive, seed-controlled evaluations of optimization landscape flatness and layer-by-layer Linear CKA, revealing that near-perfect mid-network CKA is a baseline property of task vector scaling rather than a benefit of test-time entropy minimization.
