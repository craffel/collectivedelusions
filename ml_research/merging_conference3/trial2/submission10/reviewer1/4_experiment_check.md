# 4. Experimental Evaluation and Results Check

## Evaluation of Experimental Setup
The experimental setup designed by the authors is remarkably rigorous and tailored to the deconstructive nature of the study:
- **Dataset Suitability**: The four selected datasets (MNIST, FashionMNIST, CIFAR-10, SVHN) vary significantly in difficulty, domain, and homogeneity. This heterogeneity is vital because it exposes the difference in prediction entropy across tasks, which is the exact trigger for the **Spatial Averaging Paradox** (multi-task gradient imbalance).
- **Scale and Rigor**: While test-time calibration is done on a practical, small-scale subset ($64$ images per task), the final evaluation is performed on the **full, standard test splits** of all datasets. The resulting evaluation scale of **56,032 images** completely eliminates data selection bias and ensures highly stable, statistically sound findings.
- **Multi-Seed Controls**: Running all experiments across **three independent seeds** ($\mathcal{S} \in \{42, 100, 2026\}$) is excellent practice. It provides error bars (mean $\pm$ standard deviation) that reveal important variance properties, such as the high sensitivity of the SVHN domain to seed-specific task experts and calibration batches.

## Adequacy of Baselines
The paper includes an exceptionally comprehensive suite of baselines for comparison:
1. **Task Arithmetic (Static Baseline)**: The foundational baseline for weight-space task vector combinations.
2. **TIES-Merging & DARE-Merging (SOTA Static Baselines)**: Advanced static merging techniques that prune, sparsify, and resolve sign conflicts.
3. **AdaMerging (SOTA Adaptive Baseline)**: The direct competitor under two optimization algorithms: **1+1 Evolution Strategy** (derivative-free) and **Adam Gradient Descent** (first-order).
4. **Task-wise AdaMerging**: Evaluating direct flat-coefficient adaptation.
5. **Calibrated Task-wise AdaMerging**: Evaluating direct flat adaptation under loss-normalized controls.
6. **Intra-Task Layer Shuffling**: A structural diagnostic control.
7. **Spatially Averaged (Mean)**: The proposed regularized diagnostic control.

This comprehensive set of baselines covers the full spectrum of relevant methods, ensuring that the results are compared against both classical and state-of-the-art static and adaptive schemes.

## Do the Results Support the Claims?
Yes, the empirical results in Table 1 and Table 2, as well as the figures, strongly and unambiguously support all of the authors' core claims:

1. **Overfitting-Optimizer Paradox (Supported)**:
   - Layer-wise AdaMerging (Adam GD) achieves $88.05\%$.
   - When **Intra-Task Layer Shuffling** is applied, accuracy collapses to $78.61\%$. This massive drop ($9.44\%$) proves that the learned layer coefficients are structurally specialized and sensitive to the network's architectural hierarchy.
   - When **Spatial Averaging** is applied, the model achieves $84.96\%$ (surpassing Task Arithmetic at $84.64\%$). This supports the claim that the spatial mean acts as an elegant low-pass filter, smoothing away transductive overfitting while preserving task scales.

2. **Spatial Averaging Paradox (Supported)**:
   - **Spatial Averaging** achieves $84.96\%$.
   - **Direct Task-wise AdaMerging** collapses to $81.19\%$.
   - This massive gap ($3.77\%$) and the collapse of Task-wise AdaMerging below its unoptimized baseline ($84.64\%$) empirically confirm the paradox.

3. **Multi-Task Gradient Imbalance Theory (Supported)**:
   - Looking at the individual task columns in Table 1 under Task-wise AdaMerging, MNIST ($96.31\%$) and FashionMNIST ($83.28\%$) maintain high accuracies, while CIFAR-10 collapses from $89.93\%$ to $81.45\%$ and SVHN crashes from $69.94\%$ to $63.71\%$.
   - This asymmetrical collapse perfectly supports the theory: the uncalibrated prediction entropy of easy tasks (with sharp distributions) dominates the joint gradient, leading the optimizer to scale them up, which causes destructive interference in early shared layers that collapses the harder tasks (CIFAR-10, SVHN).

4. **Incompatibility of Global Bottlenecks with Joint Entropy Minimization (Supported)**:
   - **Calibrated Task-wise AdaMerging** still fails ($80.59\%$), proving that even when task losses are balanced at initialization, a flat, global bottleneck forces joint entropy minimization to scale up parameters pathologically, creating overconfident misclassifications and severe weight-space interference.

5. **Hierarchical Representational Routing (Supported)**:
   - The layer-by-layer CKA curves in Figure 4 show that early layers (Layers 1--4) maintain near-perfect representational alignment ($CKA > 0.995$) across all methods, while late layers (Layers 8--12) show distinct specialization and lower alignment in the optimized layer-wise models.
   - This empirically confirms that high-dimensional optimization routes updates locally through task-specific late layers, keeping early general representations unperturbed and avoiding destructive global interference.

## Critical Critique of Experiments
While the experiments are highly robust, there are a few areas for constructive critique:
- **ViT Isotropic Bias**: The backbone used is a CLIP ViT-B/32, which is an isotropic architecture (same hidden dimension and structure across all layers). In hierarchical architectures like Swin Transformers or ResNets, representation sharing and capacity vary drastically across stages (e.g., stage 1 vs. stage 4). The authors acknowledge this under "Future Directions" in Section 5, but testing on ConvNeXt or Swin would have made the architectural claims even more robust.
- **Standard Deviation on SVHN**: The standard deviation of SVHN is large (e.g., $69.94\% \pm 5.97\%$ for Task Arithmetic and $63.71\% \pm 6.64\%$ for Task-wise Adam). This high variance indicates that SVHN is very sensitive to seed-specific experts and the calibration split. Although this variance is transparently discussed and handled via seed-controlled protocols, it suggests that conclusions regarding SVHN should be interpreted with some caution, though the average trends remain highly consistent.
