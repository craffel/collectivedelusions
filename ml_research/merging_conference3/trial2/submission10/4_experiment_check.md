# Experimental Evaluation Check

## Evaluation of Experimental Setup
The experimental setup is exceptionally rigorous, especially for a deconstructive study:
* **Diverse Multi-Task Vision Benchmark**: The authors evaluate their ideas on four distinct image classification datasets of varying domains and difficulties (MNIST, FashionMNIST, CIFAR-10, SVHN).
* **Watertight Data Partitioning**: To eliminate data selection bias, the authors shuffle and partition each dataset into three unique disjoint splits per seed: (1) Head Training Split (512 labeled images), (2) Calibration Split (64 unlabeled images), and (3) Evaluation Split.
* **Large-Scale Evaluation**: Crucially, the final classification accuracies are evaluated on the **full, standard test splits** of all four datasets, representing a total evaluation scale of **56,032 images**. This eliminates evaluation split bias and ensures exceptionally tight, high-precision confidence intervals (shown by the small standard deviations in Table 1, e.g., MNIST standard deviation is $\le 0.94\%$).
* **Two Optimization Paradigms**: The authors evaluate both derivative-free zero-order optimization (1+1 ES, 500 steps) and first-order gradient-based optimization (Adam GD, 200 steps).

---

## Baselines Choice
The choice of baselines is comprehensive and highly contextualized:
1. **Task Arithmetic (Ilharco et al., 2022)**: Standard static baseline, representing uniform initialized scale $\lambda = 0.3$.
2. **Task-wise AdaMerging (Yang et al., 2024)**: Directly optimizing $T$ global parameters.
3. **Layer-wise AdaMerging (Yang et al., 2024)**: SOTA adaptive merging, optimizing $L \times T$ parameters.
4. **TIES-Merging (Yadav et al., 2023)**: SOTA static baseline featuring magnitude pruning, sign election, and conflict resolution.
5. **DARE-Merging (Yu et al., 2024)**: SOTA static baseline featuring random dropping and rescaling.
6. **Task Arithmetic Scaling Sweep**: A complete grid sweep over scaling factors ($\lambda \in \{0.1, 0.2, 0.3, 0.4, 0.5\}$) to verify whether any static scale can match adaptive merging.

This extensive baseline evaluation represents a major strength, allowing the authors to position post-hoc Spatial Averaging in the broader literature.

---

## Do Results Support Claims?
Yes, the results fully and unambiguously support every claim made in the paper:
* **The Overfitting-Optimizer Paradox**: Unconstrained layer-wise AdaMerging achieves $88.05\%$, but shuffling collapses performance to $78.61\%$ (a collapse of $9.44\%$). This confirms that learned coefficients are structurally specialized to their network positions.
* **Spatial Averaging as a Low-Pass Regularizer**: Post-hoc Spatial Averaging (Adam GD) achieves $84.96\%$ accuracy, outperforming static Task Arithmetic baseline ($84.64\%$). This confirms that taking the spatial mean smooths away the transductive test-time overfitting component of individual layers while successfully retaining the global task-level scaling signal.
* **The Spatial Averaging Paradox**: Direct Task-wise AdaMerging collapses average performance to $81.19\%$ (a drop of $3.45\%$ below Task Arithmetic initialization). This drop is driven by the easy tasks (MNIST remains at $96.31\%$ and FashionMNIST at $83.28\%$, while hard tasks collapse: CIFAR-10 drops from $89.93\%$ to $81.45\%$ and SVHN drops from $69.94\%$ to $63.71\%$). This empirical data perfectly supports the multi-task gradient imbalance theory.
* **Calibrated Remedy Failure**: Calibrated Task-wise AdaMerging collapses to $80.59\%$, confirming that uncalibrated entropy is only part of the pathology, and the core issue is the fundamental structural limitation of low-dimensional global bottlenecks under prediction entropy minimization.
* **Hierarchical Representation Routing**: Figure 4 (Layer-by-layer CKA representational similarity) visually confirms that early layers (1--4) maintain near-perfect representational alignment ($CKA > 0.995$) with the target task expert, while late task-specific layers (8--12) show distinct specialization. This beautifully supports the "local degrees of freedom" hypothesis.
* **Landscape Flatness under Noise**: Sweeping Gaussian noise $\gamma \in [0.05, 0.50]$ demonstrates that low-parameter configurations are flatter and highly robust to noise, validating the geometric stability of flat spatial regularization.
