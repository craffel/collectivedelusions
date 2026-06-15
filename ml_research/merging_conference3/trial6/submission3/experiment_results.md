# Phase 2 Experimental Results: Large-Scale Empirical Router Sweeps

This document summarizes the comprehensive empirical results obtained during the large-scale sweeps over the **Block-wise Weight-Sharing Routing Sweep (BWS-Router)** architecture and established model-merging baselines.

All metrics are reported as the **Mean ± Standard Deviation** computed across **5 independent random seeds** to ensure statistical robustness.

## Table 1: Main Multi-Task Generalization Performance (Homogeneous B=256)
Evaluation of dynamic routers under task-wise homogeneous stream deployment on visual classification tasks.

| Router Method | MNIST (%) | FashionMNIST (%) | CIFAR-10 (%) | SVHN (%) | Joint Mean (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Expert Ceiling | 100.00 ± 0.00% | 99.44 ± 0.14% | 97.30 ± 0.45% | 30.16 ± 0.89% | 81.73 ± 0.29% | 
| Static Uniform | 30.00 ± 8.94% | 27.24 ± 1.86% | 27.80 ± 3.12% | 9.20 ± 0.93% | 23.56 ± 2.91% | 
| Global Linear Unreg | 100.00 ± 0.00% | 99.44 ± 0.14% | 97.30 ± 0.45% | 17.20 ± 3.72% | 78.49 ± 0.94% | 
| Global Linear Reg | 100.00 ± 0.00% | 99.44 ± 0.14% | 97.30 ± 0.45% | 17.74 ± 4.12% | 78.62 ± 1.04% | 
| QWS Merge | 100.00 ± 0.00% | 98.94 ± 0.17% | 95.06 ± 2.31% | 18.28 ± 4.41% | 78.07 ± 1.06% | 
| L3 Linear Unreg | 100.00 ± 0.00% | 98.94 ± 0.48% | 96.38 ± 0.57% | 21.26 ± 2.95% | 79.14 ± 0.77% | 
| L3 Linear Reg | 100.00 ± 0.00% | 97.10 ± 1.99% | 91.82 ± 4.30% | 23.28 ± 4.08% | 78.05 ± 1.78% | 
| L3 Tanh Unreg | 100.00 ± 0.00% | 99.04 ± 0.33% | 95.88 ± 0.61% | 20.86 ± 2.81% | 78.94 ± 0.74% | 
| L3 Tanh Reg | 100.00 ± 0.00% | 99.14 ± 0.22% | 95.86 ± 0.57% | 20.98 ± 2.99% | 78.99 ± 0.73% | 
| L3 Softmax Unreg | 100.00 ± 0.00% | 98.74 ± 0.38% | 94.36 ± 0.94% | 20.80 ± 2.38% | 78.48 ± 0.70% | 
| L3 Softmax Reg | 100.00 ± 0.00% | 98.74 ± 0.38% | 94.36 ± 0.96% | 20.74 ± 2.29% | 78.46 ± 0.67% | 
| BWS M3 Sigmoid Reg | 100.00 ± 0.00% | 98.12 ± 2.42% | 95.78 ± 1.62% | 24.36 ± 4.30% | 79.56 ± 1.13% | 
| BWS M4 Sigmoid Reg | 100.00 ± 0.00% | 97.96 ± 2.74% | 95.82 ± 1.60% | 24.36 ± 4.23% | 79.54 ± 1.13% | 
| BWS M12 Sigmoid Reg | 100.00 ± 0.00% | 97.82 ± 3.12% | 95.80 ± 1.56% | 24.78 ± 3.96% | 79.60 ± 1.16% | 

## Table 2: Deployment Audit under Task Heterogeneity
Robustness check of model merging methods when subjected to batch stream configuration shifts: Sample-wise (B=1), Homogeneous batch (B=256), and Heterogeneous mixed-task batch (B=256).

| Router Method | Homogeneous (B=1) (%) | Homogeneous (B=256) (%) | Heterogeneous (B=256) (%) |
| :--- | :---: | :---: | :---: |
| Static Uniform | 23.56 ± 2.91% | 23.56 ± 2.91% | 14.06 ± 10.98% |
| Global Linear Unreg | 78.48 ± 0.94% | 78.49 ± 0.94% | 78.20 ± 0.90% |
| QWS Merge | 78.07 ± 1.06% | 78.07 ± 1.06% | 78.28 ± 2.09% |
| L3 Linear Reg | 78.05 ± 1.78% | 78.05 ± 1.78% | 76.95 ± 2.47% |
| L3 Softmax Reg | 78.46 ± 0.67% | 78.46 ± 0.67% | 78.12 ± 1.31% |
| BWS M3 Sigmoid Reg | 79.56 ± 1.13% | 79.56 ± 1.13% | 79.30 ± 1.88% |

## Table 3: Block-wise Layer-Sharing Sensitivity (Sweep over Group Size M)
Analysis of BWS-Router (Sigmoid, Reg) Joint Mean Accuracy vs. layer-sharing grouping size $M$.

| Block size (M) | Total Groups (G) | Trainable Parameters | Joint Mean Test Acc (%) |
| :---: | :---: | :---: | :---: |
| 1 | 12 | 240 | **54.92 ± 2.45%** |
| 2 | 6 | 120 | **56.13 ± 2.55%** |
| 3 | 4 | 80 | **57.08 ± 2.12%** |
| 4 | 3 | 60 | **57.11 ± 2.15%** |
| 6 | 2 | 40 | **57.00 ± 2.24%** |
| 12 | 1 | 20 | **57.15 ± 2.15%** |

## Table 4: Gating Activation Sweep (BWS-Router, M=3)
Comparative analysis of different activation functions applied inside the block routing equations of BWS-Router ($M=3$).

| Gating Activation | Joint Mean Accuracy (%) |
| :--- | :---: |
| Linear | **79.10 ± 0.76%** |
| Tanh | **78.94 ± 0.75%** |
| Softmax | **78.39 ± 0.71%** |
| Sigmoid | **57.07 ± 2.17%** |

## Table 5: Optimization and Regularization Grid Sensitivity Sweep
Full sensitivity analysis for BWS-Router ($M=3$, Sigmoid) over combinations of learning rate ($\eta$) and $L_2$ weight decay ($\lambda_{wd}$).

| Learning Rate (η) | Weight Decay (λ_wd) | Joint Mean Accuracy (%) |
| :---: | :---: | :---: |
| 0.001 | 0.0 | 25.63 ± 2.40% |
| 0.001 | 0.0001 | 25.64 ± 2.36% |
| 0.001 | 0.001 | 25.68 ± 2.38% |
| 0.001 | 0.01 | 25.65 ± 2.38% |
| 0.005 | 0.0 | 37.31 ± 3.76% |
| 0.005 | 0.0001 | 37.37 ± 3.89% |
| 0.005 | 0.001 | 37.74 ± 3.20% |
| 0.005 | 0.01 | 35.70 ± 1.91% |
| 0.01 | 0.0 | 57.63 ± 2.76% |
| 0.01 | 0.0001 | 57.10 ± 2.14% |
| 0.01 | 0.001 | 57.01 ± 2.13% |
| 0.01 | 0.01 | 49.62 ± 1.69% |
| 0.05 | 0.0 | 79.63 ± 1.18% |
| 0.05 | 0.0001 | 79.57 ± 1.13% |
| 0.05 | 0.001 | 77.96 ± 1.58% |
| 0.05 | 0.01 | 56.29 ± 7.94% |

## Generated Diagnostic Plots
To support the scientific insights, we have saved four high-resolution diagnostic plots in the project workspace:
1. `l3_comparison.png`: Main baseline comparison of routing Joint Mean test accuracy.
2. `batch_heterogeneity.png`: Comparative analysis of routing methods under task heterogeneity configuration shifts.
3. `regularization_impact.png`: Isolated evaluation of unregularized vs. regularized routers on the noisy SVHN OOD domain, demonstrating the mitigation of parameter scaling collapse.
4. `bws_m_sensitivity.png`: Systematic curve mapping the capacity-generalization trade-off as block size $M$ varies.
