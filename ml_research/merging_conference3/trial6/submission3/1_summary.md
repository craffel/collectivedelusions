# 1. Paper Summary and Key Contribution

## Overview of the Paper
This paper tackles the capacity-generalization trade-off in dynamic weight-space model merging. Static model merging techniques combine specialized expert networks into a single multi-task model at zero additional inference cost but remain fundamentally task-agnostic, using the exact same weight configuration for all inputs. Existing dynamic model merging methods predict sample-dependent routing coefficients at runtime to address this. However, they suffer from two severe empirical flaws: (1) **Cascading Representation Drift and Coefficient Ruggedness** caused by learning independent unshared parameters for each of the $L$ layers, which can overfit on small calibration splits and cause coefficients to diverge abruptly layer-by-layer; and (2) **Parameter Scaling Excess** which leads to high computational overhead and overparameterization. Other alternatives using quantum wavefunction superposition (QWS-Merge) introduce non-monotonic gating dynamics but exhibit extreme optimization ruggedness and high sensitivity to random seeds.

To solve these limitations, the authors introduce the **Block-wise Weight-Sharing Router** (**BWS-Router**), which groups the $L$ layers of the model into $G = L/M$ uniform blocks and shares routing weights within each block. Under the BWS-Router, parameters are drastically compressed, routing forward pass overhead is reduced by up to 91.7%, and layer-to-layer routing coefficient ruggedness is mathematically mitigated. The BWS-Router is combined with an unsupervised low-dimensional PCA state pre-projector and bounded independent sigmoidal gating, providing exceptional parameter efficiency, OOD deactivation, and multi-task feature co-activation robustness.

## Summary of Key Achievements and Metrics
The paper demonstrates outstanding empirical rigour through an exhaustive grid sweep of over 1,280 experiment configurations across 5 independent seeds. The main empirical results include:

*   **Virtual-Layer Sandbox Benchmark (MNIST, FashionMNIST, CIFAR-10, SVHN):**
    *   **BWS-Router ($M=3$)** achieves a Joint Mean Accuracy of **79.57 $\pm$ 1.14%** (and up to **79.63 $\pm$ 1.18%** under the optimal learning rate of $\eta = 5 \cdot 10^{-2}$ and $\lambda_{wd} = 0$) using only 80 parameters. This represents a **66.7% parameter reduction** compared to the unshared baseline (240 parameters).
    *   **BWS-Router ($M=12$, Global Shared)** achieves a Joint Mean Accuracy of **79.60 $\pm$ 1.15%** using only 20 parameters. This represents a **91.7% parameter reduction** with zero loss in accuracy.
    *   **Static Uniform Merging** completely collapses under severe weight-space semantic conflicts, achieving only **23.56 $\pm$ 2.91%** Joint Mean Accuracy.
    *   **L3 Linear Unregularized Baseline** achieves **79.14 $\pm$ 0.77%** Joint Mean Accuracy, meaning block sharing slightly improves or maintains the unshared upper performance limit with a fraction of the parameters.
*   **Physical Sequential Weight-Space Model Merging across 3-layer MLP Experts:**
    *   **BWS-Router ($M=3$)** achieves **45.26 $\pm$ 10.11%** (homogeneous stream) and **43.20 $\pm$ 22.49%** (heterogeneous mixed-batch stream) Joint Mean Accuracy.
    *   It completely outperforms **Static Uniform physical merging** (**17.88 $\pm$ 3.78%** Joint Mean Accuracy).
    *   Under mixed heterogeneous streams, BWS-Router ($M=3$) dramatically boosts accuracy by **+10.93% absolute** over the unshared physical baseline ($M=1$, which collapses to **32.27 $\pm$ 21.28%**).
*   **Gating Activation and Sluggishness Analysis:**
    *   While Softmax achieves **80.56 $\pm$ 0.72%** under optimal learning rates in the closed sandbox due to implicit sum-to-one regularization, it fails under OOD inputs, where it is mathematically forced to inject a strict $1.0$ coefficient sum of expert noise.
    *   **Sigmoidal Gating (Ours)** deactivates task vectors under OOD inputs (gating sum of **0.4584 $\pm$ 0.0382** vs. Softmax's **1.0000 $\pm$ 0.0000**) and supports concurrent multi-task co-activation.
    *   **Negative Bias Initialization ($B_{group} = -2.0$)** successfully resolves Sigmoid's optimization sluggishness under moderate learning rates ($\eta = 10^{-2}$), boosting Joint Mean Accuracy from **57.25%** to **74.50 $\pm$ 1.99%** (+17.25% absolute gain) by establishing a sparse, inactive default state.
*   **Variance Stabilization under Deep Physical Sequential Propagation:**
    *   **Sequential Smoothing Regularization ($\lambda_{\text{smooth}} = 10^{-2}$)** successfully reduces seed-specific standard deviation in physical sequential merging from **21.28%** to **13.41%** under mixed streams while maintaining a robust, ceiling-competitive accuracy of **36.48%**. This is shown to be highly superior to **Residual Gating Links** which stabilize variance but severely degrade mean performance by forcing coefficients towards a static task-agnostic average.
*   **PCA Dimension and Kernel Sensitivity:**
    *   Sweeping PCA dimension $d$ reveals an optimal sandbox sweet spot at $d \approx K$ ($d = 4$, achieving **79.57 $\pm$ 1.14%**), but physical sequential weight merging sees monotonic improvements up to $d = 16$ (achieving **48.16 $\pm$ 7.68%** homogeneous and **46.17 $\pm$ 20.69%** heterogeneous) because deeper sequential propagation is highly sensitive to representation distortion.
    *   Kernel PCA sweeps (RBF, Cosine, Polynomial) yield statistically identical results to linear PCA, confirming linear PCA is a highly sufficient, stable, and parameter-free projection baseline.
*   **Expert Scaling Sweep ($K \le 10$):**
    *   Under $K=10$ expert tasks, BWS-Router maintains a robust **41.25 $\pm$ 5.18%** Joint Mean Accuracy, dramatically outperforming Static Uniform merging which collapses to **11.56 $\pm$ 0.67%** (+29.69% absolute improvement).
*   **Inference Latency Profile (Vision Transformer Pilot):**
    *   A PyTorch pilot demonstration on a physical Vision Transformer backbone (\texttt{vit\_tiny\_patch16\_224}, $L=12$, $D=192$, $K=4$, $B=16$) shows that **Coarse-to-Fine BWS-Router** reduces the dynamic merging overhead from 190.01 ms (under uniform BWS-Router) down to only **110.65 ms**, representing a **17.2% overall latency reduction** and a **75% parameter reduction** compared to unshared configurations.
