# 1. Summary of OmniMerge

## Paper Topic & Scope
The paper addresses the challenge of deploying multi-task models on resource-constrained edge hardware. Specifically, it focuses on weight-space model merging (e.g., Task Arithmetic, Model Soups, AdaMerging) as a zero-overhead alternative to deploying separate large-scale models. The main hurdle targeted is **hardware heterogeneity** under post-training quantization (PTQ), where different edge accelerators and runtime compilers employ incompatible PTQ standards (e.g., Symmetric vs. Asymmetric, Per-Channel vs. Per-Tensor).

## Proposed Methodology (OmniMerge)
OmniMerge is a training-free, multi-schema stochastic co-optimization framework for robust model merging. It does not require hardware metadata or extra test-time compute. Its core mechanisms are:
1. **Stochastic Operator Sampling (SOS):** During test-time adaptation, the active quantization operator is stochastically and uniformly sampled from a discrete *Stochastic Operator Pool* $\mathcal{Q} = \{Q_{\text{sym, tens}}, Q_{\text{sym, chan}}, Q_{\text{asym, tens}}, Q_{\text{asym, chan}}\}$. This acts as parameter-space data augmentation to prevent continuous merging coefficients from overfitting to a single discretization grid.
2. **Scale and Zero-Point Noise Perturbation (SZNP):** Gaussian noise is dynamically injected into scale ($\sigma_{\text{scale}} = 0.01$) and zero-point ($\sigma_{\text{zero}} = 0.02$) parameters during test-time calibration to smooth out the non-differentiable rounding landscape.
3. **Unsupervised Shannon Prediction Entropy:** Minimized over a tiny, unlabeled calibration stream ($N_{\text{cal}} = 64$ images per task).
4. **Task-Consensus Regularization (TCR):** Integrates penalties to restrain parameter drift and encourage inter-task consensus ($\beta = 0.1$, $\gamma = 0.5$).
5. **Straight-Through Estimator (STE) Gradient Flow:** Autograd backpropagates gradients strictly through rounded weights, detaching scale and zero-point min/max operations to maintain stability.

## Key Empirical Findings
- **Cross-Schema Performance Degradation:** Direct optimization under a single quantization operator (as done by Q-Merge) overfits to that specific operator's rounding boundaries. Mismatched deployment (e.g., Q-Merge optimized under Symmetric Per-Channel but deployed on Symmetric Per-Tensor) results in significant performance drops.
- **State-of-the-Art Performance:** Evaluated on a `ViT-Tiny` backbone across 4 classification tasks (MNIST, FashionMNIST, CIFAR-10, SVHN) under 8-bit quantization. OmniMerge achieves up to **50.78%** average accuracy, outperforming Q-Merge, Quantized AdaMerging, and Naive M-then-Q across all 5 target schemas (Symmetric/Asymmetric, Per-Tensor/Per-Channel, Double Quantization).
- **Out-of-Pool Generalization:** Demonstrates robust performance on Double Quantization, which was excluded from the stochastic operator pool during test-time co-optimization.
- **Quantization as Regularization:** Reports that discrete weight discretization can act as a beneficial noise filter, allowing quantized models to occasionally outperform their unquantized counterparts.

## Explicitly Claimed Contributions (with Evidence)
1. **Demonstration of Cross-Schema Performance Degradation:** Backed by evidence in Table 1 showing Q-Merge accuracy dropping from 47.07% (matched) to 45.90% (mismatched) under the Symmetric Per-Tensor operator.
2. **Introduction of OmniMerge:** Formulation of a unified framework integrating Stochastic Operator Sampling (SOS) and Scale/Zero-Point Noise Perturbation (SZNP).
3. **Comprehensive Empirical Evaluation:** Outperforming state-of-the-art baselines across 5 diverse target post-training quantization schemas, closing the cross-schema generalization gap.
4. **Analysis of Weight Denoising:** Control experiments showing that quantized OmniMerge can outperform its unquantized FP16 baseline (50.78% vs. 50.39% under Symmetric Per-Channel).
