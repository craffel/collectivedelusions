# 1. Summary of the Paper

## Main Topic and Goal
The paper addresses a critical bottleneck in the practical deployment of multi-task model ensembling on resource-constrained edge hardware. Weight-space model merging (e.g., Task Arithmetic, Model Soups, and AdaMerging) combines multiple specialized task experts fine-tuned from a shared pre-trained backbone into a single model, incurring zero computational or memory overhead at inference time. However, to run on edge accelerators (like mobile TPUs, DSPs, and automotive ASICs), these merged models must undergo post-training quantization (PTQ). 

The key problem identified is **Cross-Schema Performance Degradation**: existing quantization-aware model merging methods (such as Q-Merge) optimize merging coefficients under a single simulated quantization operator (e.g., Symmetric Per-Channel) using the Straight-Through Estimator (STE). When deployed on heterogeneous edge hardware running mismatched compilers or different PTQ standards (such as Symmetric Per-Tensor or asymmetric schemas), the learned coefficients fail because they overfit to the specific, discrete rounding grid of the training operator.

The goal of the paper is to develop a training-free, multi-schema co-optimization framework called **OmniMerge** that produces robust, hardware-invariant model merging coefficients without requiring hardware metadata, extra test-time compute, or incurring inference-time latency/memory overhead.

## Proposed Approach
OmniMerge optimizes layer-wise merging coefficients $\Lambda \in [0, 1]^{K \times L}$ over a small, unlabeled calibration stream ($N_{\text{cal}} = 64$ images per task) using unsupervised test-time adaptation (prediction entropy minimization) combined with Task-Consensus Regularization (TCR). It introduces two core techniques to achieve cross-schema robustness:
1. **Stochastic Operator Sampling (SOS):** Instead of using a single static quantization operator during test-time adaptation, the active quantization operator is stochastically and uniformly sampled at each optimization step from a discrete pool of four hardware-standard PTQ schemas: Symmetric Per-Tensor, Symmetric Per-Channel, Asymmetric Per-Tensor, and Asymmetric Per-Channel. This acts as parameter-space data augmentation to prevent boundary-overfitting.
2. **Scale and Zero-Point Noise Perturbation (SZNP):** Multiplicative and additive Gaussian noise is injected into the scale factors ($\epsilon_s \sim \mathcal{N}(0, \sigma^2_{\text{scale}})$) and zero-point offsets ($\epsilon_z \sim \mathcal{N}(0, \sigma^2_{\text{zero}})$) during the forward pass. This dynamically shifts rounding boundaries to smooth the rugged, non-differentiable loss landscape and help gradient-based optimization escape fragile local minima.

During backward propagation, the Straight-Through Estimator (STE) is used to pass gradients to the continuous merging coefficients, while the scale and zero-point parameters (including noise) are treated as constants and detached from the computation graph. Once optimized (in 15 steps), the final merged model is compiled with zero noise ($\epsilon_s = 0, \epsilon_z = 0$) to maintain standard compatibility and zero inference overhead.

## Key Findings and Claims
- **Overfitting to Discretization Boundaries:** The authors empirically show that standard Q-Merge, when optimized strictly under Symmetric Per-Channel quantization, drops from 47.07% to 45.90% accuracy when deployed on Symmetric Per-Tensor quantization.
- **Superior Cross-Schema Robustness:** OmniMerge is claimed to completely close the cross-schema generalization gap, outperforming Q-Merge, Quantized AdaMerging, and Naive Merge-then-Quantize across all five target accelerators under 8-bit quantization. It achieves up to **50.78%** average multi-task accuracy (under Symmetric Per-Channel).
- **Out-of-Pool Generalization:** OmniMerge is evaluated on Double Quantization (which was not in the stochastic operator pool during optimization) and achieves **50.29%** accuracy, outperforming Q-Merge by **+3.71%** absolute, demonstrating true schema invariance.
- **Denoising Effect of Quantization:** The authors claim that discrete weight rounding can act as a beneficial noise filter rather than a purely lossy operation, showing that quantized OmniMerge (50.78%) can outperform both its unquantized FP16 optimized ceiling (46.68% for AdaMerging) and, in some cases, the exact unquantized model under the same coefficients (50.78% vs 50.39%).

## Explicit Contributions
1. **Identification of Cross-Schema Degradation:** Outlining how first-order optimization via STE under a single quantization operator causes coefficient overfitting to specific discretization boundaries.
2. **OmniMerge Framework:** A training-free, metadata-free, zero-inference-overhead co-optimization method combining Stochastic Operator Sampling (SOS) and Scale/Zero-Point Noise Perturbation (SZNP).
3. **Task-Consensus Regularization (TCR):** An unsupervised joint objective penalizing absolute coefficient deviation from starting uniform values and group-consensus task deviation.
4. **Empirical Evaluation:** Rigorous benchmarking of a `ViT-Tiny` backbone across four diverse classification tasks (MNIST, FashionMNIST, CIFAR-10, SVHN) under 8-bit quantization across 5 heterogeneous target schemas.
