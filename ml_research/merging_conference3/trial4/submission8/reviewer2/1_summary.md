# Evaluation Component 1: Summary of the Paper

## Main Topic
The paper addresses the challenge of post-hoc model merging—specifically, test-time adaptation (TTA) of layer-wise merging coefficients—under downstream deployment conditions involving post-training quantization (PTQ). 

## Approach
The authors identify a critical vulnerability in standard unregularized TTA (e.g., AdaMerging), which they term **Quantization-Operator Overfitting**. While unconstrained test-time adaptation can find coefficients that maximize multi-task accuracy in high-precision (FP32) formats, the optimization process converges to extremely sharp local minima. When these merged models are subjected to PTQ (e.g., INT8 or INT4 formats), the injected rounding noise triggers a catastrophic collapse in multi-task accuracy.

To solve this, the authors propose **CR-PolySACM (Clipping-Regularized Sharpness-Aware Subspace Model Merging)**, which combines:
1. **Global Structural Subspace Constraints (PolyMerge):** Restricting layer-wise blending coefficients to a low-degree polynomial of network depth:
   $$\lambda_k^l = \sigma\left(a_k + b_k \left(\frac{l-1}{L-1}\right) + c_k \left(\frac{l-1}{L-1}\right)^2\right)$$
   This parameterization reduces the optimization search space from $L \times K$ independent variables (56 in this setup) to a compact $3 \times K$ parameter matrix (12 variables), preventing transductive overfitting on small test-time calibration streams.
2. **Local Landscape Flatness Optimization (CR-SACM):** Explicitly minimizing local loss sharpness in the parameter space of the blending coefficients using a first-order minimax approximation (SACM) of sharpness.
3. **Clipping-Regularized Scale Balancing:** Identifying and resolving a fundamental **task-vector norm scale pathology** where unnormalized sharpness optimization is blind to highly sensitive, low-norm layers (such as final layer normalization, where the task-vector norm is $\approx 0.014$, nearly $50\times$ smaller than intermediate transformer layers). Standard scale normalization scales the adversarial perturbation inversely by the task-vector norm, which leads to gradient explosion. CR-SACM resolves this by clipping the task-vector norms to a robust minimum floor ($\beta = 0.10$), balancing scale sensitivity across layers without numerical instability.

## Key Findings
- **Overparameterization and Overfitting:** Standard unconstrained TTA (AdaMerging, RegCalMerge) suffers from extreme overfitting to the tiny calibration stream ($N=64$), leading to sharp minima that are highly fragile under quantization noise.
- **The Power of Subspace Constraints:** Restricting blending coefficients to a depth-dependent polynomial subspace (PolyMerge) provides an incredibly strong regularizer, outperforming all layer-wise adaptive merging baselines by $+8\%$ to $+9\%$ across all precision formats (including FP32, INT8, and INT4).
- **Synergy of Flatness and Subspaces:** Under aggressive 4-bit quantization (INT4 symmetric per-channel), CR-PolySACM achieves a joint mean accuracy of **19.07%**, outperforming standard PolyMerge (**18.10%**) by nearly **+1.0%**, demonstrating that local flatness optimization within a stable, low-dimensional subspace provides crucial robustness safeguards under severe noise.
- **Scale Pathology Resolution:** Correcting scale-blindness via CR-SACM allows unconstrained sharpness-aware adaptation (HessMerge) to consistently outperform standard AdaMerging across all six target schemas (+1.36% in FP32).

## Explicitly Claimed Contributions
1. **Quantization-Operator Overfitting Identification:** Conceptualizing and demonstrating that unregularized test-time coefficient optimization converges to sharp minima that collapse under post-training quantization.
2. **Task-Vector Norm Scale Pathology Definition:** Mathematically analyzing weight-space sharpness optimization to show that unnormalized flatness regularizers are blind to low-norm layers, leading to optimization instability or collapse.
3. **Clipping-Regularized SACM (CR-SACM):** Designing a robust perturbation mechanism that balances scale sensitivity across layer groups and prevents gradient explosion.
4. **CR-PolySACM Unified Framework:** Proposing a model merging framework integrating structured depth-dependent polynomial subspace constraints with local CR-SACM flatness optimization.
5. **Comprehensive Empirical Evaluation:** Validating the proposed framework across six quantization schemas (FP32 down to INT4) on a Vision Transformer (ViT-Tiny) backbone across four diverse visual classification domains (MNIST, FashionMNIST, CIFAR-10, SVHN).
