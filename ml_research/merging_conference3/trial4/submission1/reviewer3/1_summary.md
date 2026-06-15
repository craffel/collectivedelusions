# 1. Summary of the Paper

## Main Topic and Goal
The paper addresses the challenge of post-hoc **model merging**, which aims to blend the parameters of multiple specialized expert networks (fine-tuned from a shared pre-trained backbone) into a single, unified multi-task model without the high cost of joint multi-task retraining. 

## Proposed Approach: PhaseMerge
Instead of merging parameters using linear interpolation or scalar scaling in static real-valued Euclidean space, the paper introduces a wave-theoretic frequency-domain framework called **PhaseMerge**. The approach consists of the following key technical steps:
1. **Fourier Projection:** The real-valued spatial task vectors ($\tau_k^l = W_k^l - W_{\text{pre}}^l$) for each expert $k$ and layer $l$ are projected into the complex frequency domain via the 2D Real Fast Fourier Transform (RFFT2D), yielding a complex-valued tensor decomposed into amplitude $A_k^l$ and phase $\theta_k^l$.
2. **Differentiable Phase Rotation:** The optimizer learns continuous phase shifts $\phi_k^l$ in $[-\pi, \pi]$ to rotate the phase angles while keeping amplitudes intact, allowing constructive/destructive wave superposition. Two parameterizations are evaluated:
   - **Uniform Phase Merge (U-PhaseMerge, $r=1$):** A single scalar phase shift per layer/task (192 or 196 parameters), acting as a low-dimensional global phase regularizer.
   - **Bilinear Continuous Phase Grids (PhaseMerge, $r=2$):** A compact $2 \times 2$ grid per layer/task, bilinearly upsampled to the full frequency tensor size (768 or 772 parameters), enforcing spatially smooth, low-frequency adjustments.
3. **Conjugate Symmetry Masking:** A strict symmetry-preserving mask ($M_{\text{sym}}$) is applied to ensure that phase shifts on DC (frequency $(0,0)$) and Nyquist components are zero. This guarantees that the reconstructed spatial matrices are strictly real-valued.
4. **Spatial Reconstruction and Scaling:** The phase-rotated frequency representations are mapped back to real space via the Inverse RFFT2D (IRFFT2D) and scaled by task-wise learnable coefficients $s_k$ to yield the final merged weights.
5. **Entropy Optimization under Quantization:** The model is projected through a post-training quantization (PTQ) operator to 8-bit or 4-bit precision. The phase parameters are optimized by minimizing unsupervised prediction entropy over a small calibration stream ($D_{\text{cal}}$) using Straight-Through Estimators (STE) for backpropagation.

## Key Findings and Claims
- **Superiority over Static Baselines:** PhaseMerge significantly outperforms the static Uniform Task Arithmetic (TA) and the static frequency-filtering baseline (FREE-Merging) across FP32, 8-bit PTQ, and 4-bit PTQ regimes.
- **Robustness under extreme 4-bit quantization:** The authors claim that the compact U-PhaseMerge ($r=1$) variant exhibits exceptional robustness under extreme 4-bit quantization and outperforms traditional real-space layer-wise optimizations (e.g., AdaMerging).
- **Stabilization under Data Scarcity:** The symmetry-preserving mask and compact parameterization help prevent the "Overfitting-Optimizer Paradox" when calibrating on small datasets ($M=4$ or $M=16$).
- **Generalizability to Schema Shift:** The smooth frequency-domain transformations are claimed to generalize well when evaluated on target schemas (e.g., 4-bit) different from the calibration schema (8-bit).

## Explicitly Claimed Contributions
1. Challenging the static Euclidean assumption and proposing a wave-theoretic model merging framework (PhaseMerge) based on complex wave superposition in Fourier space.
2. Formulating a highly compact Uniform PhaseMerge (U-PhaseMerge, $r=1$) that acts as a frequency-domain matrix-basis regularizer.
3. Designing a symmetry-preserving frequency mask to secure real-valued spatial reconstructions and stabilize optimization.
4. Conducting empirical evaluations on Vision Transformers (`vit_tiny_patch16_224`) across 4 conflicting classification tasks (MNIST, FashionMNIST, CIFAR-10, SVHN).
