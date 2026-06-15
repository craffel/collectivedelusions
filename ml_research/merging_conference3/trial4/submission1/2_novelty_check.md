# 2. Novelty and Originality Check

## 2.1 Conceptual Originality
The submission stands out for its conceptual departure from standard model-merging literature. Traditional model merging methods (such as Task Arithmetic, TIES-Merging, AdaMerging, and PolyMerge) operate strictly under the assumption that parameter space is flat, isotropic, and real-valued, using direct linear combinations or scalar weight interpolations.

In contrast, **PhaseMerge** proposes a wave-theoretic view of weight updates:
1. **Frequency Projection**: It utilizes the 2D Real Fast Fourier Transform (RFFT2D) to map real-valued weight matrices to the complex-valued Fourier domain.
2. **Phase-Amplitude Decoupling**: It decouples parameter matrices into magnitude (amplitude) and phase angle, allowing independent operations on both.
3. **Constructive and Destructive Interference**: By optimizing a continuous phase shift, the optimizer can rotate conflicting updates out-of-phase ($\Delta \phi \approx \pi$) to actively cancel parameter conflicts, or align synergistic features in-phase ($\Delta \phi \approx 0$).

This represents the first attempt to model neural network parameter blending as complex wave superposition. This shift is highly creative and original.

## 2.2 Algorithmic Innovations
The paper introduces two concrete algorithmic contributions:
- **Low-Dimensional Phase-Rotation Parameterizations**:
  - **U-PhaseMerge ($r=1$)**: Uses a single uniform scalar phase shift per task and layer (192 parameters), serving as a compact, low-dimensional phase regularizer.
  - **PhaseMerge ($r=2$)**: Uses a compact $2 \times 2$ phase grid, bilinearly upsampled to the full complex tensor's dimensions, to allow spatially smooth, low-frequency phase adjustments.
- **Symmetry-Preserving Frequency Mask**: The formulation enforces a strict mask to zero out phase adjustments on DC and Nyquist components. This is mathematically necessary because real-to-complex Fourier transforms constrain these frequencies to have zero imaginary parts (restricting valid phase angles to $\{0, \pi\}$). By zeroing these out, the formulation guarantees mathematically consistent, real-valued spatial reconstructions without relying on silent autograd imaginary-discarding.

## 2.3 Distinction from Prior Art
- **Vs. Real-Space Adaptive Merging (AdaMerging, RegCalMerge, PolyMerge)**: These optimize unconstrained coefficients in real space. PhaseMerge optimizes phase rotations in complex frequency space, introducing an implicit coordinate-locked **matrix-basis regularizer**.
- **Vs. Static Spectral Merging (FREE-Merging)**: FREE-Merging applies static, hard-coded low-pass/high-pass filters to task vectors before linear merging. The paper demonstrates that FREE-Merging performs catastrophically on the vision transformer benchmarks ($27.17 \pm 1.96\%$ vs. U-PhaseMerge's $42.83 \pm 1.76\%$). This demonstrates that *adaptive* continuous phase synchronization is necessary to manage expert feature alignments, and that static filtering is too rigid.

## 2.4 Verdict on Originality
The originality of the submission is **excellent**. It introduces a highly novel mathematical paradigm to weight-space model merging. Even though its final performance does not beat PolyMerge in this setup, the proposed continuous wave-superposition and phase-rotation concepts represent a refreshing, creative addition to the machine learning literature.
