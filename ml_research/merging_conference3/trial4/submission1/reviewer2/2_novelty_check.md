# 2. Novelty Check

## Key Novel Aspects
The main conceptual novelty of **PhaseMerge** is the shift from standard real-space weight merging (e.g., linear combination or interpolation) to **complex-valued frequency-space phase rotation**. Instead of treating weight updates as static spatial vectors, PhaseMerge treats them as complex wavefunctions, decoupling the update into amplitude (magnitude) and phase (alignment), and optimizing phase angles to achieve constructive or destructive interference. 

Additionally, the paper introduces a **symmetry-preserving frequency mask** to enforce conjugate symmetry, and proves a spatial-domain equivalence showing that uniform phase rotation in Fourier space is mathematically dual to a spatial coordinate rotation in a 2D subspace spanned by the task vector and its directional Hilbert transform.

## Delta from Prior Work
The paper positions PhaseMerge relative to several distinct lines of work:
1. **Adaptive Merging (AdaMerging, RegCalMerge):** These methods learn layer-wise linear coefficients ($\Lambda$) in real Euclidean parameter space. PhaseMerge's delta is that it optimizes phase rotation in the complex frequency domain. It claims this acts as a frequency-domain regularizer that avoids overfitting to validation noise ("Overfitting-Optimizer Paradox").
2. **Constrained Merging (PolyMerge, SplineMerge):** These methods constrain real-space merging coefficients using depth-wise polynomials or splines. PhaseMerge differs by constraining the optimization via frequency-space phase parameters (either uniform $r=1$ or smooth $2 \times 2$ bilinear grids $r=2$).
3. **Spectral/Fourier-domain Merging (FREE-Merging, SWUDI, STAR):** FREE-Merging applies static, non-adaptive high-pass and low-pass Fourier filters to task vectors to filter out redundant parameters. PhaseMerge's delta is that it is **differentiable and adaptive**, optimizing continuous phase-shift grids end-to-end to learn constructive and destructive interference patterns, rather than applying hard-coded, static frequency boundaries.

## Characterization of Novelty (Incremental vs. Significant)
While the wave-mechanics framing is highly creative, elaborate, and academically interesting, the overall novelty is **theoretically questionable** and **empirically incremental (or even regressive)**:

1. **Theoretical Incoherence on Dense Matrices:**
   The paper applies 2D FFT to dense weight matrices ($W \in \mathbb{R}^{H \times W}$). However, dense matrices are fundamentally permutation-invariant: swapping rows or columns preserves the network's function (provided corresponding inputs/outputs are swapped). In contrast, the Discrete Fourier Transform is highly coordinate-dependent, as it is constructed using complex exponentials over a rigid 2D grid. Swapping two rows of a dense matrix completely changes its 2D FFT representation, mixing real and imaginary parts. 
   Therefore, defining "frequencies" or "phase alignment" on dense weights is an artifact of the arbitrary indexing used in the software implementation, rather than a physical property of the network. While the authors try to bypass this by saying they "lock this permutation index to the base pre-trained model," this is a superficial fix. The underlying mathematical operations (like the directional Hilbert transform) remain tied to an arbitrary ordering of neurons. Applying Fourier wave-mechanics to non-spatial, dense weight matrices is conceptually flawed and lacks physical meaning. It is only physically justified for convolutional kernels, as noted in the appendix, but the core evaluation is on a Vision Transformer (which uses dense projections).

2. **Negative Empirical Delta Relative to Simpler Real-Space Methods:**
   The ultimate measure of novelty in a new parameterization is its ability to unlock better performance or efficiency. However, the proposed complex frequency-domain phase-rotation is completely dominated by **PolyMerge**—a simple, real-space depth-wise polynomial baseline. PolyMerge achieves $48.00\%$ FP32 accuracy, whereas U-PhaseMerge achieves $42.83\%$ (a massive $5.17\%$ absolute performance drop) and PhaseMerge ($r=2$) achieves $40.75\%$ ($7.25\%$ absolute drop).
   Thus, PhaseMerge introduces immense mathematical and computational complexity (RFFT2D, IRFFT2D, complex numbers, Straight-Through Estimators, symmetry-preserving masks) only to perform significantly worse than a straightforward 12-parameter real-space polynomial baseline. This severely diminishes the practical significance of the claimed novelty.

3. **Incremental Adaptive Extension over FREE-Merging:**
   The idea of applying Fourier transforms to task vectors for model merging is not new; it was introduced by **FREE-Merging**. While PhaseMerge makes this process adaptive by learning continuous phase rotations, the poor performance of both FREE-Merging ($27.17\%$) and PhaseMerge ($40.75\%$) compared to real-space baselines suggests that frequency-domain representations are fundamentally ill-suited for dense weight merging, making this adaptive extension of questionable utility.
