# 2. Novelty Check and Delta from Prior Work

## Delta from Prior Work
1. **Vs. Real-Space Merging (Task Arithmetic, Model Soups, AdaMerging, PolyMerge):**
   - Prior works operate entirely in the real-valued Euclidean parameter space, using static linear combinations or learning layer-wise real scalar coefficients. 
   - **PhaseMerge's Delta:** It rejects real-space scalar constraints entirely. It projects task vectors into complex-valued Fourier space, separating magnitude (amplitudes) and direction/alignment (phase angles). It updates the merged network via *differentiable phase rotations* that allow constructive or destructive interference of parameter updates.
2. **Vs. FREE-Merging:**
   - FREE-Merging applies static, non-adaptive Fourier filters (e.g., keeping only the lowest 85% of frequencies) to task vectors before linear merging.
   - **PhaseMerge's Delta:** PhaseMerge is *adaptive* and optimized end-to-end via gradient descent. Instead of hard-coded frequency filtering, it learns dynamic phase rotations to align tasks or filter out quantization noise.
3. **Mathematical Contribution:**
   - The paper provides a mathematical proof (Theorem 3.1) showing that applying a uniform phase-shift to the Real 2D Fourier representation is dual to a structured spatial coordinate rotation in the 2D subspace spanned by the task vector and its directional Hilbert transform.

## Characterization of Novelty
- **Conceptual Novelty: Significant.** The concept of viewing weight merging as continuous wave superposition and phase synchronization in the complex frequency domain is highly creative. The connection to the directional Hilbert transform is mathematically elegant and provides a solid theoretical foundation.
- **Practical/Structural Novelty: Moderate to Low.**
  - **Grid Representation Mismatch for Dense Weights:** The 2D FFT is inherently coordinate-dependent and assumes a physical, ordered 2D spatial grid (where neighboring elements have spatial correlation, like in pixels or 2D signals). However, the dense weight matrices of transformers (`qkv`, `proj`, `fc1`, `fc2`) are permutation-invariant and do not have an innate physical 2D grid structure. Swapping two rows/columns in a dense weight matrix does not change its mathematical function, but it completely alters its 2D Fourier transform. Applying a 2D FFT and spatial upsampling ($2\times 2$ grid) to dense weights is structurally contrived. While locking the coordinate system to a shared pre-trained initialization ($W_{\text{pre}}$) is a valid optimization workaround, it does not resolve this fundamental conceptual-structural mismatch.
  - **Incremental Empirical Value:** From a practical deployment perspective, the proposed complex-domain method underperforms a simpler, real-space baseline (**PolyMerge**) by a large margin (approx. 5-6% absolute accuracy). A practitioner looking for high-performance and simple deployment would prefer PolyMerge's 12-dimensional polynomial scaling in real space over PhaseMerge's complex 2D FFT, inverse FFT, straight-through estimation, and conjugate symmetry masking. Thus, while the mathematical framing is highly novel, its practical utility does not represent a significant advance over existing real-valued baselines.
