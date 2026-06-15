# 3. Soundness & Methodology Evaluation

## Methodological Clarity
The paper is generally well-structured and written with high mathematical formalism. It walks the reader through weight-space deconstruction, 2D RFFT, phase-rotation, and the IRFFT reconstruction. However, beneath the dense wave-theoretic jargon, there are severe conceptual and technical inconsistencies.

## Conceptual and Technical Flaws

### 1. Inherent Theoretical Mismatch (Arbitrary Coordinates for 2D FFT)
The most significant soundness issue is the application of the 2D Discrete Fourier Transform (and corresponding phase rotations) to the weight matrices of *dense layers*. 
* Unlike images or convolutional kernels, the rows and columns of dense weight matrices (such as the `qkv` projection or the MLP layers in a ViT) have no physical spatial layout. The order of the neurons is completely arbitrary.
* The 2D FFT treats these rows and columns as physical, topologically aligned spatial dimensions. Any 2D frequency, amplitude, or phase rotation is heavily dependent on the specific layout.
* If you apply a permutation to the neurons of a dense layer (which results in an identical functional network), the 2D FFT changes completely. This means that a uniform phase shift $\phi_k^l$ or a continuous $2 \times 2$ grid shift ($r=2$) operates on an arbitrary coordinate system.
* Since adjacent weights in a dense matrix do not represent adjacent physical positions, the "spatial frequency smoothing" grid ($r=2$) has no physical validity. This theoretical mismatch is directly reflected in the empirical results, where PhaseMerge ($r=2$) consistently performs worse than the simpler U-PhaseMerge ($r=1$).

### 2. Contradiction Regarding the "Overfitting-Optimizer Paradox"
The paper argues that optimizing unconstrained layer-wise coefficients (like AdaMerging) causes the optimizer to overfit to transductive noise in small calibration streams ($M \le 16$), leading to "noisy, jagged parameter profiles." 
* **Parameter Count Parity:** However, AdaMerging optimizes $L \cdot K$ parameters (192 parameters for 48 layers and 4 tasks). Our proposed U-PhaseMerge ($r=1$) optimizes $L \cdot K + K$ parameters (196 parameters).
* Since both methods optimize roughly the same number of unconstrained layer-wise parameters over the exact same small calibration dataset ($M=16$) using Adam and prediction entropy, there is no clear theoretical reason why U-PhaseMerge should be immune to overfitting.
* Indeed, the empirical results show that the performance of U-PhaseMerge is highly comparable to AdaMerging ($42.83 \pm 1.76\%$ vs $42.00 \pm 0.89\%$ under FP32; and $37.42 \pm 1.94\%$ vs $37.50 \pm 1.22\%$ under 4-bit PTQ, where AdaMerging actually wins). The "overfitting" is not resolved by shifting to the frequency domain; both methods perform nearly identically.
* In contrast, **PolyMerge**—which actually solves the overfitting problem by constraining the search space to a microscopic 12 parameters total using a continuous depth-wise polynomial—dramatically outperforms both methods by **5.17% to 6.00% absolute accuracy**. This demonstrates that the paper's claim of Fourier phase rotation acting as an effective "matrix-basis regularizer" is weak compared to actual parameter-space regularization.

### 3. Purely Hypothetical "PolyPhaseMerge" Framework
To bridge the empirical gap to PolyMerge, the paper devotes significant space in the Appendix (Section A.3) and mentions in the main text a hybrid model called **PolyPhaseMerge**.
* However, PolyPhaseMerge is **purely speculative** and is not implemented or evaluated in the paper. 
* Proposing a hypothetical method to explain away why the proposed method underperforms a simpler baseline by $6\%$ is not a sound scientific contribution. The paper must be evaluated on its actual implemented models, which fail to beat the state-of-the-art.

### 4. Overstated Claims on "Noise-Cancelling" and "Active Phase-Cancellation"
The paper's title and text repeatedly use the terms "noise-cancelling," "constructive wave alignment," and "active phase-cancellation." 
* However, there is **no direct empirical or analytical proof** provided that such phase-cancellation actually occurs. The authors do not visualize the learned phases, nor do they demonstrate that conflicting task updates are rotated out of phase by approximately $\pi$.
* The optimization simply minimizes prediction entropy. The parameters could easily be adjusting amplitudes and mixing updates in an arbitrary way (via the Hilbert transform dual) rather than executing structured physical wave cancellations.

## Reproducibility Gaps
While the mathematical formulation is present, critical implementation details required for reproducibility are completely missing:
1. **Matrix Reshaping and Dimensions:** The targeted dense matrices in a Vision Transformer (`vit_tiny_patch16_224`) have diverse shapes that are not square (e.g., `qkv` is `[576, 192]`, `fc1` is `[768, 192]`, and `fc2` is `[192, 768]`).
   * Applying 2D FFT requires either treating these 2D shapes directly or reshaping them. The paper does not specify whether the 2D FFT is applied to these original non-square shapes, or if they are padded, reshaped, or flattened.
   * Since the 2D Fourier transform shape heavily influences the frequency grid coordinates and conjugate symmetry boundary coordinates, omitting these details makes exact replication of the frequency grid upsampling ($r=2$) and conjugate symmetry masking impossible.
2. **Post-Training Quantization (PTQ) Details:** The exact PTQ setup (e.g., min-max calibration, number of bins, grouping, or standard symmetric/asymmetric channel-wise configuration details) is not specified.
