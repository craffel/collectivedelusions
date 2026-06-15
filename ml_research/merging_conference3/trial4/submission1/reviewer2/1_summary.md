# 1. Summary of the Paper

## Main Topic and Approach
The paper introduces **PhaseMerge**, a frequency-domain model-merging framework designed to blend multiple task-specific expert neural networks (fine-tuned from a shared pre-trained backbone) without joint multi-task retraining. 

To address parameter conflicts (destructive interference), "The Overfitting-Optimizer Paradox" in test-time calibration, and target post-training quantization (PTQ) schema shifts, PhaseMerge projects real-valued spatial task vectors (weight updates) into complex-valued Fourier space using the 2D Real Fast Fourier Transform (RFFT2D). This decomposes the parameter updates into amplitude and phase. 

The framework then applies a differentiable phase-rotation mechanism to the phase angles while keeping amplitudes intact, using either:
1. **Uniform Phase Rotation (U-PhaseMerge, $r=1$):** A single, uniform scalar phase shift parameter per layer and task.
2. **Continuous Phase-Shift Grids (PhaseMerge, $r=2$):** A compact $2 \times 2$ phase grid per layer, bilinearly upsampled to the frequency tensor dimensions.

Both configurations are constrained by a symmetry-preserving frequency mask that zeroes out phase-shifts on DC and Nyquist components to ensure mathematically consistent real-valued spatial reconstruction via the Inverse RFFT2D (IRFFT2D). Finally, the reconstructed task updates are combined using learnable task-wise scaling coefficients ($s_k$), and the merged weights are added to the pre-trained backbone. 

For post-training quantization, the merged model is projected through a PTQ operator, and the phase parameters are optimized by minimizing unsupervised prediction entropy on a small calibration stream ($D_{\text{cal}}$) using Straight-Through Estimation (STE) for backpropagation.

## Key Findings
- **Competitive Multi-Task Performance:** PhaseMerge and U-PhaseMerge outperform static linear merging (Uniform Task Arithmetic) and static frequency-domain filtering (FREE-Merging) across unquantized (FP32), 8-bit, and 4-bit PTQ regimes on a Vision Transformer backbone (`vit_tiny_patch16_224`) across four vision datasets (MNIST, FashionMNIST, CIFAR-10, SVHN).
- **Outperformed by PolyMerge:** A simpler, real-space depth-wise polynomial baseline (PolyMerge) consistently and significantly outperforms all PhaseMerge variants across all evaluation setups by approximately $5\%$ to $7\%$ absolute accuracy.
- **Ablation on Grid Dimension:** The compact U-PhaseMerge ($r=1$, 192 parameters) outperforms the larger PhaseMerge ($r=2$, 768 parameters) by $+2.08\%$ in FP32 and $+1.50\%$ in 8-bit, suggesting that lower-dimensional phase representations serve as a stronger, more robust regularizer in dense layers that lack spatial coordinates.
- **Sample Complexity and Optimization Instability:** Under larger calibration streams ($M=32$), unregularized PhaseMerge optimization experiences significant instability (standard deviation $\ge 4.1\%$), which is mitigated by introducing an $L_2$ phase decay regularization penalty ($\gamma = 10^{-4}$).

## Explicitly Claimed Contributions and Accompanying Evidence
1. **Frequency-Domain Parameter Superposition Paradigm:** Proposes a wave-theoretic formulation that represents parameter updates as complex wavefunctions in Fourier space, allowing continuous, differentiable phase-rotation to align complementary features and cancel conflicting elements. 
   - *Evidence:* Section 3 formulation, Euler's formula representation, and Section 3.4 proof of mathematical equivalence to spatial coordinates rotation using the directional Hilbert transform.
2. **Uniform PhaseMerge (U-PhaseMerge, $r=1$):** A highly parameter-efficient phase-rotation formulation (196 parameters) acting as a stable matrix-basis regularizer.
   - *Evidence:* Section 3.3 formulation and experimental evaluation in Tables 1, 2, 3, and 4.
3. **Symmetry-Preserving Frequency Mask:** A mask designed to restrict phase rotations on DC and Nyquist components to zero, preserving real-valued spatial reconstructions and stabilizing optimization.
   - *Evidence:* Section 3.4 mathematical formulation of the mask. *Critique:* The paper claims this mask "dramatically stabilizes optimization and prevents performance degradation" (citing Section 3.5/4.3), but no empirical ablation or quantitative evidence omitting this mask is provided in the experimental results.
4. **Empirical Evaluation on Vision Transformers:** Validation on `vit_tiny_patch16_224` across 4 highly conflicting datasets, showing robustness under quantization (FP32, 8-bit, 4-bit) and calibration stream sweeps.
   - *Evidence:* Tables 1-3, and Figures 1-3.
