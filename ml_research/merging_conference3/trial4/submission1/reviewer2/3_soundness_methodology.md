# 3. Soundness and Methodology

## Clarity of the Description
The methodology of **PhaseMerge** is written with a high degree of mathematical detail. The projection equations (RFFT2D), decomposition into amplitude and phase, parameterizations of $r=1$ and $r=2$ phase adjustments, spatial equivalence (Theorem 1), and optimization objectives (prediction entropy and Straight-Through Estimation) are clearly stated. 

However, several implementation details are omitted, which hinders full clarity and reproducibility:
- **Reshaping of Dense Weights:** A dense weight matrix $W^l$ can have various non-square dimensions (e.g., in a ViT-Tiny, the QKV projection matrix has shape $192 \times 576$, while MLP projection matrices have shapes $192 \times 768$ or $768 \times 192$). How are these matrices padded or reshaped to apply the 2D Real FFT? Since the 2D FFT is highly coordinate-dependent, the choice of padding, transposition, or reshaping directly impacts the frequency components, the phase-shift grid alignment, and the reconstructed spatial matrices. This arbitrary choice is not discussed.
- **PTQ Specifications:** The exact details of the post-training quantization ($Q_{\text{opt}}$) are vague. The authors mention "asymmetric channel-wise 8-bit or 4-bit quantization," but do not specify whether scales and zero-points are re-calibrated at each step of optimization, or if they are static. Since STE is used to propagate gradients through the quantized weights, the dynamics of scale/zero-point adjustments are crucial for understanding the optimization.

## Appropriateness of Methods and Potential Technical Flaws
While mathematically elaborate, the methodology contains several fundamental technical and conceptual flaws:

### 1. Conceptual Incoherence: Fourier Transforms on Permutation-Invariant Weights
The application of the 2D Discrete Fourier Transform (DFT) relies on the assumption of **spatial/grid topology**, where adjacent elements share a continuous coordinate relationship (like pixels in an image). Dense layers in neural networks are fundamentally **permutation-invariant**—the ordering of rows and columns is an arbitrary artifact of the implementation's memory layout. 
Swapping two neuron indices changes the spatial layout of the weight matrix, completely reshuffling its 2D FFT representation and mixing real/imaginary parts. Therefore, assigning physical concepts like "frequencies," "high-pass filtering," "wave coherence," or "phase alignment" to dense weight matrices is mathematically incoherent. Although the authors state that they "lock this permutation index to the base pre-trained model," this does not resolve the core issue: the "frequencies" being rotated are entirely arbitrary and dependent on the initial software implementation's neuron indexing.

### 2. The "Overfitting-Optimizer Paradox" Contradiction
The primary motivation of PhaseMerge is to avoid the Overfitting-Optimizer Paradox under extreme data scarcity by using a highly constrained, low-dimensional phase-rotation space. However, a close look at the empirical results in Table 2 contradicts this claim:
- As the calibration stream size $M$ increases from $4$ to $32$, the performance of **U-PhaseMerge ($r=1$) actually degrades and becomes highly unstable**:
  - $M=4$: $42.42 \pm 1.64\%$
  - $M=16$: $42.33 \pm 1.76\%$
  - $M=32$: $40.67 \pm 3.65\%$ (accuracy drops by $1.75\%$ and standard deviation spikes to $3.65\%$).
- In contrast, the unconstrained baseline **AdaMerging improves steadily** as more data is provided:
  - $M=4$: $40.75 \pm 1.24\%$
  - $M=16$: $41.67 \pm 1.45\%$
  - $M=32$: $42.50 \pm 1.59\%$.
- A sound optimization framework should exhibit *better* generalizability and lower variance when provided with more calibration data. The fact that PhaseMerge degrades and becomes highly unstable as $M$ increases indicates severe optimization pathology, such as optimization drift or chaotic landscape navigation in the complex frequency domain. This completely undermines the claim that PhaseMerge successfully resolves the Overfitting-Optimizer Paradox.

### 3. Hyper-Fragile Optimization Window
To prevent "class collapse" during unsupervised prediction entropy minimization, the authors state that they "restrict the test-time calibration budget to a highly conservative duration (exactly 5 optimization steps)." 
If an optimization framework is so fragile that running it for more than 5 steps triggers catastrophic class collapse or parameter degeneration, the underlying optimization objective and parameterization are highly unstable. Relying on an extremely short, arbitrary stopping criterion (5 steps) as a "structural design constraint" is an ad-hoc band-aid rather than a robust, principled solution.

### 4. Overstated Claim on Symmetry-Preserving Mask
In Section 3.4, the authors claim that the symmetry-preserving frequency mask $M_{\text{sym}}$ "dramatically stabilizes optimization in complex frequency-space and prevents performance degradation." They refer to "our experimental sweeps (Section 3.5)" (which actually describes the entropy loss and contains no sweeps). 
A search through the entire paper reveals that **there is no ablation study or quantitative comparison showing results without this mask**. Claiming that a specific component is vital for stabilizing optimization without providing any empirical evidence of its omission is a major scientific flaw and represents an overstated, unsupported claim.

### 5. Straight-Through Estimator (STE) Noise in Fourier Space
Backpropagating gradients from a non-differentiable 4-bit rounding operator through the IRFFT2D back to complex-valued phase angles using STE is highly problematic. Rounding operators introduce highly discontinuous, step-like loss landscapes. When mapped back through the non-linear, coordinate-dependent inverse Fourier transform, these gradients are likely to be extremely noisy and chaotic. This is a highly probable cause for the massive performance drop and instability observed as the calibration steps/data increase. The paper provides no analysis of gradient quality, variance, or stability in this complex chain.

## Reproducibility
The reproducibility of the work is **fair**. While the mathematical equations are detailed, reproducing the exact results would be challenging without:
1. The exact code or description of the dense weight matrix reshaping and padding before the RFFT2D.
2. The exact implementation details of the asymmetric channel-wise PTQ operator (scale/zero-point calculation frequency).
3. The exact optimization setup, such as weight decay or specific Adam hyperparameter settings.
