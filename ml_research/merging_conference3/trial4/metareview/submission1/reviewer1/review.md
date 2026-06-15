# Peer Review of "PhaseMerge: Fourier Phase Interference for Noise-Cancelling Model Merging"

## 1. Summary of the Paper
The paper proposes **PhaseMerge**, a frequency-domain model-merging framework designed to combine specialized neural network expert models (fine-tuned from a shared pre-trained base) into a single model. Instead of linearly interpolating task vectors in real spatial coordinate space, PhaseMerge projects task updates into the complex-valued Fourier domain using the 2D Real Fast Fourier Transform (RFFT2D). This decouples updates into amplitude and phase. 

The paper explores two differentiable phase-rotation parameterizations:
1. **Uniform Phase Rotation (U-PhaseMerge, $r=1$):** Learning a single uniform phase-shift per layer and task (196 parameters).
2. **Bilinear Continuous Phase Grids (PhaseMerge, $r=2$):** Learning a $2 \times 2$ phase-shift grid per layer, bilinearly upsampled to the frequency tensor's dimensions (772 parameters).

The phase shifts are constrained using a symmetry-preserving mask on the DC and Nyquist components to ensure mathematically consistent real spatial reconstructions via IRFFT2D. The parameters are optimized over a small, unsupervised calibration stream using a prediction entropy minimization objective. The authors evaluate their framework on a Vision Transformer Tiny (`vit_tiny_patch16_224`) backbone across four conflicting image classification tasks (MNIST, FashionMNIST, CIFAR-10, SVHN) under FP32, 8-bit PTQ, and 4-bit PTQ regimes.

---

## 2. Overall Recommendation
* **Overall Rating:** **2: Reject**
* **Soundness:** **Fair**
* **Presentation:** **Excellent**
* **Significance:** **Poor**
* **Originality:** **Good**

### Justification of Rating
While the paper is exceptionally well-written, mathematically polished, and conceptually creative, it suffers from a fundamental theoretical mismatch, severe scale limitations in its evaluation, and a lack of empirical competitiveness. Specifically, the strongest baseline (**PolyMerge**) consistently and substantially outperforms the proposed PhaseMerge and U-PhaseMerge frameworks by **5.17% to 6.00% absolute accuracy** across all tested configurations. Since a simpler, existing real-space baseline from prior work achieves vastly superior performance under all settings, the practical significance of the proposed method is very low. Additionally, the core theoretical assumption—applying 2D Fourier transforms to dense, permutation-invariant layers—is mathematically arbitrary, as dense weights lack any physical coordinate layout. Combined with the toy scale of the experimental setup (using a 5.7M parameter model and test sets of only 100 samples), the empirical claims are statistically fragile and fail to justify the proposed wave-theoretic complexity.

---

## 3. Strengths and Weaknesses

### Major Strengths
1. **Conceptual Originality:** Viewing weight-space model merging through a wave-interference lens in complex Fourier space is highly creative and represents an interesting departure from standard static linear averaging.
2. **Mathematical Rigor:** The paper is mathematically solid, providing a rigorous spatial dual equivalence proof to the directional Hilbert transform (Theorem 3.1) and proactively addressing conjugate symmetry requirements on the DC and Nyquist components.
3. **Writing and Presentation Quality:** The paper is exceptionally polished, clearly structured, and easy to follow. The appendices are comprehensive, covering complexity, CNN extensions, and stability.
4. **Honesty and Transparency:** The authors are highly transparent about their limitations, openly acknowledging that their method is outperformed by PolyMerge and discussing the theoretical underpinnings of this gap.

### Major Weaknesses
1. **Lack of Empirical Competitiveness:** A critical requirement for any new optimization framework is to improve upon existing methods. However, the proposed PhaseMerge/U-PhaseMerge is consistently and significantly beaten by **PolyMerge** (e.g., $48.00\%$ vs. $42.33\%$ under 8-bit PTQ). The proposed method fails to deliver on empirical utility.
2. **Theoretical Mismatch (Permutation Invariance vs. Coordinate-Dependent Fourier Basis):** Dense weight matrices in self-attention projection and MLP blocks are permutation-invariant. Swapping the rows and columns does not alter the mathematical function of the network. Yet, the 2D FFT treats these dimensions as physical, topologically aligned spatial dimensions, which means the resulting frequencies, phase-rotations, and Hilbert transforms are entirely dependent on the arbitrary layout of the matrix. This weakens the physical justification of the "wave-superposition" framing on dense weights.
3. **Toy-Scale Experimental Evaluation:** 
   * The backbone model is a tiny, obsolete 5.7M parameter Vision Transformer (`vit_tiny`).
   * The expert models are trained on a mere 500 samples per task.
   * Most critically, the test sets are subsampled to only **100 samples per task**, making the empirical results highly susceptible to statistical noise.
4. **Unsupported and Overstated Claims:**
   * The authors claim "exceptional robustness under extreme 4-bit quantization." Yet, under 4-bit PTQ, U-PhaseMerge ($37.42\%$) is worse than both AdaMerging ($37.50\%$) and PolyMerge ($43.42\%$).
   * The title and text repeatedly claim "noise-cancelling" and "active phase-cancellation," but no empirical or analytical proof (such as a visualization of the learned phase angles or wave interference pattern) is provided to show that actual phase-cancellation occurs.
   * The "Overfitting-Optimizer Paradox" is claimed to be resolved, but U-PhaseMerge ($r=1$, 196 parameters) has roughly the same parameter count as AdaMerging (192 parameters), and its performance actually **degrades** from $42.33\%$ to $40.67\%$ as the calibration size increases from $M=16$ to $M=32$, indicating severe optimization instability.

---

## 4. Detailed Evaluation

### Soundness (Rating: Fair)
* **Permutation Invariance:** Because dense layers have no natural spatial indices, taking the Hilbert transform or 2D FFT of their weight matrices relies entirely on the arbitrary coordinate indexing. Reshuffling the rows or columns would yield a completely different Hilbert transform, demonstrating that the wave-theoretic alignment lacks physical foundation for dense networks.
* **Overfitting-Optimizer Paradox:** Since U-PhaseMerge ($r=1$) and AdaMerging have comparable parameter footprints (196 vs. 192) and both optimize unconstrained variables over small calibration streams, they exhibit similar overfitting characteristics. This is proven by U-PhaseMerge's performance degradation at $M=32$, whereas PolyMerge (only 12 parameters) remains highly stable.
* **PolyPhaseMerge Speculation:** Section A.3 proposes "PolyPhaseMerge" as a future direction to bridge the gap to PolyMerge, but since this hybrid model is purely speculative and was not actually implemented or evaluated, it cannot be considered a contribution of this paper.

### Presentation (Rating: Excellent)
The writing is clear, logical, and highly articulate. The authors did an outstanding job presenting the mathematical derivations and structuring the paper. However, there are some gaps in reproducibility details:
* The paper does not specify how non-square dense matrices of varying dimensions (e.g., `qkv` of shape `[576, 192]` and MLP `fc1` of shape `[768, 192]`) are reshaped or padded prior to the 2D FFT. Since the shape of the 2D FFT determines the frequency grid coordinates, this omission makes exact replication impossible.

### Significance (Rating: Poor)
The practical significance of this work is low. The proposed framework is more computationally complex than standard real-space model merging due to the 2D FFT and IFFT operations, yet it consistently delivers significantly worse multi-task accuracy than the simpler PolyMerge baseline. Practitioners are unlikely to adopt a more complex frequency-domain optimization that drops $6\%$ in absolute performance.

### Originality (Rating: Good)
The work has good conceptual originality. Although FREE-Merging (late 2024) already explored applying Fourier Transforms to weight vectors to prune harmful frequency bands, PhaseMerge's transition to a learnable, differentiable phase-rotation framework represents a distinct technical "delta."

---

## 5. Questions and Constructive Suggestions for Authors

1. **Empirical Validation of "Phase-Cancellation":** Can you provide direct empirical proof or visualization of the learned phase angles ($\phi_k^l$)? To support your "active phase-cancellation" and "noise-cancelling" claims, you should show that conflicting task updates are rotated out of phase ($\Delta \phi \approx \pi$) at high-sensitivity layers while complementary updates align in-phase ($\Delta \phi \approx 0$).
2. **Matrix Reshaping for 2D FFT:** How are the non-square dense weight matrices of different layers (e.g., `qkv`, `fc1`, `fc2`) reshaped or padded prior to executing the `RFFT2D`? Please explicitly document these layout, padding, and boundary conventions.
3. **Application to Convolutional Neural Networks:** As discussed in Section A.4, 2D convolutional kernels have physical height and width dimensions with genuine spatial correlation. Evaluating your framework on convolutional architectures (e.g., ResNet-50 or ConvNeXT) would resolve the permutation-invariance mismatch and could show substantial advantages for the continuous phase grid ($r=2$). Why did you choose to evaluate strictly on a Vision Transformer where dense layers lack this spatial coordinate foundation?
4. **Scale Up Evaluation:** To demonstrate practical relevance, can you evaluate PhaseMerge on standard, larger-scale benchmarks (such as a ViT-Base or ViT-Large backbone) and evaluate on the full test sets rather than subsampled 100-sample sets to reduce statistical noise?
5. **PolyPhaseMerge Implementation:** Since the proposed PolyPhaseMerge is highly promising and mathematically elegant, why was it left as a future direction? Implementing and evaluating PolyPhaseMerge could potentially bridge the performance gap and make the framework empirical-competitive with PolyMerge.
