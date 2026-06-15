# Peer Review

## 1. Summary of the Paper
This paper proposes **PhaseMerge**, a post-hoc model merging framework that operates in the complex Fourier frequency domain. To blend task-specific specialized experts fine-tuned from a shared pre-trained backbone, PhaseMerge projects parameter updates (task vectors) into Fourier space using the 2D Real Fast Fourier Transform (RFFT2D). The framework decomposes updates into amplitude and phase components, allowing the optimizer to learn continuous, differentiable phase-rotations ($\phi_k^l$) to adjust constructive and destructive wave interference. 

The authors explore two phase-shift parameterizations: a uniform layer-wise phase rotation (U-PhaseMerge, $r=1$) and a continuous upsampled 2D phase-shift grid (PhaseMerge, $r=2$). To ensure mathematically consistent, real-valued spatial weight reconstructions, they introduce a symmetry-preserving frequency mask that zeroes out phase shifts on the DC and Nyquist components. The parameters are optimized under post-training quantization (PTQ) constraints by minimizing prediction entropy on a small calibration dataset using Straight-Through Estimators (STE). The framework is evaluated using Vision Transformers (`vit_tiny_patch16_224`) across four vision datasets (MNIST, FashionMNIST, CIFAR-10, SVHN).

---

## 2. Strengths
- **Conceptual Originality:** The wave-theoretic framing of model merging—representing task updates as complex wavefunctions and optimizing constructive/destructive interference—is highly creative and intellectually stimulating. It shifts the merging paradigm from static real-space interpolation to frequency-space phase synchronization.
- **Strong Mathematical Foundation:** Section 3.4 contains a rigorous proof (Theorem 3.1) establishing that a uniform phase shift in the Real 2D Fourier domain is equivalent to a spatial coordinate rotation in the 2D subspace spanned by the task vector and its directional Hilbert transform.
- **Meticulous Technical Execution:** The symmetry-preserving frequency mask ($M_{\text{sym}}$) is a clever design choice that mathematically guarantees real-valued spatial weight reconstructions, preventing complex autograd leakage and stabilizing optimization.
- **Excellent Clarity and Presentation:** The paper is beautifully written, easy to follow, and the mathematical formulas are laid out with precision. The authors also include an admirable discussion of limitations and future directions (e.g., proposing "PolyPhaseMerge").

---

## 3. Weaknesses

While the theoretical and conceptual framing of this work is outstanding, there are several major empirical, methodological, and practical weaknesses that limit its significance and real-world utility:

### A. Representation Mismatch on Dense Weights
The 2D Fast Fourier Transform operates on ordered grids where neighboring dimensions share structural spatial/geometric correlations (such as pixels in images, audio signals, or convolutional filters). However, the dense linear layers of transformers (`qkv`, `proj`, `fc1`, `fc2`) are permutation-invariant and lack an inherent 2D topology. Swapping two rows or columns does not alter the mathematical function of a dense layer, but it completely changes its 2D Fourier spectrum. While locking the coordinate system to a shared pre-trained initialization ($W_{\text{pre}}$) provides an optimization workaround, the 2D FFT remains structurally mismatched for dense weights. This mismatch is empirically confirmed by the fact that the upsampled 2D spatial frequency smoothing grid ($r=2$) consistently performs *worse* than the simpler uniform phase shift ($r=1$).

### B. Empirical Claims Contradicted by Reported Data
The paper repeatedly claims that the compact U-PhaseMerge ($r=1$) variant exhibits superior robustness to extreme 4-bit quantization and outperforms traditional real-space layer-wise optimizations (e.g., AdaMerging). However, a close inspection of the tables reveals the opposite:
- **Table 1 (4-bit PTQ):** AdaMerging achieves $37.50 \pm 1.22\%$, while U-PhaseMerge achieves $37.42 \pm 1.94\%$.
- **Table 3 (Target Schema Shift under 4-bit PTQ):** AdaMerging achieves $37.50 \pm 1.22\%$, while U-PhaseMerge achieves $37.42 \pm 1.94\%$.

In both cases, U-PhaseMerge underperforms AdaMerging (the traditional real-space layer-wise optimization baseline) by $0.08\%$ absolute accuracy, while exhibiting substantially higher variance. The claims of outperforming traditional real-space optimizations under low-bit quantization are thus empirically unsupported.

### C. Severe Optimization Instability and Scaling Failure
In Table 2 (Sample Complexity Sweep), U-PhaseMerge exhibits severe instability as the calibration dataset size $M$ increases from $16$ to $32$:
- U-PhaseMerge's accuracy drops significantly from $42.33 \pm 1.76\%$ (at $M=16$) to $40.67 \pm 3.65\%$ (at $M=32$), and its standard deviation more than doubles.
- In contrast, AdaMerging stably improves from $41.67 \pm 1.45\%$ to $42.50 \pm 1.59\%$, and PhaseMerge ($r=2$) improves from $40.83 \pm 1.18\%$ to $42.00 \pm 1.34\%$.

In standard machine learning optimization, increasing calibration data should improve generalizability and reduce variance. The collapse in U-PhaseMerge's average performance and spike in variance suggest that its optimization trajectory in complex phase-space is highly unstable and prone to severe overfitting or local minima that the soft $L_2$ phase decay penalty fails to address. This scaling failure is a major concern for practical deployment.

### D. Practical Utility vs. Simpler, Superior Baselines
For practitioners, the value of a model merging method is determined by its trade-off between deployment simplicity and performance. PhaseMerge is highly complex to implement (requiring complex 2D FFT, 2D IFFT, straight-through estimators for PTQ gradients, symmetry masks, and phase rotators). Yet, it is consistently and substantially outperformed (by $5.17\%$ to $6.00\%$ absolute accuracy) by **PolyMerge**, which is a simpler, real-space baseline that parameterizes layer coefficients using a low-dimensional polynomial (only 12 parameters) and requires no frequency-domain transformations. Given this massive performance gap and the high implementation complexity of PhaseMerge, there is currently no practical reason to adopt PhaseMerge over simpler, superior real-valued baselines.

### E. Toy Experimental Scale
The experimental evaluation is conducted on a tiny model (`vit_tiny_patch16_224`, ~5.7M parameters) and simple vision classification datasets. Furthermore, the test sets are subsampled to only 100 samples per task, and the baseline expert performances are extremely low (e.g., $81\%$ on MNIST and $74\%$ on FashionMNIST), indicating under-trained experts. It is highly unproven whether PhaseMerge's complex Fourier transformations and Straight-Through Estimators can scale efficiently or provide any real-world utility on massive foundation architectures like multi-billion parameter LLMs or ViT-Huge, which are the main targets for post-training quantization and model merging in industry.

---

## 4. Questions and Constructive Feedback for the Authors
1. **Empirical Correction:** Please revise the text in the abstract, introduction, and experiments section to accurately reflect the 4-bit PTQ results. The claims that U-PhaseMerge outperforms traditional real-space layer-wise optimizations under 4-bit PTQ must be toned down, as the tables show AdaMerging is slightly superior and more stable.
2. **Analysis of $M=32$ Failure:** Could the authors provide a deeper analysis of why U-PhaseMerge collapses and becomes highly unstable when calibration data increases from $M=16$ to $M=32$? Is this a gradient-vanishing issue through the complex-valued autograd, or is the $L_2$ phase decay penalty insufficiently tuned?
3. **Evaluation Scale:** Can the authors provide results on a larger backbone model (e.g., `vit_base_patch16_224`) and evaluate on the full test sets rather than subsamples of 100 images, to prove that the proposed method scales and stabilizes under more realistic benchmarks?
4. **Complexity vs. Performance:** Given that PolyMerge significantly outperforms PhaseMerge across the board, how do the authors justify the added computational and deployment complexity of projecting weights to the complex Fourier domain?

---

## 5. Ratings

- **Soundness:** **Fair** (The mathematical derivations and conjugate symmetry constraints are sound, but the severe optimization instability at $M=32$ and the empirical claims being contradicted by their own tables represent significant weaknesses.)
- **Presentation:** **Excellent** (The writing style, mathematical clarity, logical structure, and overall presentation quality are of an outstanding standard.)
- **Significance:** **Fair** (While conceptually highly original, the practical significance is limited because the method is complex to implement but gets worse results than a simpler real-valued polynomial baseline, and is evaluated on a very small, toy scale.)
- **Originality:** **Excellent** (The introduction of complex wave superposition and phase synchronization to model merging, coupled with the Hilbert transform dual proof, is highly creative and novel.)

## 6. Overall Recommendation
**3: Weak Reject**

*Rationale:* The paper introduces a highly creative and mathematically elegant frequency-domain merging paradigm with an excellent theoretical foundation and writing quality. However, from a practical standpoint, the proposed method has major limitations: it is structurally mismatched for dense weights, exhibits severe optimization instability when scaling calibration data, is evaluated on a toy scale, and is consistently and substantially outperformed by a simpler real-space polynomial baseline. Crucially, the paper's key claims regarding U-PhaseMerge outperforming real-space layer-wise optimizations under 4-bit quantization are empirically contradicted by its own tables. I encourage the authors to correct their empirical claims, address the optimization collapse, and scale their evaluations, which would make this highly creative framework a strong candidate for publication.
