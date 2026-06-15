# 5. Impact and Presentation Quality

## Major Strengths
1. **High Conceptual Creativity:** The idea of modeling model-merging task updates as wavefunctions in the complex Fourier space and optimizing their phase rotation (wave alignment/interference) is exceptionally original and intellectually stimulating.
2. **Strong Mathematical Rigor:** Section 3.4 contains a highly elegant proof (Theorem 3.1) showing that uniform frequency phase-rotation in U-PhaseMerge is equivalent to rotating the real spatial weight in a 2D subspace spanned by the task vector and its directional Hilbert transform. This connects frequency-space concepts to solid spatial coordinate operators.
3. **Meticulous Technical Design:** The implementation of the symmetry-preserving frequency mask ($M_{\text{sym}}$) is technically precise, preventing complex residuals and ensuring mathematically consistent real-valued spatial matrix reconstructions.
4. **Honest Discussion of Limitations:** The paper has a commendable level of self-reflection in Section 4.5. The authors discuss the sandbox evaluation scale, the representational mismatch of 2D FFT on dense weights, and the non-convex optimization instability. They also propose "PolyPhaseMerge" as a very thoughtful future hybridization path.

## Areas for Improvement
1. **Correction of Empirical Overstatements:**
   The paper's narrative repeatedly claims that U-PhaseMerge outperforms traditional real-space layer-wise optimizations (specifically AdaMerging) under 4-bit PTQ. However, as noted in Table 1 and Table 3, U-PhaseMerge ($37.42 \pm 1.94\%$) actually slightly underperforms AdaMerging ($37.50 \pm 1.22\%$) and has higher variance. The authors must tone down these claims to accurately align with their tables.
2. **Investigation of optimization instability at $M=32$:**
   The paper notes that U-PhaseMerge's performance drops to $40.67 \pm 3.65\%$ and standard deviation doubles at $M=32$. This optimization instability as calibration data grows is counter-intuitive and a major practical concern. The authors should investigate this failure mode more deeply—is it due to phase wrap-around, gradient exploding/vanishing in complex autograd, or failure of the $L_2$ phase decay penalty?
3. **Scaling the Experimental Scope:**
   Evaluating on a tiny Vision Transformer (`vit_tiny_patch16_224`, ~5.7M parameters) and toy datasets (MNIST, FashionMNIST, etc.) makes it hard to gauge real-world utility. For practitioners in the industry, post-training quantization and model merging are most crucial for multi-billion parameter Large Language Models (LLMs) or large-scale foundation vision models. The authors need to run PhaseMerge on at least a ViT-Base or a small LLM (e.g., LLaMA-1B/3B) to show it is scalable and computationally efficient under realistic constraints.

## Overall Presentation Quality
The presentation quality is **Excellent**. The paper is superbly structured, highly readable, and the academic tone is professional and engaging. The figures (e.g., entropy convergence and overfitting paradox) and tables are clearly presented and integrated well with the main narrative.

## Potential Impact and Significance
- **Theoretical Impact: High.** The wave-theoretic framing opens up an exciting new direction for spectral deep learning and parameters-space analysis. It could inspire researchers to study weight properties in frequency domains and design more sophisticated spectral operators.
- **Practical Impact: Low to Moderate.** From a deployment standpoint, the immediate impact is limited. The method is complex to implement (requiring 2D FFT, 2D IFFT, straight-through estimators for PTQ gradients, symmetry masks, and phase rotators) but is significantly outperformed (by 5-6% absolute accuracy) by **PolyMerge**, which is a simpler, real-space baseline with far fewer parameters. A practitioner looking to merge models for edge-device deployment would choose PolyMerge due to its simplicity, stability, and superior accuracy.
