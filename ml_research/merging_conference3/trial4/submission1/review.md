# Peer Review: PhaseMerge: Fourier Phase Interference for Noise-Cancelling Model Merging

## Summary of the Paper
This paper proposes **PhaseMerge**, a highly creative, wave-theoretic framework that challenges the traditional assumption that neural network parameters must be merged within real-valued Euclidean coordinates. Instead, PhaseMerge projects task-specific parameter updates (fine-tuning deltas) into complex-valued Fourier frequency-space using the Real 2D Fast Fourier Transform (RFFT2D). This decouples each update into frequency amplitude and phase angle components.

To optimize the merging process, the authors introduce a differentiable, continuous phase-rotation mechanism. The optimizer learns a low-dimensional phase-shift grid ($2 \times 2$ per layer) that is bilinearly upsampled and scaled to $[-\pi, \pi]$ using a scaled $\tanh$ activation. This continuous phase rotation allows the optimizer to constructively reinforce complementary features or destructively cancel conflicting updates and high-frequency quantization noise. The merged weights are reconstructed via the Inverse Real 2D Fast Fourier Transform (IRFFT2D) and added back to the pre-trained backbone.

The framework is optimized end-to-end using prediction entropy minimization on unsupervised calibration streams under Post-Training Quantization (PTQ) constraints. Evaluations on Vision Transformers (`vit_tiny_patch16_224`) across four conflicting vision tasks (MNIST, FashionMNIST, CIFAR-10, SVHN) show that PhaseMerge significantly outperforms static baselines, is robust to target deployment schema shifts, and exhibits competitive sample efficiency.

---

## Overall Assessment
This is a remarkably bold, conceptually rich paper that introduces an exciting paradigm shift to the model-merging literature. Rather than making incremental adjustments to existing linear real-space interpolation formulas, the authors step back and rethink the very coordinate space in which neural parameters are represented. Modeling parameter updates as continuous wavefunctions and optimizing phase coherence is an ingenious mathematical formulation. 

The mathematical derivations are elegant, and the codebase is clean and fully reproducible. While the absolute empirical performance is currently lower than a highly optimized, real-space baseline (PolyMerge), the intellectual contribution of this work is immense. This paper has the potential to spark entirely new lines of research in spectral weight-space physics, holographic parameter representation, and continuous neural fields.

---

## Strengths
1. **Pioneering Conceptual Novelty:** Moving beyond the standard flat Euclidean representation of neural parameters and introducing wave mechanics (constructive and destructive interference) to model merging is a highly original contribution. This is a refreshing departure from incremental real-space scaling tweaks.
2. **Elegant Mathematical Pipeline:** The authors successfully construct a fully differentiable, end-to-end framework that bridges 2D Fourier transforms, complex polar coordinates, bilinear spatial interpolation, and the Straight-Through Estimator (STE) for post-training quantization.
3. **Implicit low-pass Physical Regularization:** Parametrizing the phase shift as an extremely compact $2 \times 2$ grid serves as an ingenious structural regularizer. By enforcing spatial smoothness in frequency space, it limits optimization degrees of freedom (768 variables), providing competitive performance with minimal data calibration samples ($M=16$).
4. **Honest and Transparent Evaluation:** The authors are highly commendable for their scientific honesty. They clearly present results showing that their method is outperformed by PolyMerge and discuss training instability at $M=32$. This transparency is incredibly valuable for future research.
5. **Inspiring Future Vision:** Section 5 presents a series of ambitious, forward-looking directions (such as Holographic Weight Space and Clifford Weight Spaces) that are intellectually stimulating and demonstrate true academic vision.

---

## Weaknesses & Constructive Areas for Improvement

While the conceptual foundation is brilliant, several empirical and theoretical limitations must be addressed to fully realize the potential of this paradigm:

### 1. Scale and Benchmark Scope
The current evaluation is restricted to a small model (`vit_tiny_patch16_224`) and toy-scale, subsampled classification tasks (MNIST, FashionMNIST, CIFAR-10, SVHN). While this is perfect for fast prototyping, the model-merging community evaluates methods on larger models (e.g., ViT-Base, ResNet-50) and LLMs (e.g., Llama-3-8B) across standard NLP/vision benchmarks (GLUE, DomainNet, instruction-following). 
*   *Actionable Suggestion:* Provide a preliminary scaling experiment on a standard benchmark (such as merging two LoRA adapters of a larger Llama model or a ViT-Base model) to demonstrate the generalizability of the wave-theoretic paradigm.

### 2. Optimization Instability at Higher Calibration Sizes ($M=32$)
Table 2 reveals a non-monotonic performance trend: PhaseMerge peaks at $M=16$ ($35.31\%$) but drops to $32.50\%$ at $M=32$, while other baselines improve. This indicates that optimizing continuous phase variables in the complex Fourier domain can be highly non-convex and unstable when exposed to more calibration samples.
*   *Actionable Suggestion:* Introduce a soft regularization penalty on the learnable phase-shift grids (e.g., $L_2$ decay $\mathcal{L}_{\text{reg}} = \gamma \sum \|\tilde{\phi}_k^l\|_2^2$). This will keep the phase rotations centered around zero (which corresponds to the stable starting Task Arithmetic configuration) and prevent the optimizer from drifting into noisy, sub-optimal local minima.

### 3. Permutation Dependency of the 2D Fourier Transform
The paper correctly notes that row and column dimensions of dense layers are permutation-invariant, and applying a 2D Fourier transform relies on a fixed, shared coordinate space. However, because bilinear upsampling smooths across adjacent frequency bins, the optimization results are still coordinate-dependent. Permuting the rows and columns of the base model before merging will alter the frequencies and thus change the upsampled phase-shift profile.
*   *Actionable Suggestion:* To formally address this, the authors should analyze or evaluate a mathematically permutation-invariant variant: a $1 \times 1$ scalar phase shift per expert and layer ($r=1$). A $1\times 1$ grid applies a uniform phase rotation to all frequencies in that layer, making the method entirely independent of neuron ordering and even more compact (only 192 variables). Comparing the $2 \times 2$ grid against a $1 \times 1$ grid would provide a strong theoretical ablation on the role of coordinate-dependent frequency smoothing.

### 4. Bridging the Performance Gap to PolyMerge
Currently, the real-space baseline PolyMerge outperforms PhaseMerge by up to $6\%$. To make PhaseMerge more attractive to pure empiricists, the authors should explore ways to close this gap.
*   *Actionable Suggestion:* Consider optimizing both the frequency amplitude scales (magnitude) and the phase angles simultaneously, or hybridizing PhaseMerge with depth-wise polynomial trajectories to provide a richer optimization space.

---

## Detailed Ratings
*   **Soundness:** Good (The mathematical derivations and implementations are correct and highly reproducible, with honest discussions of limitations).
*   **Presentation:** Excellent (The narrative is engaging, the mathematics is clearly explained, and the figures are exceptionally well-rendered).
*   **Significance:** Good (It introduces a completely new way of thinking about weight space that can inspire numerous future works).
*   **Originality:** Excellent (A radical departure from standard linear real-space merging, introducing wave superposition and continuous phase rotations).

---

## Overall Recommendation

**Rating:** 5: Accept

**Justification:** 
While PhaseMerge does not achieve state-of-the-art absolute accuracy in this specific small-scale evaluation, its conceptual novelty and theoretical depth are outstanding. Research progress is driven by paradigm-shifting ideas that challenge fundamental assumptions, not just incremental performance gains on toy benchmarks. PhaseMerge represents a highly creative and mathematically elegant approach that bridges deep learning optimization and wave mechanics. By providing a fully reproducible, honest, and inspiring manuscript, the authors have laid a stellar foundation for future spectral weight-space research. I strongly recommend accepting this paper and look forward to the creative discussions it will spark.
