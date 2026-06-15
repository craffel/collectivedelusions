# Paper Outline: Quantization-Aware Model Merging (Q-Merge)

## Section 0: Abstract
- **Context**: Fusing specialized task experts via model merging is highly efficient but deployed models must be quantized for edge-device constraints.
- **Problem**: Naive post-merging quantization degrades performance; merging pre-quantized experts fails due to alignment loss.
- **Proposed Solution**: Quantization-Aware Model Merging (Q-Merge). Optimize layer-wise merging coefficients directly under the post-training quantization operator.
- **Key Findings**: 
  - Under 8-bit quantization, Q-Merge (Adam with STE) achieves **72.97%** average accuracy, surpassing the unquantized FP16 baseline (**71.88%**).
  - First-order gradient optimization with Straight-Through Estimator (STE) is more stable and outperforms zero-order 1+1 ES (71.76%).
  - 4-bit quantization leads to a "Catastrophe" (~11-12% accuracy) revealing a hard pragmatic boundary for simple PTQ in model merging.

## Section 1: Introduction
- **Deep Learning Deployment Constraints**: SOTA models are too large for edge deployment; multitasking is key but memory limits require quantization.
- **Model Merging**: Task Arithmetic merges models without retraining but ignores quantization.
- **The Core Dilemma**: 
  - Merge-then-Quantize (M-then-Q): quantization noise disrupts weight spaces.
  - Quantize-then-Merge (Q-then-M): discrete quantization intervals prevent linear connectivity.
- **Our Contribution (Q-Merge)**: 
  - A pragmatic, calibration-free/low-data test-time adaptation framework that optimizes merging coefficients $\Lambda$ directly on the quantized network.
  - Evaluation of zero-order (1+1 ES) vs. first-order (Adam GD with STE) optimization.
  - Highlighting the pragmatic boundaries: 8-bit success and 4-bit PTQ failure.

## Section 2: Related Work
- **Model Merging**: Task Arithmetic (Ilharco et al.), AdaMerging (Yang et al.), Fisher-based techniques. Contrast our focus on quantization-awareness.
- **Model Quantization**: Post-Training Quantization (PTQ) vs. Quantization-Aware Training (QAT). Highlight why QAT is computationally heavy and why PTQ is preferred by practitioners.
- **Test-Time Adaptation**: Entropy minimization (TENT), adaptive coefficients. Contrast with our optimization of merging weights under discrete rounding.

## Section 3: Methodology
- **Task Arithmetic Formulation**: Task vectors $\tau_k = \theta_k - \theta_{\text{base}}$. Merged full-precision weights $\theta_{\text{merged}}(\Lambda)$.
- **Uniform Symmetric Quantization**: Formulas for scale factor $S^l$, round-and-clip, and mapping to $b$-bit integer representations.
- **The Optimization Objective**: Joint Shannon Entropy minimization on an unlabeled calibration stream ($64$ total images).
- **Optimizer 1: Zero-Order (1+1 ES)**: Formulation of random-walk candidate mutations and acceptance criteria.
- **Optimizer 2: First-Order (Adam GD with STE)**:Straight-Through Estimator $\frac{\partial \text{round}(x)}{\partial x} \approx 1$ to propagate gradients back to full-precision $\Lambda$.
- **Pragmatic Perspective**: Why this is low-resource, extremely fast, and highly practical.

## Section 4: Experiments & Results
- **Experimental Setup**: Backbone (ViT-Tiny from `timm`), 4 datasets (MNIST, FashionMNIST, CIFAR-10, SVHN), 3 random seeds (42, 100, 2026) for statistical rigor.
- **Layer-wise Partitioning**: $L=14$ parameter groups.
- **Baselines**: FP16 Merged, Q-then-M, M-then-Q, AdaMerging (FP16 optimized, then quantized).
- **8-Bit Performance Analysis**:
  - Presentation of Table 1.
  - Detailed analysis of Q-Merge (Adam GD) surpassing FP16. Explain this as a regularizing effect of optimizing under quantization.
  - Comparison of STE first-order vs. 1+1 ES. First-order is more stable (2.5x lower variance).
- **4-Bit Catastrophe Analysis**:
  - Presentation of Table 2.
  - Detailed discussion on why RTN 4-bit fails completely. The mathematical reason: extreme truncation and rounding errors collapse representation manifolds in low-capacity models like ViT-Tiny.
  - Concrete recommendations for practitioners (e.g., AWQ, GPTQ, or QAT).
- **Visual Results**: Reference to the grouped bar chart `qmerge_vs_baselines.png`.

## Section 5: Conclusion & Future Work
- **Pragmatic Summary**: Q-Merge bridges the gap between model merging and post-training compression.
- **Actionable Takeaways**: Use first-order STE for 8-bit; avoid simple PTQ for 4-bit merging.
- **Limitations**: RTN quantization limit, small backbone focus.
- **Future Directions**: Fusing with weight-activation scaling, larger LLMs/diffusion models.
