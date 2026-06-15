# 5. Presentation, Impact, and Significance Evaluation

This document outlines the major strengths of the paper, constructive areas for improvement, a detailed evaluation of its presentation quality, and its overall potential impact/significance in the field of machine learning.

## Major Strengths

1. **Innovative Paradigm Shift**: Rethinking the physical spatial representation of layer-wise model merging coefficients as a discrete 1D signal and optimizing its frequency-domain representation via the DCT-II is highly creative, refreshing, and theoretically elegant.
2. **Mathematical and DSP Rigor**: The paper provides a highly rigorous mathematical justification for its choices. The virtual boundary derivative flatness analysis ($\frac{d\alpha}{dl}=0$) and the perfect conditioning ($\kappa = 1.0$) of the orthonormal DCT basis compared to standard ill-conditioned polynomial bases (PolyMerge) are exceptional.
3. **Outstanding Simulation Rigor**: Evaluating all simulation results over **30 independent random seeds** with standard deviations, and multi-axially stress-testing the algorithms under multiple adversarial scenarios (extreme label shift, bursty task streams, validation selection bias sweeps, small batch noise) sets a very high empirical standard.
4. **Diagnostic Honesty and Insight**: The identification and explanation of the **PEFT-Induced Step-Function Discontinuity** is a brilliant highlight. Instead of omitting the failure of SpectralMerge-LP on ResNet-18, the authors used DSP principles to explain *why* hard cutoffs fail on localized adapters (infinite frequency support) and *why* soft spectral decay (SpectralMerge-Reg) is mathematically required to succeed.
5. **Practical Extensions**: Proposing and evaluating **Block-wise Spectral Merging** to handle architectural heterogeneity, and introducing **Adaptive Bandwidth (LP-Adaptive)** to dynamically scale capacity represent highly valuable practical insights for deep learning practitioners.

## Constructive Areas for Improvement

### 1. Address the Empirical Gap on Physical Networks (Statistical Rigor)
- **The Issue**: While simulated experiments are backed by 30 seeds and standard deviations, the physical MLP and pre-trained ResNet-18 checkpoint experiments are reported as single-run point estimates.
- **Actionable Suggestion**: The authors should run the physical experiments (the Heterogeneous MLP and the ResNet-18 CIFAR-10 experiments) across multiple random seeds (e.g., 5 or 10 seeds) and report means and standard deviations in Table 3 and Figure 7. This is especially critical because validation few-shot tuning is highly sensitive to the specific samples chosen at $M=10$ or $M=15$.

### 2. Scaling Up to Modern Foundation Models and Standard Benchmarks
- **The Issue**: The physical experiments are conducted on relatively small models (12-layer MLP, 18-layer ResNet) and small datasets (synthetic classification, CIFAR-10 binary splits).
- **Actionable Suggestion**: To showcase the true scaling power of SpectralMerge, the authors should evaluate it on larger modern models, such as standard vision transformers (ViT-B/16 or ViT-L/16 on VTAB) or Large Language Models (e.g., Llama-3-8B on GLUE or instruction-following benchmarks) using standard parameter-efficient adapters (LoRA). Since LoRA updates are distributed across all layers, it would also serve as a perfect validation for the SpectralMerge-LP and LP-Adaptive variants.

### 3. Hyperparameter Sensitivity Ablations
- **The Issue**: The paper introduces critical hyperparameters like the cutoff frequency $F$ and the soft decay weight $\mu$. However, there are no sensitivity sweeps showing how these choices affect performance.
- **Actionable Suggestion**: The authors should include an ablation section or appendix sweeping $\mu \in [0.1, 10.0]$ and $F \in \{1, 2, 3, 4, 5, 6\}$ to demonstrate the robustness of their parameter selection and provide a guide on how to tune them in practice.

### 4. Direct Sparsification Baseline Comparison
- **The Issue**: The related work section claims that SpectralMerge can operate in synergy with weight-level sparsification heuristics like TIES-Merging and DARE.
- **Actionable Suggestion**: It would be highly valuable to include a hybrid baseline, such as "TIES-Merging + SpectralMerge-Reg" or "DARE + SpectralMerge-Reg", to empirically prove this synergy and demonstrate whether the combined method yields state-of-the-art results compared to standalone TIES or DARE.

---

## Presentation Quality
- **Rating: Excellent**
- **Structure and Narrative**: The narrative is incredibly cohesive, engaging, and easy to follow. The paper clearly identifies the spatial coordinate representation bottleneck, establishes a visionary frequency-domain alternative, provides complete mathematical formulations, and backs them up with robust simulation and physical validation.
- **Clarity of Writing**: The prose is highly professional, precise, and grammatically polished. Technical terms are clearly defined, and the digital signal processing concepts are translated into deep learning equivalents with extreme clarity.
- **Visualizations and Captions**: The figures are well-chosen, and the captions are highly descriptive, making the paper's findings self-contained and easily understandable for an expert reader.

---

## Potential Impact and Significance
- **Rating: High**
- **Significance of Contribution**: Model merging is a critical, resource-efficient area of machine learning. The unconstrained spatial search bottleneck (Overfitting-Optimizer Paradox) is a major practical challenge, especially in few-shot regimes. SpectralMerge offers a mathematically elegant, computationally negligible ($<0.0001\%$ forward pass overhead), and highly stable solution.
- **Broader Influence**: The core idea of treating layer-wise variables as a 1D signal and optimizing them in the spectral domain is a highly generic concept. This could easily influence other areas of deep learning, such as regularizing hyperparameter tuning across depth, analyzing layer sensitivity patterns in foundation models, or modeling training trajectories. The proposed Wavelet-based expansion and 2D DCT-II ideas in Future Directions also open up exciting new avenues of research.
