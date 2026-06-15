# Evaluation Component 5: Impact and Presentation

## Overall Presentation Quality
The presentation quality of the paper is **outstanding (Excellent)**. 
- The text is beautifully structured, highly articulate, and exceptionally easy to follow. 
- The mathematical notations are precise, and the equations flow logically from one to another.
- The figures (including the teaser in Figure 1 and the sensitivity sweep in Figure 2) are high-quality, professional, and do an excellent job of visually illustrating the main concepts.
- The authors are highly transparent and intellectually honest. They do not hide their limitations; they explicitly discuss the "domain disconnect gap" (the performance drop from individual experts to merged models) and the fact that absolute 4-bit (INT4) performance is practically unusable. This level of intellectual honesty is exemplary and makes the paper a pleasure to read.

## Major Strengths
1. **Pioneering Theoretical Analysis:** Connecting local landscape flatness to post-training quantization (PTQ) robustness at *test-time adaptation (TTA)* represents a significant theoretical contribution. The quadratic noise decomposition in Equation 11 is elegant and insightful.
2. **Identification of Norm Scale Pathology:** Uncovering the physical 50-fold scale discrepancy in task-vector norms and demonstrating why unnormalized regularizers suffer from scale-blindness is a major high-signal finding. It resolves a long-standing question of why sharpness optimization has historically failed in test-time model merging.
3. **High-Efficiency Design (CR-SACM):** Developing a scale-balanced, first-order sharpness approximation (CR-SACM) that requires only two forward-backward passes and runs in $1.56$ seconds ($52.8\times$ faster than exact Hessian trace calculation) is highly practical and suited for real-world edge devices.
4. **Comprehensive Ablation and Sensitivity Sweeps:** The ablations on the regularization strength $\gamma$ (Table 2), the clipping threshold $\beta$ (Table 3), the calibration stream size $N$ (Table 4 in Appendix), and class imbalance (Table 7 in Appendix) are exceptionally thorough and provide robust empirical validation of the theoretical claims.
5. **Outstanding Transparency:** Admitting that absolute INT4 performance is non-viable for production and explaining the expert-to-merge drop provides excellent context and scientific integrity.

## Areas for Improvement (Empiricist Focus)
1. **Regularization Bias under FP32/INT8:** The authors need to address why CR-PolySACM consistently degrades performance under standard high-precision formats (FP32 and all INT8 variants) compared to standard PolyMerge. They briefly mention this as a "regularization trade-off," but a deeper empirical analysis of this trade-off is needed.
2. **Backbone and Dataset Scaling:** The empirical results are restricted to a toy-sized 5.7M parameter ViT-Tiny model and simple image datasets (MNIST, FashionMNIST, CIFAR-10, SVHN). To demonstrate the true edge-deployment viability, the method must be evaluated on larger, more realistic backbones (e.g., ViT-Base or small LLMs) and more challenging benchmarks (e.g., DomainNet or VTAB).
3. **Inclusion of Stronger Baselines:** Standard, highly popular static merging methods like TIES-Merging and DARE should be included in Table 1. Furthermore, comparing against a tuned global scaling factor for Task Arithmetic (rather than a fixed $\lambda=0.25$) would provide a much fairer static baseline.
4. **Direct Statistical Reporting in Main Table:** Standard deviations across the independent splits should be reported directly in Table 1, and the number of splits should ideally be increased to at least 5 to ensure a more robust statistical paired t-test.

## Potential Impact and Significance
The potential impact of this paper is **Good**. 
- **Scientific Impact:** High. The theoretical insights into task-vector norm scale pathologies and the quadratic noise decomposition represent highly valuable contributions to the model merging and TTA literature. Researchers working on weight-space composition will likely build upon these findings to design better scale-aware optimization methods.
- **Practical Impact:** Modest. The practical utility is currently limited because the proposed method (CR-PolySACM) only outperforms the baseline in a regime (INT4) where the absolute accuracy is too low to be useful (19.07%), while degrading performance in standard, practically useful regimes (FP32/INT8). Scaling the method to larger models and milder domain shifts (where absolute performance is viable) is required to unlock its practical impact.
