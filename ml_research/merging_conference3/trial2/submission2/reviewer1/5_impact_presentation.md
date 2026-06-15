# 5. Impact and Presentation Evaluation

## Major Strengths

1. **Theoretical Insight on Scale Invariance:** The formal proofs showing that standard downstream feature normalization layers (L2, LayerNorm, RMSNorm) mathematically neutralize global weight scaling factors are correct, elegant, and highly valuable. It provides a strong theoretical explanation for why complex weight-scaling algorithms are often redundant in modern architectures.
2. **Mathematical Elegance & Simplicity:** SVS is non-parametric, training-free, and has a closed-form solution. It avoids the massive parameter and computational overhead of test-time optimization and coordinate warping methods.
3. **Information-Theoretic Rank Allocation (Entropy-SVS):** Using normalized Shannon spectral entropy of singular values to adaptively scale slicing ranks across different layers is an elegant and principled concept that respects the hierarchical representation in deep networks.
4. **Writing and Narrative Quality:** The paper is exceptionally well-written, with clear definitions, logical structure, and an engaging narrative style. It includes a remarkably honest and transparent limitations section (Section 4.6) discussing head routing, SwiGLU gates, and causal attention.

## Major Areas for Improvement

1. **Empirical Rigor and Statistical Validation:**
   *   Evaluate on **full test sets** instead of a truncated subset of 1,000 samples per task, as the current $0.05\%$ accuracy difference is equivalent to only 2 samples out of 4,000 and is statistically meaningless.
   *   Provide **multiple random seeds** and report **standard deviations/confidence intervals** for all experiments in Table 1, Table 2, and the figures.
2. **Direct Baseline Comparisons:**
   *   Compare SVS directly against existing SVD-based merging methods such as **Task Singular Vectors (TSV-Compress)** (Gargiulo et al., 2025) and **SVD-Merging** (Stoica et al., 2025).
3. **Ablation of Dynamic vs. Uniform Rank:**
   *   Compare **Entropy-SVS** directly against **uniform SVS** at the same average rank (e.g., comparing Entropy-SVS with average rank of 43.9 to uniform SVS with $k=44$) to prove that the information-theoretic allocation actually provides a performance benefit.
4. **Rigorous Training of MLP Experts:**
   *   Re-train the un-normalized MLP experts so they achieve standard, converged accuracies ($>95\%$ on MNIST, $>85\%$ on FashionMNIST), as the current experts ($77\%$ and $69\%$) are extremely poorly trained, undermining the validation of BWN.
5. **Hyperparameter and Training Transparency:**
   *   Include all missing hyperparameters and training details (optimizers, learning rates, epochs, weight decay, batch sizes, etc.) for both the CLIP experts and MLP models in a dedicated section or appendix.

## Overall Presentation Quality
The overall presentation is **Excellent**. The mathematical notation is clean and precise. Figures 1 through 5 are highly legible, professional, and well-contextualized. The paper's narrative flows smoothly, and the discussion of why coordinate-basis pruning methods (TIES, DARE) outperform SVS (Section 4.2) shows a high degree of intellectual honesty.

## Potential Impact and Significance
The paper has **Moderate Potential Impact**:
*   **Theoretical Impact (High):** The global scaling cancellation proofs and the concept of using Shannon spectral entropy for rank allocation are highly relevant and will likely influence future model-merging and weight-compression research.
*   **Empirical Impact (Low-to-Moderate):** Because SVS is outperformed by simpler, coordinate-basis pruning methods like TIES-Merging ($77.98\%$ vs. $74.83\%$), practitioners are unlikely to adopt SVS in its purest form. However, if extended to the hybrid spectral-spatial merging paradigm proposed in Section 4.6 (SVS + coordinate masking), its practical significance could increase substantially.
