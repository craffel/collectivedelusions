# 4. Experimental Setup and Baseline Evaluation

## Evaluation of Experimental Setup
- **Model Choice:** The authors implement a compact 12-layer Vision Transformer (\texttt{ViTTiny}) with custom coefficient-aware weight interpolation (\texttt{MergedViTTiny}). While this allows for fully differentiable coefficient optimization on the fly, the model size is extremely small ($d_{\text{model}}=64$, 2 heads, 12 layers).
- **Datasets:** The 4-task visual benchmark (MNIST, FashionMNIST, CIFAR-10, SVHN) is a standard, lightweight multi-task setup in academic model merging literature.
- **Statistical Rigor:** Testing is done on 1000 samples per task (4000 total) across 3 random seeds, which provides robust error bars. The inclusion of standard deviations in Table 1 is excellent and ensures that the results are statistically grounded.

## Evaluation of Baselines
- **Upper Bound:** Individual task-specific experts are evaluated (41.48% mean), providing a clear reference upper bound.
- **Static Baseline:** Uniform Task Arithmetic (30.41% mean) is included, which is the most critical baseline here, as it represents a zero-cost, zero-overhead merging strategy.
- **AdaMerging Baseline:** Layer-wise merging (Level 2) serves as the baseline for modern adaptive merging.

## Do the Results Support the Claims?
Yes, the empirical evidence presented in Table 1 strongly supports the authors' key claims:
1. **The Generalization-Granularity Trade-off:**
   - Coarse-grained L1 Global Adam (23.21%) and ES (24.84%) show severe underfitting (low capacity).
   - Moving to intermediate granularities (L2 to L4) increases performance to ~29.1% (Adam) and ~29.9% (ES), representing a performance plateau where standard deviations overlap.
   - Going to unregularized L5 Tensor-wise causes a clear performance drop: Adam falls to **26.91%** and ES falls to **29.43%**, showing transductive overfitting.
2. **Optimizer Overfitting Vulnerability:**
   - Unregularized L5 Adam collapses by **1.47%** from L4 Component-wise, while unregularized L5 ES only drops by **0.55%**. This supports the claim that gradient-based Adam is much more vulnerable to rapid local exploitation than the zero-order ES walk.
   - The "sluggishness" explanation of ES robustness is supported by the fact that as dimensionality increases, ES performance reverts closer to the uniform baseline initialization (30.41%), indicating it is failing to optimize far from the starting point.
3. **Effectiveness of Regularization:**
   - Adding ESR and TV to L5 ES recovers its performance to **30.17%** (nearly full recovery).
   - Adding ESR and TV to L5 Adam recovers performance to **28.51%** (+1.60% improvement). However, it remains significantly below the uniform baseline (30.41%) and even L2 Layer-wise Adam (29.18%), supporting the claim that soft L2 penalties cannot fully arrest gradient overfitting.

## Missed Opportunities and Weaknesses in the Setup
1. **Varying Calibration Batch Sizes ($N$):**
   - The study evaluates transductive overfitting on a single fixed calibration batch size ($N=256$). 
   - A critical aspect of transductive overfitting is its dependence on data volume. If the calibration stream was larger (e.g., $N=1024$ or $N=4096$), the overfitting threshold would likely shift, potentially allowing Level 5 Tensor-wise merging to overcome the uniform baseline and achieve higher generalization.
   - Investigating how the optimal structural granularity scales with calibration stream size ($N$) is a major missed opportunity that would have provided invaluable, actionable scaling laws for practitioners.
2. **Extremely Low Expert Accuracies:**
   - As discussed in the soundness evaluation, the task-specific experts are extremely weak (e.g., SVHN at 17.50% vs. 10% random chance, CIFAR-10 at 24.93% vs. 10% random chance).
   - This low-performance setting represents a "stress test" but does not reflect real-world deployment scenarios where experts are fully converged. The conclusions regarding the absolute failure of test-time adaptation to beat static blending should be heavily caveated as potentially unique to low-fidelity models.
