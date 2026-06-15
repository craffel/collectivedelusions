# 4. Experiment Check

A critical evaluation of the experimental setup, datasets, baselines, and whether the results actually support the claims.

## Evaluation of Experimental Setup and Datasets
The experimental setup is designed with high scientific integrity and rigor:
- **Model Backbone**: The use of a standard compact Vision Transformer (`vit_tiny_patch16_224`) is an appropriate choice to reproduce and deconstruct prior work.
- **Datasets**: The selection of MNIST, FashionMNIST, CIFAR-10, and SVHN provides a diverse set of tasks ranging from simple grayscale digit classification to complex real-world digit and natural object recognition. This offers a challenging OOD and heterogeneous task mixture.
- **Calibration Set**: A deterministic set of 16 samples per task (64 total images) is drawn, which aligns with standard training-free model merging constraints.
- **Multi-Seed Sweep**: Evaluating across 5 random calibration seeds ensures that the results are statistically reliable and not the result of a lucky draw.
- **2D Hyperparameter Grid Search**: The sensitivity analysis across a wide range of $L_2$ regularization coefficients ($\alpha$) and temperatures ($T$) demonstrates the stability of the method across diverse settings.

## Evaluation of Baselines
The baselines used in the paper are comprehensive and highly representative:
1. **Individual Experts (Ceiling)**: Correctly establishes the empirical upper bound ($96.27\%$ Joint Mean).
2. **Uniform Merging (Task Arithmetic)**: Represents a simple static merging baseline.
3. **OFS-Tune**: A state-of-the-art static model merging method, which represents a strong baseline for the heterogeneous test-stream experiments.
4. **AdaMerging (TTA)**: An unsupervised dynamic test-time adaptation baseline.
5. **Classical Linear Router**: An unregularized classical linear gating model.
6. **QWS-Merge (Reported & Local Baseline)**: The key comparison. The authors' inclusion of a **locally re-implemented QWS-Merge trained on the exact same expert weights** is an exemplary practice that ensures absolute fairness, eliminating any confounding factors such as differing expert checkpoints.

## Do the Results Support the Claims?
We evaluate the paper's primary claims against the empirical evidence provided:

### Claim 1: Classical linear routing does not suffer from structural limitations and does not catastrophically collapse on SVHN.
- **Verdict: Fully Supported.**
- **Evidence**: The classical unregularized Linear Router achieves $94.87\%$ SVHN accuracy and $95.46\%$ Joint Mean accuracy on seed 42. In the multi-seed sweep, it averages $91.20\% \pm 1.85\%$ SVHN accuracy and $91.53\% \pm 0.41\%$ Joint Mean accuracy across 5 seeds. This is vastly superior to the $15.30\%$ SVHN accuracy reported in Vance et al. (2025) and local baseline results of $88.40\%$. The diagnostic configuration table (Table 2) clearly identifies the sub-optimal parameters (routing layer, high learning rate, over-optimization) that likely caused the reported collapse in prior work, convincingly demonstrating that classical linear routing is inherently robust.

### Claim 2: Robust Linear Routing (RLR) stabilizes dynamic parameter fusion under heterogeneous test streams.
- **Verdict: Partially Supported with modest effect sizes.**
- **Evidence**: Table 4 shows that as the evaluation batch size increases from $B=1$ to $B=256$, RLR consistently maintains an accuracy advantage over the unregularized Linear Router ($76.85\%$ vs $75.48\%$ at $B=16$; $75.03\%$ vs $73.15\%$ at $B=256$). This proves that the regularization is working as intended to smooth out logit variance. However, **the magnitude of this benefit is modest (1.3% to 1.8% absolute improvement)**, and both RLR and the unregularized router still drop by more than 17% absolute accuracy compared to the $B=1$ setting (heterogeneity collapse). Crucially, static OFS-Tune remains vastly superior at $B=256$ ($86.23\%$ accuracy vs. RLR's $75.03\%$). Thus, while RLR is a "stabilizer" that mitigates some of the collapse, it does not solve the fundamental vulnerability of dynamic merging in heterogeneous settings.

### Claim 3: In homogeneous settings, RLR and unregularized routing are statistically indistinguishable.
- **Verdict: Fully Supported.**
- **Evidence**: The authors honestly acknowledge that across 5 seeds, the classical Linear Router achieves $91.53\% \pm 0.41\%$ Joint Mean accuracy while RLR achieves $91.46\% \pm 0.42\%$. This demonstrates scientific honesty, showing that in standard settings, RLR's regularization is redundant and does not yield an empirical benefit.

### Claim 4: The performance is robust to hyperparameter selections.
- **Verdict: Fully Supported.**
- **Evidence**: The 2D sensitivity analysis (Figure 4) shows that SVHN and Joint Mean accuracies remain remarkably stable across a wide grid of $\alpha \in [0.0, 0.02]$ and $T \in [1.0, 5.0]$, confirming that RLR does not require delicate hyperparameter tuning to achieve stable convergence.
