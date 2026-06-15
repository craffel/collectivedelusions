# Evaluation 4: Experimental Evaluation

## Experimental Setup and Datasets
- **Model Backbone**: CLIP ViT-B/32 (86M parameters), which is a standard but relatively small-scale pre-trained transformer model.
- **Datasets**: MNIST, FashionMNIST, CIFAR-10, SVHN. These are small-scale, homogeneous classification datasets (all feature exactly 10 categories). Using these toy-scale datasets limits the generalizability of the findings to more complex, modern vision or language tasks.
- **Scale of Evaluation**:
  - The calibration stream is exceptionally small: **1 unlabeled batch of size 16 per dataset** (64 samples total).
  - The test split is also highly restricted: **256 images per domain** (two batches of size 128). While this is sufficient for a quick, data-efficient test-time adaptation analysis, such a small test split is highly vulnerable to sample-split noise. Evaluating on full validation splits would provide more reliable and robust generalization metrics.

## Baseline Comparisons
The baseline selection is very comprehensive and represents a strong scientific control:
1. **Task Arithmetic (Uniform)**: Standard parameter vector addition.
2. **Unconstrained AdaMerging (Adam GD & 1+1 ES)**: The standard test-time adaptive model merging framework.
3. **Spatially Averaged AdaMerging (Mean)**: Collapses parameters to a single scalar per task to prevent drift.
4. **Calibrated Spatial Mean (Cal-Mean)**: A newly introduced baseline combining SNEW/CCN calibration with the Spatially Averaged (4-parameter) model.
5. **Spatially Shuffled Diagnostics**: Shuffles optimized layer-wise coefficients to test spatial localization.

This is a highly rigorous suite of baselines that allows for isolating the effects of calibration vs. parameter reduction.

## Do the Results Support the Claims?

While the empirical findings are thoroughly documented across multiple seeds, a critical analysis of the results reveals that they do not fully support the authors' claims regarding the necessity of complex layer-wise parameters:

1. **The Overfitting-Optimizer Paradox (Fully Supported)**:
   - The results in Table 1 strongly support this claim. When the optimized coefficients are spatially shuffled (Method 5), the model retains 60.94% accuracy—nearly identical to the unshuffled 61.62% accuracy. This clearly confirms that the 52 layer-wise parameters are heavily overfitted to the transductive calibration batch rather than learning genuine layer-specific interactions.

2. **Sacrificial Task Bias (Fully Supported)**:
   - The results show a clear drop in SVHN performance under standard unconstrained optimization (to 28.26% under 1+1 ES). SNEW-based CalMerge (Method 8) successfully restores SVHN performance to 32.03% and Joint Mean to 61.82%, verifying that Scale-Normalized Entropy Weighting is an effective solution to sacrificial task bias.

3. **The Necessity of Layer-wise Flexibility (Weakly Supported / Contradicted by Minimalism)**:
   - The authors claim that "fine-grained layer-wise parameter flexibility is indeed necessary and not redundant" because CalMerge (61.82%) outperforms the Calibrated Spatial Mean baseline (61.13%).
   - However, from a minimalist perspective, **Cal-Mean is a far more elegant, simple, and robust model**. It optimizes only **4 parameters** (one scalar per task) instead of **52 parameters** (layer-wise). 
   - For a massive $13\times$ reduction in parameter space complexity, Cal-Mean achieves 61.13% Joint Mean accuracy, which is **only 0.69% lower** than CalMerge. 
   - Furthermore, Cal-Mean is **completely immune to the Overfitting-Optimizer Paradox by construction**, requires no complex spatial regularization (ESR), and requires no hyperparameter tuning. The extremely marginal gain of 0.69% from CalMerge does not justify the massive risk of transductive overfitting and the need for complex, layer-wise optimization.

4. **The Generalization-Regularization Trade-off of ESR (Failed Practical Utility)**:
   - The authors claim that ESR provides a "stable, highly predictable generalization surface." However, the ablation study (Table 2) reveals that any positive value of $\beta$ or $\gamma$ strictly degrades performance.
   - At standard settings ($\beta=1, \gamma=1$), ESR degrades the joint mean accuracy of RegCalMerge to **60.26%**, which is **worse than naive Task Arithmetic (60.35%)** that has zero test-time optimization and zero hyperparameters.
   - Therefore, the empirical results show that ESR is of **no practical utility**. It introduces complex mathematical constraints and two new hyperparameters only to deliver performance inferior to the most basic, training-free baseline.
