# Intermediate Evaluation: Experimental Check

This document provides a critical empirical evaluation of the experimental setup, datasets, baselines, and whether the results support the paper's claims, written from a rigorous empiricist perspective.

## 1. Experimental Setup and Dataset Evaluation
- **Backbone:** The authors evaluate their method on a **ViT-Tiny** backbone (`vit_tiny_patch16_224`). While ViT-Tiny is a standard academic sandbox, it is a very small model (5.7M parameters). Modern model merging is typically deployed on much larger models (e.g., ViT-Base/Large, LLaMA-7B, Mistral-7B). Testing exclusively on a tiny model raises concerns about whether these dynamics scale to larger architectures.
- **Datasets:** The evaluation uses four vision datasets: **MNIST, FashionMNIST, CIFAR-10, and SVHN**. These datasets are highly canonical but relatively simple and represent low-resolution (28x28 and 32x32) inputs upscaled to 224x224.
- **SVHN Performance Bottleneck:** The fine-tuned SVHN expert achieves a test accuracy of only **64.60%** (due to fine-tuning for only 5 epochs). This weak expert behaves as a severe performance bottleneck for all merged models, resulting in SVHN performance hovering between **17.00% and 30.00%** (where 10% is random guess). Evaluating model merging on a poorly converged task expert introduces high variance and undermines the reliability of the average multi-task accuracy metrics. The experts should be fully converged to represent a realistic merging scenario.

## 2. Statistical Soundness and Evaluation Gaps

A major weakness of the empirical evaluation is the **complete absence of statistical rigor**:

- **No Multi-Seed Evaluation:** All tables (Tables 1, 2, 3, 4, 5, 6) present single point estimate percentages (e.g., 65.62%). There is **no mention of running multiple random seeds**, and there are **no standard deviations, confidence intervals, or error bars**.
- **High Sensitivity in Sparse Regimes:** The calibration set size is extremely small ($N=64$, containing only 16 samples per task). In such low-data regimes, the optimization process and the offline covariance matrix estimation are highly sensitive to the specific random split of the 64 samples. Without reporting the mean and standard deviation across multiple random splits and seeds (e.g., 5 or 10 runs), it is impossible to determine whether the reported differences are statistically significant or merely statistical noise.
- **Unsubstantiated Marginal Differences:** For example, in Table 5, at $N=128$, CFR is shown to outperform L2 decay by a mere **0.12%** (66.50% vs. 66.38%). At $N=256$, the gap is **0.24%** (67.12% vs. 66.88%). Without error bars, these tiny fractional differences are highly likely to be within the margin of random seed variance, rendering the claimed "superiority" of CFR over L2 decay unproven.

## 3. Baseline Adequacy and Fairness of Comparison
The authors include a competitive set of baselines: Static Uniform, AdaMerging, Global Linear Router (Unreg), QWS-Merge, Standard L2 Reg L3-Router, and Static Layer-Wise (Optimized). However, a closer look at the results reveals that **the proposed method is outperformed by its own baselines**:

- **Standard L2 Reg L3-Router Outperforms R2D-Merge:**
  - Under **Homogeneous Streams**, Standard L2 Reg achieves **66.88%** average accuracy, whereas R2D-Merge achieves **65.62%** (L2 is **+1.26%** better).
  - Under **Heterogeneous Collapsed Streams**, Standard L2 Reg achieves **65.88%** average accuracy, whereas R2D-Merge achieves **65.62%** (L2 is **+0.26%** better).
  - Standard L2 decay is computationally much simpler, requires no offline profiling step, zero storage of auxiliary matrices, and no loading workflows. The fact that standard L2 Reg outperforms R2D-Merge on average in both settings strongly undercuts the practical value of CFR.
- **The "Dynamic Collapse" Paradox:** 
  - Under CFR ($\lambda_{\text{wd}} = 10^{-2}$), the router weights shrink to almost zero ($\mathcal{M}_{\text{drift}} \approx 0.012$), causing the model to act as a static layer-wise merger.
  - Crucially, the **Static Layer-Wise (Optimized) baseline** (which sets $w_{l,k}=0$ and only optimizes biases) achieves **exactly identical accuracy (65.62%)** to R2D-Merge across all three evaluation stream configurations.
  - Therefore, R2D-Merge does not provide any empirical advantage over a simple static model. Setting the routing weights to zero offline is a much simpler, zero-overhead solution that matches R2D-Merge's performance exactly.

## 4. Do the Results Support the Claims?
- **Claim:** R2D-Merge "achieves comparable multi-task accuracy while demonstrating absolute resilience under batch-averaged heterogeneous streams (0.00% drop)."
  - **Verdict:** Yes, but this claim is highly misleading. The model achieves 0.00% drop because **the dynamic router collapses to a static model**. An actually dynamic router (like standard L2 Reg) achieves **65.88% collapsed accuracy** (which is *higher* than R2D-Merge's 65.62%), despite having a tiny -1.00% collapse drop. Most practitioners would prefer a model with 65.88% accuracy over one with 65.62% accuracy, even if the latter has a mathematically "perfect" 0.00% drop, because the former achieves higher classification performance.
- **Claim:** "CFR's covariance-weighted regularization is essential for maintaining routing capacity on more complex, real-world tasks."
  - **Verdict:** Unsupported. CFR suppresses the router weights to the static limit ($\mathcal{M}_{\text{drift}} \approx 0.012$), so it is actually *suppressing* routing capacity, not maintaining it. 

**Conclusion:** From an empirical perspective, the results do not convincingly justify the use of R2D-Merge. A simple static layer-wise optimized baseline achieves the exact same performance, and a standard L2-regularized L3-Router achieves higher average accuracy across all stream configurations. The evaluation lacks statistical significance testing, which is a major gap in a low-data calibration regime.
