# 4. Experimental Evaluation and Empirical Claims

## Experimental Design and Datasets
The experimental setup is standard and well-controlled, though restricted in scale:
* **Model Backbone**: The choice of CLIP ViT-B/32 is highly appropriate and aligns with foundational model merging literature.
* **Datasets**: The authors evaluate on a 4-dataset visual suite: MNIST, FashionMNIST, CIFAR-10, and SVHN. This represents diverse domain shifts (synthetic digits, clothing, natural objects, real-world numbers) and task complexities. However, evaluating on only 4 datasets is a notable limitation. Most contemporary CLIP model merging papers evaluate on an 8-dataset suite (adding EuroSAT, DTD, RESISC45, GTSRB, etc.).
* **Test Sub-Sampling**: To manage computational overhead, the authors evaluate on a subset of 1024 randomly sampled test images per dataset. While sub-sampling is a practical and necessary constraint under strict resource limits, it introduces potential statistical variance. The authors mitigate this by running all experiments across three independent random seeds (42, 100, 2026) and reporting standard deviations. This makes the empirical results statistically reliable.

## Baselines
The baselines selected are comprehensive and highly representative of the current literature:
* **Zero-Shot Baselines**: Task Arithmetic (TA) (the foundational method), TIES-Merging (a prominent pruning and sign-agreement baseline), and DARE (a random sparsification baseline).
* **Test-Time Adaptation (TTA) Baselines**: Task-Wise AdaMerging (optimizing 4 parameters) and Layer-Wise AdaMerging (optimizing 52 parameters).
* **Baseline Fairness**: Standard Task Arithmetic, TIES-Merging, and DARE are all evaluated using the exact same global scaling factor $\lambda_0 = 0.3$, ensuring an apples-to-apples comparison of weight-space transformations under a fixed scaling budget.
* **Omitted Baselines**: The omission of RegMean is explicitly justified (it requires activation covariance matrices on calibration data, violating the data-free, zero-shot setting), which is scientifically sound.

## Analysis of Empirical Claims and Evidence

### Claim 1: NETA resolves task vector magnitude imbalances, outperforming standard Task Arithmetic zero-shot on MNIST and FashionMNIST.
* **Evidence**: Supported by Table 1. On MNIST, NETA ($\alpha = 1.0$) achieves **96.29% ± 0.45%** (vs. 96.03% ± 0.26% for TA). On FashionMNIST, NETA achieves **82.75% ± 0.80%** (vs. 82.10% ± 0.64% for TA). NETA also substantially outperforms TIES-Merging on both datasets (+2.02% on MNIST, +4.20% on FashionMNIST).
* **Verdict**: Fully supported.

### Claim 2: There is an honest peak performance vs. representation fairness trade-off.
* **Evidence**: Supported by Table 1. On SVHN, NETA's accuracy is reduced compared to Task Arithmetic (77.02% ± 0.56% vs. 80.14% ± 1.07%). Because SVHN has the largest update norm, Task Arithmetic is heavily biased toward it. Equalizing norms prevents SVHN from dominating, which slightly reduces its peak performance to restore a fairer, more balanced multi-task representation across all four experts.
* **Verdict**: Fully supported. The authors are commendable for their scientific honesty and transparency regarding this trade-off, rather than trying to hide the slight drop in average accuracy (87.17% vs. 87.76%).

### Claim 3: Test-time adaptation (TTA) via joint entropy minimization suffers from the "Overfitting-Optimizer Paradox."
* **Evidence**: Supported by Table 1. Under Task-Wise AdaMerging, FashionMNIST accuracy collapses to **77.54% ± 0.00%** (a drop of -4.56% from TA) and CIFAR-10 drops to **89.70% ± 0.54%** (a drop of -3.07%). 
* **Explanation**: The authors provide a compelling explanation: because prediction entropy depends heavily on task difficulty, the unsupervised optimizer minimizes joint entropy by solely optimizing for simpler, highly confident tasks (MNIST and SVHN) while suppressing harder, high-entropy tasks (FashionMNIST and CIFAR-10) from 0.30 to 0.23-0.24. This relative suppression is sufficient to degrade their weight representations, resulting in a delicate, transductive overfit state.
* **Layer-Wise Exemption**: The authors also explain why Layer-Wise AdaMerging does not collapse (average 90.89%). With 52 continuous parameters, the optimizer has enough spatial degrees of freedom to suppress a hard task in some layers while strongly amplifying it in others, satisfying joint entropy without globally suppressing any task. However, they rightly point out that this is highly overparameterized, transductive, and prone to calibration-set overfitting.
* **Verdict**: Fully supported. This analysis is the strongest part of the paper and represents a highly valuable contribution to the community's understanding of test-time optimization.

## Ablation Studies and Extensions
The ablation studies in Table 2 and Table 3 provide exceptional empirical support for the authors' methodology and theoretical claims:
1. **$\alpha$-Relaxation**: Setting $\alpha = 0.5$ recovers SVHN performance to **78.55% ± 0.65%** while maintaining superior performance over Task Arithmetic on MNIST (96.16%) and FashionMNIST (82.62%), yielding a higher average accuracy of **87.51%**.
2. **Composite Grouping (Group 0)**: Omitting Group 0 degrades performance across MNIST and FashionMNIST, validating the heuristic that normalizing positional embeddings and class tokens independently introduces early positional distortions.
3. **Noise-Damping Stabilizer ($\beta$)**: NETA is highly robust to variations in $\beta$ ($10^{-6}$ vs. $10^{-3}$ and $10^{-2}$), confirming its stability.
4. **Scale Compensation ($\gamma^l$)**: Applying the closed-form scale-compensation factor $\gamma^l$ successfully yields improvements across all metrics (average accuracy increases to **87.28%**, with MNIST rising to 96.32% and FashionMNIST to 82.85%), showing that resolving directional norm contraction analytically is highly effective.
5. **Grid Search over $\lambda_0$**: Table 3 shows that when both methods are optimized over $\lambda_0$, Task Arithmetic remains slightly superior on average ($89.16\%$ vs. $89.06\%$ for relaxed NETA). This is a very honest detail that shows NETA acts as a representational fairness regularizer rather than a raw average accuracy maximizer.
