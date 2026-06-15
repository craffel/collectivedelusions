# Empirical & Experiment Check: GranMerge

An evaluation of the experimental design, quantitative results, and statistical claims presented in the paper.

## 1. Quantitative Analysis of Results (Table 1)

Let's carefully examine the test-set generalization accuracies reported in Table 1:

| Merging Strategy | MNIST | FashionMNIST | CIFAR-10 | SVHN | Overall Mean | Std Dev |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Uniform Task Arithmetic (Baseline)** | 39.07 | 48.97 | 19.67 | 13.93 | **30.41** | 1.48 |
| **Adam Level 5 (With Reg)** | 35.83 | 45.97 | 19.33 | 12.90 | **28.51** | 2.01 |
| **1+1 ES Level 5 (With Reg)** | 38.80 | 48.57 | 19.63 | 13.70 | **30.17** | 1.67 |

### Key Empirical Observations:
1.  **The Supremacy of the Static Baseline:** The simple, zero-overhead **Uniform Task Arithmetic** baseline achieves **30.41%** overall mean accuracy. No test-time adaptive configuration, even when regularized with ESR and TV, outperforms this static baseline.
2.  **Regularization Recovery & Limitations:** 
    *   For **1+1 ES**, enabling ESR and TV successfully recovers Level 5 performance from 29.43% (unregularized) back to **30.17%** (a 0.74% recovery), nearly reaching the uniform baseline.
    *   For **Adam**, the regularizers recover Level 5 performance from 26.91% to **28.51%** (a 1.60% recovery). Although this is a substantial and robust stabilization effect, Adam remains more vulnerable to overfitting, lagging behind the baseline by 1.90%.
3.  **Surrogate Loss Misalignment Analysis:** The authors provide a brilliant diagnostic analysis of why adaptation fails to beat the uniform baseline. They identify a fundamental misalignment between the unsupervised surrogate loss (prediction entropy) and true classification accuracy under small calibration budgets ($N=256$). The optimizer finds degenerate, "confident but incorrect" parameter configurations that yield low entropy but incorrect classes. Under high parameter resolution, this misalignment is severely amplified, as the optimizer has enough degrees of freedom to fit the transductive noise of the local calibration batch without learning generalizable features.

## 2. Statistical Significance and Margin of Error
The results are averaged over 3 random seeds. 
The standard deviations reported in Table 1 (ranging from 1.18% to 2.13%) indicate that while adjacent intermediate granularities (e.g., L3 vs. L4 ES) have overlapping standard deviation margins, the macro-level findings are highly robust:
*   The sharp collapse of unregularized L5 Adam (26.91%) compared to the baseline (30.41%) and L4 Component-wise (28.38%) is statistically distinct and highly significant.
*   The recovery of L5 Adam under regularizations (28.51%) and L5 ES under regularizations (30.17%) represents clear and robust stabilization trends.
The authors are extremely careful and honest about these statistical margins, explicitly discussing them in Section 4.3 and Section 4.4.

## 3. Visualizations and Diagnostics
The paper includes a highly descriptive conceptual trade-off curve in Figure 1, demonstrating the Generalization-Granularity curve. Additionally, the authors discuss the qualitative drift behavior of coefficients in the appendix, providing strong support for their analytical claims regarding optimization trajectories.
