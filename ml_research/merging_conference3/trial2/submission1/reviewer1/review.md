# Peer Review Report

## Summary of the Paper
This paper investigates **test-time model merging**, an emerging paradigm that combines multiple task-specific expert neural networks (fine-tuned from a shared pre-trained base model) into a single multi-task model at test-time, without accessing joint training data. Specifically, it deconstructs adaptive test-time model merging (e.g., AdaMerging) and exposes two severe, previously under-reported failure modes:
1. **The Overfitting-Optimizer Paradox (Transductive Overfitting)**: Where fine-grained layer-wise coefficients overfit to the statistical noise of tiny, restricted test-time calibration batches, rather than learning stable spatial interactions.
2. **Sacrificial Task Bias**: Where joint entropy-based optimization landscapes systematically degrade complex, high-entropy tasks (e.g., SVHN) to prioritize easier, low-entropy domains (e.g., MNIST).

To address these limitations, the authors introduce **RegCalMerge**, a robust, calibration-aware test-time model merging framework. The framework is composed of:
* **A Core Calibration Engine (CalMerge)**: Combining **Class-Capacity Normalization (CCN)** (mapping task entropies to a uniform $[0, 1]$ interval via normalization by $\log C_k$) and **Scale-Normalized Entropy Weighting (SNEW)** (scaling task entropies by the inverse of their step-0 baseline uniform task arithmetic entropy) to completely eliminate sacrificial task bias.
* **Elastic Spatial Regularization (ESR)**: An optional structural stabilizer that applies a dual **Proximity Penalty** ($\beta$) to keep coefficients near their robust uniform initialization and a **Spatial Deviation Penalty** ($\gamma$) to penalize the variance of layer-wise coefficients around their task-wise spatial average.

---

## Strengths and Weaknesses

### Strengths:
1. **Rigorous Empirical Deconstruction**: The paper's primary strength is its self-critical, analytical approach. Instead of merely proposing another performance-boosting heuristic, it carefully deconstructs standard test-time model merging.
2. **Elegant Spatial Shuffling Diagnostic**: The introduction of the spatial shuffling diagnostic is a highly innovative and convincing empirical check. Showing that randomly shuffling optimized coefficients across layers retains almost all performance gains (~95% of optimization benefits) is an elegant proof that standard layer-wise adaptation behaves as a transductive parameter-drift mechanism.
3. **Calibrated Baseline Isolation**: The authors introduce a **Calibrated Spatial Mean (Cal-Mean)** baseline to isolate the causal contributions of layer-wise flexibility versus calibration. Comparing CalMerge to Cal-Mean demonstrates a clear, absolute **0.69%** Joint Mean advantage (with massive improvements on CIFAR-10 (+6.51%) and SVHN (+1.95%)), proving that layer-wise spatial degrees of freedom are indeed necessary and beneficial.
4. **Heterogeneous Class-Capacity Simulation**: To address the homogeneous class limit ($C_k = 10$) of the main visual suite, the authors go out of their way to design and execute a specialized simulation in Section 4.3.3. Restricting task class spaces ($C_k \in [3, 5, 8, 10]$) and showing how SNEW/CCN mathematically resolves gradient imbalances is an outstanding demonstration of empirical thoroughness.
5. **Transparency and Scientific Integrity**: The authors are exceptionally honest about their deterministic path convergence ($\pm0.00\%$ standard deviation for Adam GD across seeds) and explain the underlying mathematical reasons (static splits and cached batches), which is refreshing and highly commendable.

### Weaknesses:
1. **Extremely Small Evaluation Splits**: The evaluation split size (256 images per domain) is very small. On a test set of 256 images, a single image misclassification corresponds to exactly **0.39%** accuracy. Looking at Table 1, the reported improvements of CalMerge over standard unconstrained Adam GD on MNIST (57.81% vs 57.42%) and CIFAR-10 (85.16% vs 84.77%) are exactly **0.39%** (literally a **single image** difference). On such a tiny evaluation set, these microscopic improvements are highly susceptible to sample selection noise and cannot be confidently declared statistically significant.
2. **Static Seed Split Variance**: While reporting standard deviations across three seeds is helpful, the random seeds do not vary the actual data splits for the deterministic first-order gradient methods (Adam GD). Using different random splits (for both calibration and evaluation) across seeds (e.g., via bootstrapping or cross-validation) would provide true, meaningful confidence intervals and statistical error bars.
3. **Lack of Calibration Stream Scaling Analysis**: The calibration stream is restricted to 1 batch of size 16 per dataset. It remains unclear how the Overfitting-Optimizer Paradox behaves as a function of the calibration stream size (e.g., $N \in [16, 64, 256, 1024]$). If the paradox diminishes or disappears with larger calibration streams, the necessity of the ESR stabilizer would be restricted only to ultra-low-data budgets.
4. **Missing Standard Baselines**: The main results table lacks comparisons with standard static merging baselines like **TIES-Merging** (which explicitly addresses parameter conflict) and **DARE**, as well as other adaptive test-time merging frameworks like **SyMerge**.
5. **Noisy or Mixed-Task Calibration Streams**: The evaluation assume clean, in-distribution, task-pure calibration batches. Real-world test-time adaptation involves noisy or task-mixed streams. The behavior of SNEW/CCN under non-ideal calibration streams is left unexamined.

---

## Detailed Evaluation

### Soundness: **Good**
The technical claims and derivations are mathematically sound, and the introduction of a calibrated scalar baseline (Cal-Mean) to isolate the causal impact of spatial flexibility is methodologically exemplary. However, the soundness of the empirical claims is limited by the tiny evaluation splits (256 samples), which make performance differences of 1-3 images look like "state-of-the-art" improvements, and the lack of dataset split variance across the random seeds. 

### Presentation: **Excellent**
The paper is beautifully written, exceptionally structured, and easy to follow. Concepts are communicated clearly, and equations are precise. The Related Work section is thorough and properly positions this work in the broader static and adaptive model merging literature. The discussion of limitations (Section 4.3) is incredibly honest and demonstrates high academic rigor.

### Significance: **Good**
The paper addresses an important problem and advances our understanding of test-time model merging by exposing transductive overfitting. The proposed SNEW/CCN calibration and ESR regularization provide a controllable safety dial that is of high practical value. However, the significance is slightly bounded by the small-scale classification tasks evaluated (MNIST, FashionMNIST, CIFAR-10, SVHN) and the lack of scaling analysis to larger architectures or more complex domains (like LLMs or large-scale heterogeneous benchmarks).

### Originality: **Excellent**
The paper exhibits high originality. Exposing transductive overfitting via a spatial shuffling diagnostic is a highly creative and insightful contribution. ESR is a novel and mathematically elegant bridging mechanism between uniform static merging (0 degrees of freedom) and fully adaptive test-time merging (unconstrained degrees of freedom), offering a scale-invariant normalized penalty.

---

## Overall Recommendation

**Rating: 4 (Weak Accept)**

**Justification**: This is a technically solid, exceptionally well-written paper that makes a substantial conceptual and empirical contribution by deconstructing the standard test-time model merging paradigm. The spatial shuffling diagnostic and the isolation of spatial degrees of freedom via the Calibrated Spatial Mean baseline are highly high-signal contributions that the community is likely to build on. 

However, the experimental evaluation has notable weaknesses, particularly the extremely small evaluation splits (256 images) where SOTA gains are literally a matter of a single image classification, and the lack of true split variance across random seeds. If the authors can address these experimental concerns or expand their evaluation to full test sets, this paper could easily be a strong accept.

---

## Questions and Actionable Feedback for the Authors

1. **Evaluation Split Scale**: Why were the evaluations restricted to tiny splits of 256 images instead of the standard, full-scale test sets (e.g., 10,000 images per domain)? Standard test splits would drastically reduce sample selection noise and confirm whether the 0.39% improvements are statistically significant. I highly encourage the authors to report results on full standard test sets.
2. **True Statistical Variance across Seeds**: For first-order gradient methods (Adam GD, CalMerge), the standard deviation across seeds is reported as $\pm0.00\%$ because the splits are static and cached. To conduct a truly robust statistical evaluation, have you considered running your seed replications across different random draws of both the calibration and evaluation splits? Reporting bootstrapped error bars would make your empirical claims far more convincing.
3. **Calibration Stream Scaling**: How does the Overfitting-Optimizer Paradox scale as the calibration batch size increases (e.g., from 16 to 64, 256, or 512 samples)? Does standard unregularized AdaMerging eventually learn stable spatial interactions and stop overfitting as more calibration data becomes available?
4. **Missing Baselines**: To make your comparison comprehensive, please include results for standard merging methods such as **TIES-Merging** and **DARE**, and compare RegCalMerge against other test-time adaptive merging baselines like **SyMerge** in your main evaluation.
5. **Noisy and Mixed Calibration Streams**: In practice, test-time data streams are noisy and contain mixed tasks. How do SNEW and CCN behave if the calibration batch contains a mixture of samples from MNIST, CIFAR-10, and SVHN, or if there is out-of-distribution noise? Can SNEW be computed online or does it strictly require task-pure calibration batches?
