# Experimental Evaluation Check

## Experimental Setup and Datasets
The experimental setup is highly artificial and detached from real-world model merging practices:
* **Datasets:** MNIST, FashionMNIST, CIFAR-10, SVHN. While standard in basic computer vision, these are not the datasets typically used in model merging literature (which focuses on large-scale Vision-Language Models or LLMs).
* **Dimensionality Reduction:** The raw images are projected down to 192 features using random Johnson-Lindenstrauss matrices. This projection severely degrades the image features.
* **Toy Architecture:** The model is a 14-layer residual Multi-Layer Perceptron (MLP) of width 64.
* **Extremely Poor Performance:** Due to this artificial setup, the classification accuracies are incredibly low. CIFAR-10 accuracy is **12.89%** and SVHN is **15.71%**, which are barely above the random guessing baseline of 10.0%. It is highly questionable to draw any generalizable machine learning conclusions from a model that is practically failing to learn the tasks.

## Baselines
The authors compare their method to Static Uniform, Ties-Merge, DARE-Merge, Offline Unconstrained, and RBPM.
* **Inappropriate Baselines:** Ties-Merge and DARE-Merge are designed for modern transformers with thousands of channels. Applying them to a custom 14-layer residual MLP with random projection features is highly artificial. 
* **Baselines Performing Worse Than Average:** In this pathological setup, both Ties-Merge (29.59%) and DARE-Merge (32.76%) perform significantly *worse* than the simple Static Uniform average (33.35%). This indicates that the experimental sandbox is highly non-standard and skewed.

## Claims vs. Results: Discrepancies and Potential Fabrication

There is a massive, critical mismatch between the claims in the text and the actual numbers in Table 1 and `results.json`.

### 1. The Abstract/Conclusion vs. Table 1 Mismatch
In the **Abstract** and **Conclusion**, the authors claim:
* Our advanced non-isotropic Fisher-guided PAC-Bayes-FIM Merge yields **36.13%** Joint Mean accuracy.
* This outperforms the Static Uniform baseline (**33.57%**), Ties-Merge (**29.68%**), DARE-Merge (**33.24%**), and unconstrained layer-wise tuning (**36.09%**).

However, in **Table 1** (and in the raw `results.json`), the actual numbers are:
* Ours (FIM Deterministic Compiled): **35.37%** (not 36.13%)
* Static Uniform: **33.35%** (not 33.57%)
* Ties-Merge: **29.59%** (not 29.68%)
* DARE-Merge: **32.76%** (not 33.24%)
* Offline Unconstrained: **35.51%** (not 36.09%)

This is a profound scientific reporting error. The text of the abstract and conclusion claims inflated numbers that do not match the main results table.

### 2. Contradiction of the Core Claim
The paper's core claim is that their method outperforms unconstrained layer-wise tuning. 
* In the **Abstract/Conclusion text**, they claim: PAC-Bayes-FIM (**36.13%**) outperforms unconstrained tuning (**36.09%**).
* In **Table 1 / JSON**, the actual results show: PAC-Bayes-FIM (**35.37%**) actually **underperforms** unconstrained tuning (**35.51%**)!

The actual data shows that their proposed method does NOT outperform the unconstrained baseline. It is numerically worse. Their main claim is completely false and contradicted by their own data.

### 3. Internal Inconsistency: Table 1 vs. Table 2 (Ablation)
In **Table 1**, their default method (Ours Deterministic Compiled) is reported as achieving **35.37%** Joint Mean accuracy.
In **Table 2** (Ablation), for the *exact same default parameters* ($\lambda_{\text{PAC}} = 0.010, \sigma = 0.05$), they report a Joint Mean of **36.09%** (and the sub-task accuracies are completely different: e.g., MNIST is 61.63% in Table 2 vs. 59.72% in Table 1).

Why does the exact same method under the exact same default settings have two completely different sets of results in Table 1 and Table 2? This suggests that the tables are either using different datasets, different seeds, or that the numbers were compiled from inconsistent runs or fabricated. This severely damages the scientific integrity of the experimental section.

### 4. Insignificance Under Noise
The standard deviation of the Joint Mean accuracy is **2.81%** for their method and **2.63%** for the unconstrained baseline. The numerical difference between the unconstrained baseline (35.51%) and their method (35.37%) is a mere **0.14%**, which is completely buried in the standard deviation of ~2.7%. Thus, even if the numbers were consistent, any claimed difference is statistically meaningless.
