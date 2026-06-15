# Experimental Evaluation and Critique: RegCalMerge

This file provides a critical, empirical evaluation of the experimental design, datasets, baselines, and statistical soundness of the submission.

---

## Evaluation of the Experimental Setup & Datasets
The authors use the **CLIP ViT-B/32 pre-trained image encoder** (86M parameters) as their backbone and target the linear projection layers (`visual.proj`) for merging across $L = 13$ discrete layer groups. The datasets used are standard computer vision benchmarks: MNIST, FashionMNIST, CIFAR-10, and SVHN. 

While these datasets are standard, they are relatively simple toy/small-scale classification tasks. Evaluating on homogeneous 10-class image datasets does not fully capture the complexity of real-world model merging (e.g., merging large language models on diverse reasoning tasks, or merging vision models on large-scale heterogeneous benchmarks like ImageNet, DomainNet, or VTAB).

---

## Deep Empirical Critiques & Flaws

### 1. Unreasonably Small Evaluation Splits (Highly Prone to Sample Noise)
* **Critique**: The authors evaluate their models on extremely restricted test splits: "exactly 2 batches of size 128 (256 test images per domain)". This is an extremely small sample size for evaluation. Standard test splits for these datasets contain 10,000 images.
* **Statistical Impact**: On a test split of 256 images, a single image misclassification corresponds to exactly **0.39%** accuracy.
  Let's look at the reported improvements of Calibrated AdaMerging (CalMerge) over standard Unconstrained AdaMerging (Adam GD) in Table 1:
  - **MNIST**: CalMerge (57.81%) vs. Adam GD (57.42%) $\rightarrow$ Difference of **0.39%** (exactly **1 image** out of 256).
  - **CIFAR-10**: CalMerge (85.16%) vs. Adam GD (84.77%) $\rightarrow$ Difference of **0.39%** (exactly **1 image** out of 256).
  This reveals that the claimed state-of-the-art superiority on two out of the four datasets is literally a matter of **a single image classification**. On such a tiny evaluation set, these microscopic differences are highly susceptible to sample selection noise and cannot be confidently declared as statistically significant or robust.

### 2. Lack of True Statistical Variance / Seed Diversity on Splits
* **Critique**: The authors report a standard deviation of $\pm0.00\%$ across three independent random seeds for Adam GD and CalMerge. They explain that this is because the calibration and evaluation splits are globally cached in-memory and the starting initializations are fixed, resulting in a deterministic gradient trajectory.
* **Empirical Impact**: While this explanation is technically correct, it highlights a major flaw in the experimental design. If the 3 seeds had been used to sample **different calibration batches** and **different evaluation splits** (e.g., bootstrapping or cross-validation), the authors would have obtained true, meaningful empirical standard deviations for all methods. Reporting $\pm0.00\%$ across "seeds" that do not vary the data splits is a missed opportunity to conduct a robust statistical evaluation, and masks the true variance of test-time adaptation under varying data streams.

### 3. Scaling Laws of the Overfitting-Optimizer Paradox
* **Critique**: The calibration stream consists of only **1 unlabeled batch of size 16 per dataset** ($N = 64$ samples total). This represents an extremely tight, low-data regime.
* **Empirical Question**: Does the Overfitting-Optimizer Paradox still occur if we increase the calibration stream to 64, 256, or 1024 samples per task? If we have a larger calibration set, does standard layer-wise AdaMerging generalize better and no longer overfit? The paper treats transductive overfitting as an inherent property of adaptive model merging, but it might simply be an artifact of an extremely low-data calibration budget. Exploring the "sample complexity scaling laws" of the paradox would make the empirical analysis far more comprehensive and convincing.

### 4. Missing Standard Baselines
* **Critique**: While the paper compares against uniform Task Arithmetic and Spatially Averaged AdaMerging, it does not include widely-known static and adaptive model merging baselines in its main results table (Table 1):
  - **TIES-Merging** (Yadav et al., 2023): The standard baseline for resolving sign and magnitude conflicts when merging task vectors.
  - **DARE** (Yu et al., 2023): A standard baseline that uses randomized pruning and scaling to merge models.
  - **SyMerge** (Jung et al., 2025): A concurrent/prior adaptive test-time merging baseline mentioned in the Related Work but missing from the empirical comparison.
* **Impact**: Without comparing against these standard, high-performance merging baselines, it is difficult to determine whether CalMerge represents a genuine advancement over the state-of-the-art in model merging, or if it only outperforms basic Task Arithmetic and basic AdaMerging.

---

## Alignment of Data with Claims
- **Claim**: CalMerge resolves sacrificial task bias to achieve a state-of-the-art Joint Mean of 61.82%.
  - *Alignment*: Supported by the data in Table 1, where CalMerge improves SVHN from 29.69% to 32.03% and MNIST from 55.86% to 57.81%. However, as noted above, the MNIST/CIFAR-10 improvements over Adam GD are only 1 image, and CalMerge actually *degrades* FashionMNIST performance compared to Adam GD (72.27% vs 73.44%, a loss of 3 images). This suggests a trade-off rather than a complete resolution of domain bias.
- **Claim**: Fine-grained layer-wise parameter flexibility is necessary.
  - *Alignment*: Strongly supported. Comparing CalMerge (61.82% Joint Mean) against Calibrated Spatial Mean (61.13% Joint Mean) proves that having layer-wise spatial degrees of freedom is indeed beneficial (+0.69% improvement, especially +6.51% on CIFAR-10), confirming that uniform spatial averaging restricts the model's capacity.
- **Claim**: ESR provides a smooth, predictable generalization surface.
  - *Alignment*: Strongly supported by the 2D grid sweep in Table 2, showing monotonic and predictable transitions as $\beta$ and $\gamma$ are adjusted.
