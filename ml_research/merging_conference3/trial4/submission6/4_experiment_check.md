# Intermediate Review File 4: Experimental Evaluation Check of the Revised Paper

This file evaluates the experimental setup, benchmarks, baselines, and empirical claims of the revised paper.

## 1. Strengths of the Experimental Setup
The revised paper shows major improvements in experimental rigor:
- **Symmetric Hyperparameter Sweeps:** Over forty complete multi-task evaluation sweeps were executed to tune the scaling coefficient $\lambda \in [0.1, 1.0]$ with a step size of $0.1$ across all methods. This ensures absolute fairness.
- **Proper Baseline Implementations:** The baselines (Task Arithmetic, DARE, TIES-Merging) are standard and implemented correctly within the PyTorch-based codebase (`run_experiments.py` and `AdaMerging/src`).
- **Reproducibility:** The codebase is fully modular, and the evaluation is highly reproducible.

## 2. Remaining Experimental Weaknesses and Limitations

### Weakness 2.1: Outdated and Toy Benchmark Suite
The paper's evaluation remains restricted to a 4-task vision suite: MNIST, FashionMNIST, CIFAR-10, and SVHN.
- **Grayscale / Toy Tasks:** MNIST and FashionMNIST are low-resolution (28x28) grayscale datasets that are extremely easy to classify and have been considered "solved" or "toy" tasks for over a decade. They are not representative of modern, challenging multi-task model merging scenarios.
- **Omission of Challenging Tasks:** Standard sparse model merging literature (such as TIES-Merging and DARE) typically evaluates on an 8-task vision suite including much more challenging tasks like **Stanford Cars**, **DTD** (Textures), **EuroSAT**, **GTSRB**, **RESISC45**, and **SUN397**. 
- **Workspace Availability:** The checkpoints and classification heads for the full 8-task suite are already downloaded and fully available in the `checkpoints/ViT-B-32/` workspace folder (e.g., `head_Cars.pt`, `head_DTD.pt`, `head_EuroSAT.pt`).
- **Severe Interference on SVHN:** On the most challenging dataset of the 4 evaluated tasks, SVHN (which has a large domain shift), standard un-rescaled STA (68.70%) is beaten by TIES-Merging (73.97%) by **5.27%** and DARE (78.71%) by **10.01%** at equivalent sparsity. This reveals that task interference is highly active. It is highly likely that on harder datasets (Cars, DTD, SUN397) where task interference is much more severe, STA's performance would drop catastrophically without sign consensus. The omission of these harder datasets is a significant limitation.

### Weakness 2.2: Omission of DARE-TIES Baseline
The paper's DARE baseline is implemented as delta-dropout and rescaling followed by direct linear addition (which is DARE-TA).
- In the original DARE paper (Yu et al., 2024), DARE's delta-dropout is combined with TIES-Merging (DARE-TIES) to achieve peak state-of-the-art performance. DARE-TIES incorporates TIES-style sign consensus and disjoint merging.
- By evaluating only DARE-TA (without sign consensus), the authors omit the stronger version of the DARE baseline. This represents an incomplete comparison against the state-of-the-art.

### Weakness 2.3: Evaluation Sample Size (Statistical Power)
The evaluation is conducted on a 16-batch validation split containing 2,048 samples per dataset.
- While this subset is helpful for reducing the computational footprint of executing over forty hyperparameter sweeps, it introduces statistical noise.
- For datasets like CIFAR-10 (10,000 test samples) and SVHN (26,032 test samples), a subset of 2,048 samples is too small to draw high-confidence conclusions when the reported performance differences are extremely marginal (e.g., the 0.37% average accuracy difference between Tuned STA and Tuned TIES). Evaluating on the full test sets would provide the necessary statistical power.
