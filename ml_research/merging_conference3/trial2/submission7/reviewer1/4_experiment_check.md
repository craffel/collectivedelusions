# 4. Experimental Evaluation and Empirical Check

This evaluation focuses on the experimental setup, choice of datasets, baseline comparisons, and whether the empirical findings actually support the paper's core claims.

## Rating: Fair

While the experiments are structured under a realistic sequential streaming test-time adaptation protocol, they suffer from a severe lack of scale, rely on inappropriate low-capacity models, and show marginal, non-functional absolute performance.

---

## 1. Scale and Realism of the Evaluation Benchmark

### A. Toy-Scale Datasets and Models
The paper evaluates model merging on a suite consisting of **MNIST, FashionMNIST, CIFAR-10, and SVHN**, using a **ResNet-18 backbone** where only `layer4` and the heads are merged. 

This setup is highly unrepresentative of modern model merging research and practice:
* **Foundation Models are the Industry Standard:** Model merging is almost exclusively applied to massive, overparameterized foundation models (e.g., CLIP, ViT, LLaMA, Mistral) because their high representation capacity allows multiple task vectors to be blended without immediate feature collapse. 
* **Low-Capacity Models are Prone to Interference:** A compact convolutional network like ResNet-18 has very little capacity in its deeper layers. Forcing it to merge four highly heterogeneous tasks (ranging from grayscale digits to natural color images) inevitably causes massive destructive interference.
* **Toy Datasets:** MNIST and FashionMNIST are toy datasets. Modern papers evaluate on complex vision-language benchmarks (e.g., 8 datasets including SUN397, EuroSAT, DTD, Resisc45) or NLP instruction tuning suites. 

While the authors provide a "Concrete Engineering Roadmap for Scaling to Foundation Models" in Appendix F, they **do not run any actual experiments** on foundation models or large-scale transformers. This severely limits the scientific value and generalizability of their empirical findings.

### B. Inappropriate Model Choice Cripples Baselines
The authors observe in Section 4.3.1 that TIES-Merging performs worse than simple Task Arithmetic under their ResNet-18 setup, and explain that "pruning 80% of parameter updates... can be slightly destructive... unlike in massive, overparameterized networks." 

This explanation is correct, but it highlights a fundamental flaw in the experimental design: **the authors chose an evaluation model that is fundamentally incompatible with the baselines they are comparing against.** By using a compact model like ResNet-18, they cripple methods like TIES-Merging (which rely on overparameterization and sparsity) and present a distorted, unrepresentative baseline comparison.

---

## 2. Inadequacy of the Empirical Results

### A. Non-Functional Absolute Accuracy
The absolute accuracy levels of the merged models across all methods are catastrophically low, making them completely non-functional in practice:
* On **MNIST** (where a single ResNet-18 easily achieves >99%), the merged model achieves only **20.00%** (ThermoMerge). This is barely above the 10% random guessing baseline.
* On **SVHN** (where ResNet-18 gets >95%), the merged model achieves **30.60%** (ThermoMerge).
* The overall average is only **29.05%**.

While ThermoMerge is technically "SOTA" because it beats Task Arithmetic (27.25%) by 1.8%, **an average accuracy of 29% on simple tasks indicates near-total representation collapse.** Claiming this as an "outstanding multi-task average accuracy" is a massive overstatement. A 1.8% improvement on a non-functional model does not represent a significant or meaningful contribution.

### B. Inconsistent Task Performance
Even if we accept the 1.8% average improvement, the performance is highly inconsistent across tasks:
* On **MNIST**, ThermoMerge (20.00%) performs **worse** than static Task Arithmetic (21.40%).
* On **FashionMNIST**, ThermoMerge (32.60%) performs **worse** than static Task Arithmetic (35.40%).

ThermoMerge only outperforms Task Arithmetic on the color datasets (CIFAR-10 and SVHN). However, even on these tasks, the absolute accuracy (33% and 30% respectively) is extremely poor. The fact that the proposed thermodynamic framework degrades performance on simpler, grayscale tasks compared to simple static averaging undermines the claim that it is a robust, "self-crystallizing" multi-task generalist.

---

## 3. The "Overfitting-Optimizer Paradox" as an Artifact of Bad Tuning
The authors attribute the poor performance and representation collapse of AdaMerging (26.10% average, 16.20% on MNIST) to a newly coined "Overfitting-Optimizer Paradox." 

However, in unsupervised test-time adaptation, it is a well-known phenomenon that unregularized objectives (like entropy minimization) are highly sensitive to optimization steps and learning rates. If run with a learning rate that is too high or for too many steps, the model's representations will drift and collapse (a well-known issue in methods like Tent). 

The authors optimized all TTA methods using a fixed, uniform learning rate of $1\times10^{-3}$ and fixed steps. It is highly likely that the "collapse" of AdaMerging is simply an artifact of insufficient hyperparameter tuning for the baselines under this specific, low-capacity ResNet-18 setting, rather than a fundamental theoretical paradox that requires a thermodynamic resolution.
