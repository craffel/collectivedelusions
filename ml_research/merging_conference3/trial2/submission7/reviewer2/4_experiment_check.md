# Evaluation Phase 4: Experimental Evaluation and Baseline Analysis

## Critical Evaluation of the Experimental Setup and Datasets
The experimental evaluation is performed on a four-dataset multi-task classification suite (MNIST, FashionMNIST, CIFAR-10, SVHN). While this represents a highly heterogeneous mix of grayscale and color domains, the absolute performance numbers expose severe flaws in the underlying experimental setup.

### 1. Extremely Low Absolute Performance (Barely Above Random Guessing)
A major red flag in Table 1 is the extremely low absolute classification accuracies achieved by **all** methods under the ResNet-18 backbone:
- **MNIST:** Task Arithmetic achieves **21.40%** and ThermoMerge achieves **20.00%**. 
- **FashionMNIST:** Task Arithmetic achieves **35.40%** and ThermoMerge achieves **32.60%**.
- **CIFAR-10:** Task Arithmetic achieves **29.80%** and ThermoMerge achieves **33.00%**.
- **SVHN:** Task Arithmetic achieves **22.40%** and ThermoMerge achieves **30.60%**.
- **Multi-task Average:** ThermoMerge peaks at a mere **29.05%**.

MNIST, FashionMNIST, CIFAR-10, and SVHN are 10-class datasets where random guessing yields **10.00%** accuracy. For a standard, pre-trained ResNet-18, fine-tuned individual experts should easily achieve:
- MNIST: >99%
- FashionMNIST: >92%
- CIFAR-10: >80%
- SVHN: >90%

The fact that the merged models are hovering around **20% to 33%** accuracy indicates that the merged representation is almost completely broken and severely degraded. While the authors claim that ThermoMerge "stabilizes" adaptation and performs "outstanding thermodynamic fusion," a model that gets **20% accuracy on MNIST** (dropping nearly 80% from standard expert performance) is a catastrophic failure. The "superior average" of 29.05% is simply a slightly less catastrophic collapse compared to Task Arithmetic (27.25%), rather than a viable multi-task model.

### 2. Fatal Backbone/Resolution Mismatch
Why is the absolute performance so catastrophically poor?
The authors state that they utilize an ImageNet pre-trained ResNet-18 backbone. However, they resize the target datasets to **$32 \times 32$ pixels** while freezing the early convolutional blocks (`conv1`, `bn1`, `layer1`, `layer2`, and `layer3`).
- **The Culprit:** An ImageNet pre-trained ResNet-18 is designed and trained on $224 \times 224$ high-resolution natural images. The early layers extract features optimized for this resolution. Passing $32 \times 32$ low-resolution images through these frozen early layers extracts highly distorted, low-quality features. 
- **Methodological Flaw:** Because early features are so heavily mismatched, the task experts are forced to learn specialized classification heads and deep features (in `layer4` only) on highly degraded inputs. When merged, these fragile deep features collide, causing near-complete representation collapse. This highly compromised, non-functional setup makes it impossible to draw meaningful scientific conclusions about the efficacy of ThermoMerge under realistic, properly configured model merging settings.

## Analysis of Baselines and Claims

### 1. The Overfitting-Optimizer Paradox Claim is Inconsistent
The authors claim that unregularized adaptive merging (AdaMerging) collapses to random chance on color datasets due to the Overfitting-Optimizer Paradox.
- **Inconsistency with ResNet-18 Results:** Looking at Table 1 under the pre-trained ResNet-18 backbone, AdaMerging achieves **28.40% on CIFAR-10** and **23.40% on SVHN**, while static Task Arithmetic gets **29.80% and 22.40%**. AdaMerging does NOT collapse to random chance under this backbone; it actually outperforms Task Arithmetic on SVHN. 
- **From-Scratch Failure:** The collapse of AdaMerging to random chance (9.60% on CIFAR-10 and 8.60% on SVHN) is only observed under the custom `SimpleCNN` backbone trained from scratch. But on this same SimpleCNN backbone, the state-of-the-art baseline **SyMerge actually outperforms ThermoMerge** on average (**31.20% vs 31.05%**) and on MNIST (**44.60% vs 40.20%**). When pre-trained mode connectivity is absent, the "superior" thermodynamic principles of ThermoMerge are actually beaten by a cold, zero-temperature teacher-student alignment baseline (SyMerge).

### 2. Missing Essential Baselines
The authors fail to compare their approach against prominent model merging baselines that directly address parameter interference, such as:
- **RegMean** (Jin et al., 2022) - a foundational regression-based merging approach.
- **DARE** (Yu et al., 2023) - which uses delta parameter pruning and rescaling.
- **BLUELine** - which scales task vectors to resolve sign conflicts.
- **Joint Multi-Task Learning (MTL) Upper Bound:** The paper completely omits the joint training upper bound, which would highlight the massive performance gap between merged models (29%) and a properly trained joint MTL model (which would easily exceed 85% average accuracy).

### 3. Insignificant Practical Margin vs. High Computational Overhead
The empirical gain of ThermoMerge over the best baseline (SyMerge) is a mere **+1.15%** (29.05% vs 27.90%), and only **+1.80%** over static Task Arithmetic (29.05% vs 27.25%).
- **Computational Overhead:** To achieve this tiny improvement, ThermoMerge requires running $K$ forward passes through all frozen expert models at *every step* of the 100-step test-time adaptation loop, leading to an $\mathcal{O}(K)$ scaling overhead. For larger-scale setups (e.g., $K=20$ or $50$ tasks), this latency and memory footprint are highly prohibitive. 
- **The Caching Limitation:** While the authors propose "expert prediction caching" as a mitigation, caching only works for static, pre-defined calibration sets. In a realistic, online test-time adaptation stream where test samples arrive on-the-fly and must be predicted in real-time, caching is impossible, rendering ThermoMerge highly impractical compared to zero-overhead static merging methods.
