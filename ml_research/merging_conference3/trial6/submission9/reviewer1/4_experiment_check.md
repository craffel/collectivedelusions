# Experimental Evaluation and Claims Verification: CAM-Router

## Critical Evaluation of Experimental Setup
- **Toy Datasets & Compact Sandbox:** The experiments are restricted to a custom "14-layer compact Vision Transformer coordinate sandbox" evaluated on four toy image datasets (MNIST, FashionMNIST, CIFAR-10, SVHN). Modern model-merging literature typically evaluates on much larger, standard vision-language models (e.g., CLIP-ViT-B/16) or large language models (e.g., LLaMA-7B) on complex benchmark tasks.
- **Tiny Calibration Set:** The calibration dataset consists of only 800 samples total. Training a 0.15M parameter router on such a tiny dataset makes the model highly susceptible to overfitting, as shown in the regularization ablations.

## Baselines
The paper compares against relevant dynamic baselines (QWS-Merge, BSigmoid-Router, L3-Router). However, there is a notable absence of standard **static merging** baselines like Ties-Merging, Task Arithmetic, or DARE in the experimental table. Since static methods are the primary alternative to dynamic routing, their omission from Table 1 is a major gap.

## Discrepancies and Unsupported Claims

### 1. Severe Numerical Discrepancies between Abstract and Text/Tables
There is a massive and highly unprofessional discrepancy between the results reported in the Abstract and those reported in the rest of the paper:
- **Joint Mean Accuracy:** The Abstract claims **57.07%** Joint Mean Accuracy (representing a **+15.10%** improvement over Static Uniform and outperforming QWS-Merge by **+32.17%**). However, Table 1 and Section 4.2 report **53.07%** Joint Mean Accuracy (representing a **+11.10%** improvement and outperforming QWS-Merge by **+28.17%**).
- **Spatial Occlusion Robustness:** The Abstract claims a stable accuracy of **53.63%** under up to 80% patch masking. However, Table 3 and Section 4.3 report **50.57%** accuracy at 80% masking.
- **Batch Heterogeneity Resilience:** The Abstract claims a stable accuracy of **55.47%** at batch size $B=256$. However, Table 4 and Section 4.3 report **54.30%** accuracy at $B=256$.
These conflicting numbers undermine the scientific accuracy and integrity of the submission. It appears the Abstract was written using an outdated or fabricated run and was never updated to match the actual results in the paper.

### 2. Under-performance Relative to Individual Experts
Despite the added complexity of multi-head cross-attention, trainable task queries, and historical gating, CAM-Router's Joint Mean Accuracy (53.07%) remains extremely far below the performance of the Individual Experts (85.85%). A staggering **32.78% performance gap** remains. This indicates that the proposed method still suffers from severe parameter interference and fails to fully restore the expert capabilities.

### 3. Overfitting & Sensitivity to L2 Regularization
In Table 7 (Sweep 5), the authors evaluate different weight decay penalties. When the default weight decay ($\lambda_{wd} = 10^{-3}$) is applied, CAM-Router's performance drops sharply from **53.07%** (at $\lambda_{wd}=0.0$) to **47.40%**. This extreme sensitivity suggests that the unregularized CAM-Router is overfitting the tiny 800-sample calibration set, and its performance is highly fragile when standard regularization is applied.
