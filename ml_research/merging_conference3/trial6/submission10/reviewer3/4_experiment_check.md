# 4. Experiment Check

## Critical Evaluation of the Experimental Setup
The experimental setup has several key characteristics:
- **Backbone**: Vision Transformer (`vit_tiny_patch16_224`) with 5.7M parameters. This is a compact, standard backbone, which is appropriate for small-scale verification.
- **Data Regime**: Task experts are trained on only 1000 images per task for 2 epochs. This results in highly under-trained experts (e.g., MNIST expert achieves only 73.20%, and SVHN expert achieves 23.20%). While the authors frame this as simulating "resource-constrained, sub-optimal parameters," it represents an extremely low performance baseline. It is unclear if the findings hold for fully converged, high-quality models.
- **Calibration Split**: 16 samples per task (64 total). This is extremely low, simulating an aggressive low-data regime, which is appropriate for showcasing overfitting challenges.
- **Evaluation Split**: 250 samples per task (1000 total). This is a reasonable size to obtain stable test accuracy estimates.
- **Control**: Strict seed control ($\mathtt{seed=42}$) is maintained, which ensures fair comparisons.

## Evaluation of Datasets
The datasets used (MNIST, FashionMNIST, CIFAR-10, SVHN) represent a highly heterogeneous and high-conflict mixture:
- MNIST and FashionMNIST are grayscale.
- CIFAR-10 and SVHN are colored, natural images and street numbers.
- SVHN contains severe visual clutter and is notoriously difficult.
- Combining these highly distinct tasks creates a strong benchmark for evaluating task conflicts and representational collapse.

## Evaluation of Baselines
The paper compares its approach against a comprehensive and rigorous set of seven baselines:
- Specialist Expert (Upper Bound)
- Uniform Merge (Task Arithmetic)
- Linear Router (Classical, Unregularized)
- BL-Router (Softmax, Unregularized)
- BL-Router (Reg)
- BSigmoid-Router (Unregularized)
- BSigmoid-Router (Reg)
- QWS-Merge (SOTA wave-interference method)
The baseline selection is excellent and represents both simple static/dynamic baselines and recent SOTA methods.

## Do the Results Support the Claims?
**Absolutely not. The empirical results directly refute the paper's central claims.**

### Claim 1: "TCPR consistently prevents high-conflict task collapse and bridges the performance gap to specialist experts."
- **Refutation (Collapse Prevention)**: In Table 1, the unregularized BSigmoid-Router achieves **25.50%** Joint Mean accuracy. When the proposed TCPR is added (at its optimal scale $\beta=10^{-6}$), the accuracy is **25.20%**. On SVHN (the highest-conflict task), both unregularized BSigmoid-Router and TCPR-Param get exactly **10.40%** (which is barely above the 10.00% random-guessing baseline). Thus, the TCPR regularizer prevents absolutely nothing; any prevention of collapse is entirely due to the decoupled sigmoid architecture of the BSigmoid-Router.
- **Refutation (Bridging the Gap)**: The Specialist Expert achieves a Joint Mean of **62.40%**. The best performing TCPR variant gets **25.20%**. A massive **37.20% absolute gap** remains between the proposed method and the specialist. Claiming that this "bridges" the gap is highly exaggerated and scientifically misleading.

### Claim 2: "TCPR provides a robust, scale-invariant pathway for low-data model merging, surpassing previous complex wave-interference methods..."
- **Refutation (Scale Invariance)**: The hyperparameter sensitivity sweep in Section 4.5 and Figure 1 shows that TCPR is highly scale-sensitive, not scale-invariant. For $\beta \le 10^{-6}$, the regularizer is mathematically dead and behaves identically to the unregularized router. For $\beta \ge 1.0$, the performance severely collapses (dropping to 21.00% and 19.90%). The regularizer only "works" when it is scaled down so far that it is effectively inactive.
- **Refutation (Surpassing SOTA)**: While TCPR ($25.20\%$) does surpass QWS-Merge ($21.8\%$), this is solely because the underlying BSigmoid-Router surpasses QWS-Merge ($25.5\%$). Adding TCPR actually *degrades* the sigmoidal router's performance.

### Summary
The experimental results are highly valuable, but they support a completely different conclusion than the one claimed in the abstract and intro. The results show that:
1. Decoupling routing pathways using independent sigmoids (BSigmoid-Router) is a simple, elegant, and superior way to merge models.
2. The proposed TCPR regularizer is a complete failure that is either mathematically dead or actively harmful, because static pre-computed priors are fundamentally incompatible with dynamic sample-level routing.
The paper's narrative is in complete denial of its own empirical findings.
