# Intermediate Evaluation: Experimental Check

## Evaluation of Experimental Setup
The experimental evaluation is designed to test weight-space model merging under highly conflicting task distributions. However, from a practical and industry-scale perspective, there are several critical weaknesses in the experimental setup, baseline comparisons, and execution:

### 1. Toy Backbone and Contrived Datasets
* **ViT-Tiny Backbone**: The authors evaluate their method on the ViT-Tiny model, which has only **5.7M parameters** and a hidden dimension of 192. This is an extremely small, toy model. Modern practitioner workloads typically involve high-capacity vision models (e.g., ViT-Base/Large with hundreds of millions of parameters) or large-scale language backbones.
* **Small Low-Resolution Datasets**: The datasets used are MNIST ($28\times28$ grayscale digits), FashionMNIST ($28\times28$ grayscale fashion items), CIFAR-10 ($32\times32$ low-resolution natural images), and SVHN ($32\times32$ low-resolution street digits). In 2026, evaluating model merging strictly on these low-resolution toy datasets does not reflect real-world, industry-scale settings.

### 2. Surprisingly Weak and Poorly Trained Experts
The independent task-specific expert performance ceilings reported in the paper are remarkably low:
* **CIFAR-10 Expert**: **54.00%**
* **SVHN Expert**: **65.20%**
* **MNIST Expert**: **98.10%**
* **FashionMNIST Expert**: **82.55%**

A standard ViT-Tiny model fine-tuned on CIFAR-10 should easily achieve **85--90% accuracy**, and on SVHN it should easily exceed **95% accuracy**. The extremely low accuracies (e.g., 54.00% on CIFAR-10) indicate that the expert models are severely under-optimized, likely due to the highly restricted training budget (fine-tuned for only 2 epochs). Utilizing poorly trained, weak experts as the starting point for model merging compromises the generalizability and scientific validity of the entire experimental analysis.

### 3. Absolute Performance and Practical Usability Gap
The paper claims that GSC-Merge achieves superior multi-task generalization. However, the absolute performance numbers tell a different story:
* **Task-Conditional Setting**: At $\gamma=0.3$, GSC-Merge achieves **42.13%** joint mean accuracy, and at $\gamma=0.5$ it achieves **43.88%**. Compared to the weak expert ceiling of **74.96%**, this represents a massive **31--33% absolute performance degradation**.
* **Truly Task-Agnostic Setting**: At $\gamma=0.3$, GSC-Merge achieves **19.08%** accuracy, and at $\gamma=0.5$ it achieves **20.61%**. This is barely above a random guessing baseline of 10% on these simple datasets.
* **The Practical Verdict**: From a practitioner's perspective, a model that degrades CIFAR-10 accuracy from 54% to 35.94% (task-conditional) or 18.29% (task-agnostic) is practically unusable. The absolute numbers demonstrate that while GSC-Merge performs better than worse-performing coordinate-wise baselines (like STA or TIES), the "generalization" achieved is still too poor to be useful in any real-world, applied domain.

### 4. Comparison to the Unconstrained Baseline
The main theoretical claim is that GSC-Merge resolves the "Overfitting-Optimizer Paradox" of unconstrained tuning. However, in the task-conditional setting:
* **Unconstrained OFS-Tune**: **$44.08 \pm 4.31\%$**
* **GSC-Merge ($\gamma=0.5$)**: **$43.88 \pm 4.07\%$**
* **GSC-Merge ($\gamma=0.3$)**: **$42.13 \pm 2.76\%$**

GSC-Merge does not actually outperform unconstrained tuning in terms of mean accuracy. At $\gamma=0.3$, GSC-Merge's accuracy is **1.95% absolute lower** than unconstrained tuning, and at $\gamma=0.5$ it is **0.20% lower**. The only benefit is a slight reduction in standard deviation (variance across splits). For a practitioner, trading off precious mean performance (which is already extremely low) to reduce split-sensitivity variance is rarely a compelling trade-off.
