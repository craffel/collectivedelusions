# 4. Experiment Check

## Baseline Comparisons and Reframing Analysis
In the revised manuscript, the authors have added more baseline comparisons and provided a scientifically honest and transparent discussion:
* They explicitly acknowledge that standard, unconstrained dynamic baselines (the Linear Router at **77.10%** and QWS-Merge at **77.05%**) achieve higher peak performance (+3.30% absolute average accuracy) compared to ChaosMerge (**73.80%**).
* They re-frame ChaosMerge as a highly parameter-efficient alternative (exactly 384 parameters, which is a $30\times$ smaller parameter footprint than the 10,808-parameter Linear Router) that provides spatial regularization to prevent transductive overfitting on small calibration splits.

While this honest reframing is highly commendable, the quantitative results still highlight several significant performance trade-offs:
1. **The Overfitting-Optimizer Paradox is Unlikely here:** A standard Linear Router with 10k parameters is still incredibly compact relative to a 5.7M-parameter model. Since both methods are optimized on the same 64-sample calibration set, and the Linear Router achieves **77.10%** while ChaosMerge achieves **73.80%**, the Linear Router clearly does *not* suffer from a debilitating overfitting paradox; it simply performs better due to its greater expressiveness. Restricting the router to 384 parameters at the cost of a -3.3% accuracy drop is a highly unfavorable trade-off.
2. **Superiority of Task-Conditional Static Baseline:** The authors introduce a Task-Specific OFS-Tune (Supervised Static, Task-Conditional) baseline that achieves an outstanding average accuracy of **82.90%** (an absolute increase of **+9.10%** over ChaosMerge's **73.80%**). This baseline optimizes a separate set of static layer-wise coefficients of shape $(14, 4)$ for each task, requiring only $14 \times 4 = 56$ parameters per task, and $224$ parameters total across all tasks. Under any scenario where task divisions are clear, this simple task-conditional static baseline is vastly superior in both accuracy and parameter footprint (224 vs. 384 parameters), requiring no complex feature projection, unit spheres, Coupled Map Lattices, or numerical stabilizer clipping.

---

## Critical Empirical Check: Resolving the Map Ablation Paradox
In the revised manuscript, the authors expanded Table 2 ("Map Ablation") to include both 10-step fast training and 50-step converged training, as well as three completely non-chaotic baselines (Identity, Sigmoid Gated, and Tanh Gated Map), and their newly proposed **Annealed Chaos-to-Order Merging** framework.
The results reveal a fascinating empirical transition:
* At full convergence (50 steps), pure non-chaotic baselines indeed outperform the pure chaotic **Logistic Map** (e.g., Tanh Gated achieves **75.45%** vs. Logistic Map at **72.90%**).
* This confirmed that while active chaos is exceptionally beneficial for exploration early in training (where the Logistic Map leads), stable and robust representational convergence requires damping the chaotic trajectories.
* **The Annealed Breakthrough:** To fully capitalize on both exploration and exploitation, the authors introduced **Annealed Chaos-to-Order Merging**. This dynamically interpolates between the chaotic Logistic Map (early training) and the contractive Tanh Gated Map (late training).
* **Outstanding Performance:** This hybrid model achieves an outstanding classification accuracy of **78.12%**! This is a major improvement, outperforming both pure G-CML (72.90%) and pure Tanh Gated (75.45%), and even outperforming over-parameterized dynamic routers like the Linear Router (77.10%) and QWS-Merge (77.05%).

This empirical triumph completely resolves the paradox, demonstrating that the chaotic map acts as an indispensable, high-utility global exploration prior early in optimization.

---

## Toy-Scale Evaluation
Despite the added discussions and analysis, the scale of the empirical evaluation remains highly restricted:
* **Backbone Model:** The authors use `vit_tiny_patch16_224` (5.7M parameters). In modern machine learning literature, model merging is typically evaluated on much larger models (e.g., ViT-Base, ViT-Large, ResNet-50, or modern LLMs like LLaMA-3 or Mistral).
* **Datasets:** The benchmark consists of MNIST, FashionMNIST, CIFAR-10, and SVHN. These are extremely small, classic computer vision datasets, and two of them (MNIST and FashionMNIST) are toy-scale grayscale datasets.
* **Low-Data Regimes:** Both fine-tuning (2,000 samples) and calibration (64 samples) are extremely small. While evaluating low-data regimes is interesting, the lack of standard-scale evaluations (such as ImageNet-subsets, GLUE benchmarks, or reasoning benchmarks) makes it impossible to verify if the method generalizes to modern real-world tasks.
