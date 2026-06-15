# Experiment Check Report (`4_experiment_check.md`)

## 1. Critical Evaluation of the Experimental Setup & Datasets
The entire empirical evaluation of the paper is heavily constrained and suffers from severe limitations in dataset scale and task complexity:
- **Reliance on Split-MNIST Toy Benchmark:** The primary experiments are conducted exclusively on Split-MNIST subsets. Grayscale handwritten digits are extremely simple, linearly separable to a high degree, and possess highly overlapping representation distributions. This is a very weak sandbox for demonstrating the viability of dynamic model merging on modern deep learning systems.
- **Unrealistic Expert Training Subsets:** As noted in the soundness report, training expert networks on only 512 samples per task is highly non-standard. This artificial constraint makes it unclear whether the findings would hold if the experts were trained to convergence on full-scale, representative datasets.
- **Natural Image Evaluation is Insufficient:** While the authors attempt to address this in Section 4.4 and Appendix 4, their natural image experiments are extremely brief. They report only a minor improvement on a custom `NaturalCNN-4` (reaching $20.20\% \pm 1.71\%$ accuracy on CIFAR-10 + SVHN, which is essentially a failure for a binary domain task). Their ViT-B/16 simulation in Appendix 4 is even weaker, as they only report the SVD Collinearity Ratio and completely omit any classification accuracy results. This suggests that their method fails on natural images, and they are hiding this by omitting the accuracies.

## 2. Evaluation of Baselines and Results
The empirical results presented in Tables 1 and 2 and Figure 4 do not support the authors' claims of the superiority or practicality of their method. In fact, a critical analysis of the results reveals several devastating weaknesses:

### A. The Static Baseline (OFS-Tune) Consistently Outperforms the Proposed Router
On the convolutional architecture (TinyCNN-4), which is the only setting where the models achieve any reasonable performance, the proposed dynamic Layer-wise Router is consistently **outperformed** by the simple static baseline **OFS-Tune** across all three task-conflict suites in the primary data split:
- **Low-Conflict:** OFS-Tune scores **$82.85\% \pm 11.52\%$** vs. Layer-wise Router's **$78.70\% \pm 14.56\%$** ($+4.15\%$ delta).
- **High-Conflict:** OFS-Tune scores **$90.75\% \pm 1.58\%$** vs. Layer-wise Router's **$81.30\% \pm 9.69\%$** ($+9.45\%$ delta).
- **Cross-Domain:** OFS-Tune scores **$53.40\% \pm 7.16\%$** vs. Layer-wise Router's **$52.52\% \pm 5.95\%$** ($+0.88\%$ delta).

The authors explain this away as a "Parameter-Variance Constraint," showing in Figure 4 that scaling the calibration budget to 1,024 samples allows the Layer-wise Router to slightly exceed OFS-Tune ($54.50\%$ vs $53.83\%$). However, a minor $+0.67\%$ accuracy gain that requires **8 times more calibration data** is practically negligible and represents a highly inefficient trade-off. This demonstrates that a simple static global compromise is highly robust and superior to the authors' high-capacity, high-variance dynamic router.

### B. Severe Performance Collapse in Deep MLPs
On DeepMLP-12, the absolute accuracies of all merged models under Cross-Domain task conflict are abysmal:
- **Static Uniform:** $11.80\% \pm 0.81\%$
- **OFS-Tune:** $12.23\% \pm 0.68\%$
- **L1-Global Router:** $11.68\% \pm 0.56$
- **Layer-wise Router (Ours):** $16.15\% \pm 5.60\%$
- **Random Guessing Threshold:** $12.50\%$

A classification accuracy of $16.15\%$ on an 8-class digit task is functionally equivalent to random guessing. The model is completely broken. The authors themselves admit that "full-parameter linear interpolation of deep, fully connected layers under multi-task conflict is fundamentally a failed paradigm" due to coordinate misalignment and activation drift. If weight-space merging on deep MLPs is a failed paradigm, it raises serious questions about why the authors are presenting a 12-layer MLP evaluation as a core component of their study.

### C. The Devastating Oracle Performance Gap
There is a massive, unaddressed chasm between the merged models and the Oracle ceiling (which simply routes inputs directly to the specialized unmerged experts):
- On TinyCNN-4 Cross-Domain, the Oracle achieves **$99.30\% \pm 0.23\%$**, while the best merged model (OFS-Tune) achieves only **$53.40\% \pm 7.16\%$**—a drop of **nearly 46% in absolute accuracy**.
- On DeepMLP-12 Cross-Domain, the Oracle achieves **$98.23\% \pm 0.41\%$**, while the best merged model achieves only **$16.15\% \pm 5.60\%$**—a drop of **over 82% in absolute accuracy**.

This massive drop indicates that weight-space merging introduces catastrophic representational damage that destroys the functionality of the models. The authors claim this gap is an "intrinsic barrier of weight-space parameter interpolation." However, this barrier severely undermines the practical utility of the entire paper: if weight-space merging is so fundamentally flawed that it destroys half to four-fifths of a model's accuracy on a simple toy dataset like MNIST, then the entire line of research is practically non-viable.

### D. The Batch-Averaged Multi-Task Inference Paradox
The authors' own conceptual analysis in Section 3.5 completely invalidates the practical viability of their proposed system:
1. **Under Mixed Batches:** To avoid memory-bandwidth bottlenecks, the system must average routing coefficients over the batch. If the batch is heterogeneous, these batch-averaged coefficients converge to a uniform static distribution, losing all dynamic sample-specific routing benefits and collapsing performance (Mixed-Batch Collapse).
2. **Under Homogeneous Batches:** The only way to avoid this collapse is to construct homogeneous, task-specific batches. However, to do this, the system must already possess the task labels. If the system already has the task labels, it can simply route the batch directly to the specialized individual expert model (the Oracle), achieving $99.30\%$ accuracy with **zero** weight interpolation overhead and zero representational damage.

This logical paradox is a fatal conceptual flaw. It shows that the entire framework of batch-averaged dynamic model merging is either logically redundant or functionally useless. The paper offers no physical solution to this paradox, only speculative "future pathways" (like LoRA adapters).
