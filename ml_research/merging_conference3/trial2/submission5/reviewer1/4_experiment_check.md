# 4. Experimental Evaluation Critique

## Experimental Setup and Datasets
The authors evaluate NETA using a CLIP ViT-B/32 backbone across four diverse classification datasets: **MNIST, FashionMNIST, CIFAR-10, and SVHN**. 
From a practical deployment standpoint, the choice of datasets is a **major limitation**:
- **Toy/Small-Scale Domains**: MNIST and FashionMNIST consist of 28x28 grayscale images, and CIFAR-10 and SVHN consist of 32x32 images. These are extremely small-scale, simplified toy datasets.
- **Foundation Model Overkill**: Using a CLIP ViT-B/32 model (pre-trained on 400M high-resolution image-text pairs) to solve 28x28 and 32x32 classification tasks is overkill. The representations required for these datasets are extremely simple compared to real-world, high-resolution visual tasks.
- **Lack of Scale**: In actual industrial or applied settings, practitioners merge models for complex, high-resolution tasks (e.g., ImageNet-1K, CUB-200, or modern vision-language benchmarks like COCO or VQA). The paper lacks evaluation on any large-scale, complex domain shift, leaving a major question mark over NETA's scalability and real-world utility.

The authors use a sub-sampling protocol of 1024 randomly selected test images per dataset to manage computational overhead. While this is a practical and acceptable compromise, running evaluations on the full test sets would provide a more robust statistical guarantee.

## Baseline Comparisons
The paper compares NETA to a robust set of zero-shot and test-time adaptation baselines:
- **Zero-shot**: Task Arithmetic (TA), TIES-Merging, and DARE.
- **Test-time adaptation (TTA)**: Task-Wise AdaMerging (4 parameters) and Layer-Wise AdaMerging (52 parameters).

These represent the correct and most relevant SOTA baselines in the current model merging literature. The omission of RegMean is properly justified, as RegMean requires validation covariance statistics, which violates the strict zero-shot, data-free paradigm.

## Do the Results Support the Claims?

The empirical results partially support the claims, but also highlight significant practical trade-offs:

1. **Successful Isotropic Regularization on Simple Tasks**:
   The claim that NETA prevents representation hijacking is supported by the zero-shot improvements on MNIST (96.29% vs 96.03% in TA) and FashionMNIST (82.75% vs 82.10% in TA). NETA successfully acts as an isotropic regularizer, giving these simpler tasks an equal footing.

2. **Catastrophic Performance Drop on the Hardest Task (SVHN)**:
   On SVHN, NETA's accuracy drops significantly from **80.14%** (Task Arithmetic) to **77.02%** (NETA), a drop of **-3.12%**. SVHN represents the most complex domain shift and the hardest dataset in the 4-task suite. In standard Task Arithmetic, SVHN achieves 80.14% because its high-magnitude update is allowed to dominate. By equalizing the norms, NETA curtails this dominance, which severely degrades performance on the most challenging task.
   From a practitioner's perspective, **this is a major drawback**. In many real-world applications, we care more about maintaining peak performance on the most difficult/critical task than achieving "fairness" by degrading the best task to boost simpler, already high-performing tasks.

3. **Lower Overall Average Performance than Baselines**:
   NETA's multi-task average accuracy (87.17%) is actually **lower** than standard Task Arithmetic (87.76%) and DARE (87.78%). While the authors characterize this as an honest peak performance vs. fairness trade-off, a practitioner would be hard-pressed to deploy a new method that reduces both overall average accuracy and peak performance on the hardest task compared to simple Task Arithmetic or DARE. 

4. **The Overfitting-Optimizer Paradox is Compelling but Nuanced**:
   The exposure of the Overfitting-Optimizer Paradox is highly convincing. The catastrophic drop of Task-Wise AdaMerging on FashionMNIST (-4.56%) clearly demonstrates that unsupervised joint prediction entropy minimization on small calibration sets is highly unstable under task-difficulty imbalances.
   However, **Layer-Wise AdaMerging (TTA)** achieves the highest average accuracy by far (**90.89%**), which is **+3.72%** higher than NETA (87.17%). While the authors critique Layer-Wise AdaMerging as overparameterized and transductive, in high-stakes practical deployments where a small unlabeled calibration set is available, a $+3.72\%$ absolute accuracy gain is a massive incentive that easily justifies the extra complexity of optimizing 52 parameters over 20 epochs.

5. **Clamping Boundary standard deviations**:
   The observation of exactly $0.00\%$ standard deviation for Task-Wise AdaMerging on FashionMNIST and Layer-Wise AdaMerging on MNIST across 3 independent seeds is a nice, rigorous finding that supports the claim that prediction entropy optimization can drive weights to extreme, clamped parameter boundaries.
