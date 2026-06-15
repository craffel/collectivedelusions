# 4. Experiment Check

## Critique of the Experimental Setup and Datasets
The experimental setup uses a standard and representative backbone, **OpenCLIP ViT-B-32**. The selection of downstream datasets (**MNIST, SVHN, CIFAR10**) is also standard for model merging literature. 

However, the scale and depth of the evaluation are extremely limited:
- **Sample Size:** Evaluating on a subset of only 1,000 samples per dataset is acceptable for quick verification, but for a conference submission, evaluation should be performed on the full standard test/validation sets to ensure statistical significance.
- **Hardware/Environment Constraints:** The paper mentions evaluating on CPU nodes due to cluster compatibility issues. While this is understandable, evaluating high-dimensional models on CPU nodes usually prevents scaling to larger datasets or more complex models, which explains why the authors restricted themselves to tiny datasets (MNIST, SVHN) rather than larger-scale benchmarks (e.g., ImageNet, Stanford Cars, RESISC45) which are standard in CLIP merging literature.

---

## Missing Baselines
The empirical evaluation misses critical baselines that are standard in this domain:
1. **MagMax (ECCV 2024):** As detailed in `2_novelty_check.md`, MagMax is the direct baseline for winner-take-all magnitude selection in model merging. A comparison is absolutely necessary to justify WTA-Sign's design choices.
2. **DARE (Drop and Rescale):** Although DARE is discussed in the related work as a key baseline, it is completely absent from the empirical tables.
3. **AdaMerging / SyMerge:** These adaptive methods are mentioned in the introduction and related work, but not compared to in the experiments. While they require test-time optimization, comparing against them would provide a necessary upper bound on how much training-free methods can close the gap.

---

## Evaluation: Do the Results Support the Claims?
**No. The empirical results completely fail to support the core claims of the paper.**

The authors claim that "WTA-Sign consistently outperforms both Task Arithmetic and TIES-Merging across scaling coefficient sweeps" and "successfully retains 12.70% on MNIST, 18.65% on SVHN, and 11.23% on CIFAR10, fully preserving the generalist capabilities of the shared pre-trained backbone."

However, a closer inspection of Table 1 reveals that these claims are highly misleading:
1. **Defaults to Zero-Shot Base:** The performance of WTA-Sign at $\lambda \in \{0.1, 0.2, 0.3\}$ is **exactly identical** to the Pretrained (Zero-shot Base) model:
   - MNIST: 12.70% (Base: 12.70%)
   - SVHN: 18.65% (Base: 18.65%)
   - CIFAR10: 11.23% (Base: 11.23%)
   - Avg Accuracy: 14.19% (Base: 14.19%)
2. **Failure to Integrate Expert Knowledge:** The goal of model merging is to integrate the task-specific capabilities learned by the experts into the base model. Here, because the experts are broken (~8-16% accuracy, which is worse than the base model and worse than random guessing), any actual merging of knowledge *degrades* the base model.
3. **Illusory "Superiority":** WTA-Sign is "superior" to Task Arithmetic and TIES-Merging only because its aggressive winner-take-all masking and small scaling coefficients ensure that the task vectors have **zero impact** on the base model weights. At $\lambda=0.4$ and $\lambda=0.5$, as the scaling increases and the method is forced to actually apply the merged vectors, WTA-Sign's performance also degrades (dropping to 12.92% and 11.07% respectively).
4. **No Practical Utility:** In a real-world scenario, a model merging algorithm that achieves 14.19% average accuracy on MNIST, SVHN, and CIFAR10 (which represents random-guessing performance) is completely useless. A successful merge should achieve >90% average accuracy. The paper's empirical results demonstrate only that WTA-Sign is effective at doing nothing and ignoring the provided task vectors, which is a trivial achievement.

Thus, the claim that WTA-Sign "completely avoids the interference-driven performance collapse of Task Arithmetic... maintaining a top average accuracy of 14.19%" is an illusion. The method is simply preventing the corrupted expert vectors from affecting the base model, which could be achieved more easily by setting $\lambda=0$ (i.e., not merging at all).
