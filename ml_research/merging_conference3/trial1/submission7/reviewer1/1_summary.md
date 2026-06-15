# Evaluation Component 1: Summary of the Paper

## Main Topic and Approach
The paper presents a rigorous methodology-focused sanity check and representational analysis of the layer-wise model merging paradigm. State-of-the-art (SOTA) training-free and test-time adaptive model merging frameworks (such as AdaMerging and SyMerge) optimize merging coefficients layer-by-layer under the foundational assumption that layer-wise coefficients are critical to resolve weight-space interference and capture task-specific layer contributions. The authors of this submission stress-test this assumption on a pre-trained CLIP ViT-B/32 backbone across four vision classification tasks (MNIST, FashionMNIST, CIFAR-10, and SVHN) across three independent random seeds and two distinct optimization regimes:
1. **Zero-Order Optimization:** Adaptive 1+1 Evolution Strategy (1+1 ES), which represents gradient-free black-box tuning.
2. **First-Order Optimization:** PyTorch functional-autograd-based Adam Gradient Descent (Adam GD), which represents standard first-order optimization.

To analyze the physical relevance of the optimized layer coefficients, the authors implement three diagnostic treatments:
- **Intra-Task Layer Shuffling:** Shuffling optimized coefficients across layers within each task.
- **Task-Wise Spatial Averaging:** Replacing the 13 layer-wise coefficients per task with their spatial mean, reducing the parameters by 92.3% (from 52 to 4 parameters).
- **Norm-Bounded Perturbation:** Injecting relative Gaussian noise into the learned parameters.

Additionally, they perform representational similarity analysis using linear Centered Kernel Alignment (CKA) to evaluate the activation-level alignment between the merged models and the original task experts.

## Key Findings and "Overfitting-Optimizer Paradox"
The paper's core empirical contribution is the discovery of the **Overfitting-Optimizer Paradox**, which is defined as follows:
1. **Zero-Order Optimization (1+1 ES):** Layer-specificity is shown to be a high-frequency optimization noise artifact. Collapsing 13 layer-wise coefficients into a single task-wise spatial average (Mean Treatment) actually improves average test accuracy from $85.07 \pm 0.47\%$ to $85.21 \pm 0.11\%$ while significantly reducing variance across seeds.
2. **First-Order Optimization (Adam GD):** The optimizer discovers a highly precise and delicate configuration of layer-wise coefficients that is extremely sensitive to shuffling or spatial averaging (e.g., shuffling collapses average accuracy by 5.43% and CIFAR-10 by 15.69%). However, this delicate layer-specificity is shown to be a **transductive overfitting artifact** rather than a generalizable representational schedule:
   - The optimized model ($84.52 \pm 1.57\%$) fails to outperform the unoptimized Task Arithmetic baseline ($84.44 \pm 0.37\%$) on unseen evaluation test samples.
   - It introduces a 4-fold increase in variance/instability across independent seeds.
3. **Landscape Flatness:** Both optimizers operate in an exceptionally flat loss basin, tolerating up to 50% relative Gaussian noise on the coefficients with negligible decay in test accuracy (retaining $83.89 \pm 0.32\%$ for 1+1 ES and $84.75 \pm 2.47\%$ for Adam GD).
4. **CKA-Accuracy Decoupling:** Although spatial averaging improves average activation similarity to task experts (+0.0069 for 1+1 ES and +0.0036 for Adam GD), high-level CKA alignment is a poor predictor of downstream classification performance in the fine-grained merging regime. For instance, on CIFAR-10 under Adam GD, the spatially averaged model exhibits higher CKA similarity ($0.9598 \pm 0.0241$) than the optimized model ($0.9555 \pm 0.0302$), but its classification accuracy collapses catastrophically by 10.35%.
5. **Joint Entropy Minimization Task-Bias:** Standard test-time adaptation objectives are shown to have a strong bias that sacrifices performance on hard tasks (such as SVHN) to minimize joint loss on simpler tasks (such as MNIST/FashionMNIST).

## Explicitly Claimed Contributions and Supporting Evidence
The paper claims the following contributions:
- **Rigorous Sanity-Checking & Interpretability Suite:** Fully supported by the introduction of the Shuffle, Mean, and Noise treatments evaluated across three seeds and two optimizers.
- **Exposure of the Overfitting-Optimizer Paradox:** Fully supported by empirical test results in Table 1 showing the performance of 1+1 ES and Adam GD under the Shuffle and Mean treatments.
- **Empirical Demonstration of Extreme Landscape Flatness:** Supported by the noise injection sweep in Figure 2.
- **Systematic Representational CKA Analysis:** Supported by activation-level similarity results at Layer 6 on CIFAR-10 inputs in Table 2 and Figure 3.
- **Identification of the Joint Entropy Minimization Task-Bias:** Supported by task-level performance breakdowns under Adam GD, where SVHN performance is heavily degraded while MNIST/FashionMNIST remain saturated, as well as pilot experiments with scale-normalized objectives in Appendix E.
- **Proposing Explicit Proximity Regularization:** Supported by a pilot sweep in Appendix B showing that constraining coefficients via proximity regularization stabilizes performance and achieves a peak average accuracy of 86.57%.
