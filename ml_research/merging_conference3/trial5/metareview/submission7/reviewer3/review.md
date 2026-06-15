# Comprehensive Peer Review

## Paper Title
Pruned Gradient Merging (PG-Merge): Deconstructing Complexity in Test-Time Model Fusion

---

## Ratings

- **Soundness:** Fair
- **Presentation:** Good
- **Significance:** Poor
- **Originality:** Fair
- **Overall Recommendation:** 2: Reject

---

## 1. Summary of the Paper
The paper proposes **Pruned Gradient Merging (PG-Merge)**, a training-free and non-parametric method to address the "Overfitting-Optimizer Paradox" in active test-time model merging. During model merging, adjusting layer-wise merging coefficients using prediction entropy on small, unlabeled test-time adaptation (TTA) calibration streams often leads to transductive overfitting and performance degradation. 

PG-Merge addresses this by applying a dynamic sparse gradient mask. At each step, the gradients of the test-time adaptation loss with respect to the merging coefficients are computed, their absolute magnitudes are sorted, and only the top-$p\%$ (e.g., $5\%$) most sensitive coordinates are allowed to update, while the remaining $(100-p)\%$ are kept frozen via a post-update parameter projection. The authors evaluate their method on a compact Vision Transformer (`vit_tiny`) across four vision datasets (MNIST, FashionMNIST, CIFAR-10, and SVHN), claiming that PG-Merge substantially outperforms unconstrained AdaMerging and matches or exceeds complex state-of-the-art regularizers (such as RegCalMerge) without adding hyperparameter bloat or computational overhead.

---

## 2. Strengths
- **Conceptual Clarity and Simplicity:** The paper is guided by the philosophy of Occam's razor, rightly arguing that the rapid escalation of complexity in model-merging regularizers (e.g., polynomial trajectories, quantum wave analogies) is often unnecessary. Limiting optimization degrees of freedom is a highly logical way to combat overfitting.
- **Clear Mathematical Formulation:** The steps for PG-Merge, including percentile thresholding and the post-update parameter projection to prevent momentum leakage, are mathematically precise and easy to follow.
- **Optimization Trajectory Analysis:** Figure 3 provides an excellent visualization of the "Overfitting-Optimizer Paradox" in action, illustrating how unconstrained AdaMerging successfully minimizes local prediction entropy while degrading joint test accuracy.
- **Computational Leanliness:** The method operates directly on standard backpropagation gradients and requires zero auxiliary parameter tuning or complex spatial regularization losses, making it very easy to implement.

---

## 3. Major Weaknesses and Technical Flaws

Despite its conceptual appeal, this submission suffers from several fundamental methodological flaws, severe empirical limitations, and statistical inconsistencies that render its claims unsubstantiated.

### A. Statistically Marginal and Inconsistent Gains over Static Baseline
The paper's primary claim is that PG-Merge provides a robust and superior solution for test-time model fusion. However, a close look at the empirical results in Table 1 reveals that the performance gains are practically negligible and highly inconsistent:
1. **Marginal Improvement:** The joint mean accuracy of PG-Merge ($p=0.05$) is **$62.70\%$**, which represents an extremely tiny improvement of **only $0.54$ percentage points** over the completely static, zero-overhead **Uniform Merging baseline ($62.16\%$)**.
2. **Inconsistent Performance:** PG-Merge does *not* consistently outperform Uniform Merging across the tasks. In fact, on two of the four datasets, Uniform Merging is actually superior:
   - On **MNIST**, Uniform Merging achieves **$65.04\%$**, which is **$1.76\%$ higher** than PG-Merge ($63.28\%$).
   - On **SVHN**, Uniform Merging achieves **$33.20\%$**, which is **$1.17\%$ higher** than PG-Merge ($32.03\%$).
3. **No Practical Justification:** Since Uniform Merging requires **zero optimization steps, zero calibration data, zero hyperparameter tuning, and zero computational overhead**, there is no practical justification for adopting a test-time adaptation pipeline (even a simple one) to obtain an inconsistent, marginal average gain of $0.54\%$.

### B. Complete Absence of Statistical Rigor and Error Bars
The authors report single-point accuracy values without any standard deviations, confidence intervals, or error bars, and there is no mention of testing over multiple random seeds. Given the extremely small sample sizes used in the evaluation, this is a critical flaw:
- The calibration set consists of only **64 images** (16 per task).
- The test set consists of only **512 images** per task.

For a test set of size $n = 512$ and an accuracy of $62\%$, the standard error of a proportion is approximately **$2.15\%$**. A $95\%$ confidence interval yields a margin of error of **$\pm 4.2\%$**. 
The difference between PG-Merge and Uniform Merging ($0.54\%$) is **far smaller than the statistical margin of error**. In fact, a $0.54\%$ difference on a 512-image test set corresponds to only **2.7 images** being classified differently. Without averaging over multiple random calibration/test splits and reporting standard deviations, the reported SOTA performance is highly likely to be a result of random noise rather than a systematic algorithmic advantage.

### C. Major Discrepancy in Optimization Methodology (Adam vs. SGD)
In **Appendix A**, the authors identify a serious mathematical issue with applying the post-update parameter projection (Equation 13) under adaptive optimizers like Adam: momentum states ($\mathbf{m}$ and $\mathbf{v}$) for frozen parameters continue to decay toward zero under zero gradient masks, causing artificial dampening when these parameters are re-selected. 
To address this, they state:
> *"To address this nuance... we advocate for pairing PG-Merge with standard Stochastic Gradient Descent (SGD) without momentum."*

**The Discrepancy:** Despite this strong advocacy, **all** of the primary results in Table 1, Table 2, and the figures are generated using the **Adam optimizer** (Section 4.1). No empirical results for PG-Merge paired with SGD are presented anywhere in the paper. This raises critical concerns:
- If SGD is the mathematically "self-consistent" and "leakage-free" optimizer for PG-Merge, why is it completely omitted from the experimental evaluation?
- Does PG-Merge actually work with SGD, or does it fail due to the lack of adaptive learning rates?
- If PG-Merge with Adam still outperforms the baselines despite momentum decay, is the momentum state mismatch actually an issue, or is the analysis in Appendix A purely speculative? 

### D. Outdated, Toy-Scale Evaluation
Evaluating on `vit_tiny` (approx 5.7M parameters) on MNIST ($28 \times 28$ grayscale), FashionMNIST, CIFAR-10, and SVHN ($32 \times 32$) is extremely outdated. Modern model merging research focuses on large foundation networks (e.g., LLaMA-2-7B, Mistral-7B, CLIP-ViT-L) on complex multi-task, high-dimensional datasets. It is highly doubtful that findings on a toy ViT on grayscale digits scale to modern, high-parameter settings where parameter conflicts, gradient distributions, and optimization dynamics are completely different.

### E. Unresolved Semantic Conflict in Classification Heads
The paper merges expert models trained on four completely distinct classification tasks. Since the models are fine-tuned from the same shared pre-trained base model, they must share the same classification head structure (e.g., a 10-class linear head). 
When merging these models in parameter space, the weights of the classification heads are also directly merged. This represents a massive semantic conflict: the same logit index (e.g., index 0) must represent a '0' for MNIST, a 'T-shirt' for FashionMNIST, and an 'airplane' for CIFAR-10. 
How is this handled? If they kept the heads separate (i.e., multi-head), then they did not merge the entire model, and the heads were not updated. If they merged a single head, the severe performance collapse ($78.08\%$ ceiling to $62.16\%$ uniform) is heavily driven by this semantic head conflict rather than "layer-wise task conflicts." The paper is entirely silent on this crucial methodological detail.

### F. Poor Baseline Calibration (PolyMerge Trajectory Collapse)
PolyMerge's joint performance is reported as a catastrophic **$46.97\%$**, with MNIST collapsing to near-random ($13.48\%$). PolyMerge restricts coefficients to a quadratic polynomial trajectory over depth. Since Uniform Merging (which performs at $62.16\%$) is a special case of PolyMerge where the polynomial is a constant 0-degree polynomial ($\alpha_{k, l} = 0.3$), PolyMerge should easily be able to find a solution that matches or exceeds Uniform Merging if properly optimized. The catastrophic collapse suggests that the PolyMerge baseline was poorly optimized, had an inappropriate learning rate, or was severely misconfigured, making the comparison unfair and highly biased.

---

## 4. Questions and Actionable Feedback for the Authors

To make this paper suitable for publication, the authors must rigorously address the following points:

1. **Perform Statistical Significance Testing:** Re-run all experiments over at least 5 different random seeds (varying the calibration subsets and evaluation subsets) and report the mean and standard deviation for all methods in Table 1. If the performance gap between PG-Merge and Uniform Merging remains within the margin of error, the claim of PG-Merge's superiority must be toned down or re-evaluated.
2. **Present SGD Results:** Since you strongly advocate for pairing PG-Merge with standard SGD in Appendix A, you must include a column/row in Table 1 showing the performance of PG-Merge using standard SGD (both with and without momentum) and discuss whether the momentum mismatch actually affects performance.
3. **Clarify classification head treatment:** Provide a clear explanation of how the classification heads are handled. Are they merged directly? If so, how do you handle the conflicting semantic label spaces? If they are kept separate, please clarify which parameters are actually optimized and how the multi-head structure is implemented.
4. **Debug and Re-optimize PolyMerge:** Investigate why PolyMerge collapses to $46.97\%$. Ensure that PolyMerge is optimized with an appropriate learning rate and is given the opportunity to recover the uniform baseline initialization.
5. **Ablate Learning Rate vs. Sparsity:** Is the learning rate of $10^{-3}$ optimal for $p=0.05$? Intuitively, highly sparse updates (updating only 3 out of 56 coefficients) require a much larger learning rate than dense updates ($p=1.0$) to make meaningful progress in 100 steps. Please provide an ablation of the learning rate across different values of $p$.
6. **Scale up the Experiments:** Evaluate PG-Merge on a modern model merging benchmark (e.g., merging CLIP-ViT-B or LLaMA-7B models on standard downstream task suites) to prove that the findings generalize beyond toy, low-resolution vision datasets.
