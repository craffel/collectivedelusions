# Intermediate Evaluation: Soundness and Methodology

## Clarity of Description
The description of the mathematical framework of PG-Merge in Section 3 is clear and easy to follow. The steps of gradient computation, magnitude sorting, percentile thresholding, masking, and the post-update parameter projection are well-defined. However, several critical details are omitted or described in a misleading manner.

## Potential Technical Flaws and Methodological Issues

### 1. Significant Optimizer Discrepancy (Adam vs. SGD)
In **Appendix A**, the authors identify a serious mathematical issue: applying the post-update parameter projection (Equation 13) in adaptive optimizers like Adam causes an "internal state mismatch" because the running momentum buffers are still updated with zero gradients. Over time, these momentum buffers decay toward zero, leading to dampened updates when a frozen coefficient becomes active again. 
To resolve this, the authors state:
> *"To address this nuance and further honor the principle of Occam's razor, we advocate for pairing PG-Merge with standard Stochastic Gradient Descent (SGD) without momentum."*

**The Flaw:** Despite this strong advocacy, **all** primary results in Table 1, Table 2, and the figures are generated using the **Adam optimizer** (as stated in Section 4.1: *"...using the Adam optimizer with a learning rate of $10^{-3}$ and zero weight decay"*). 
The paper does not present any empirical results for PG-Merge paired with SGD. This creates a major gap:
- If SGD is the mathematically "self-consistent" and "redundancy-free" choice, why was it completely omitted from the experiments?
- Does PG-Merge actually work with SGD, or does it fail due to SGD's lack of adaptive step sizes?
- If PG-Merge with Adam still outperforms the baselines despite the momentum decay issue, does the momentum state mismatch actually matter, or is Appendix A purely speculative? 
The lack of SGD results is a glaring methodological omission.

### 2. Mislabeled "Online Test-Time Adaptation" (Offline Calibration in Reality)
The paper repeatedly frames the setup as "online test-time adaptation" (e.g., in Sections 1, 3, and 4). However, the actual experimental setup in Section 4.1 contradicts this:
> *"Active model merging methods minimize prediction entropy on this calibration set over 100 gradient steps..."*

In a true "online test-time adaptation" (TTA) setup (e.g., Tent), data arrives in a stream of sequential, non-repeating batches, and the model performs a single (or very few) gradient steps on each incoming batch before predicting and discarding it. 
Here, the authors take a fixed "calibration validation set" of 64 images and optimize the merging coefficients over **100 gradient steps on this exact same static set**. This is **offline batch calibration**, not online/streaming test-time adaptation. Framing this as "online adaptation" is highly misleading. Under 100 optimization steps on a tiny, static dataset of 64 images, overfitting is inevitable, but it is an artifact of the multi-epoch optimization on a tiny static cache, not a realistic online streaming scenario.

### 3. Classification Head Semantic Conflict
The paper evaluates on 4 highly distinct tasks: MNIST, FashionMNIST, CIFAR-10, and SVHN. Each expert model is fine-tuned starting from the same pre-trained `vit_tiny` model. 
Since they start from the same pre-trained weights, they must share the same classification head structure (likely a 10-class linear head). 
When merging these models in parameter space (Equation 4), the weights of the classification heads are also directly merged. 
**The Flaw:** This represents a massive semantic conflict. The same output logit index (e.g., index 0) represents '0' in MNIST, 'T-shirt' in FashionMNIST, 'airplane' in CIFAR-10, and '0' in SVHN. Directly merging these disjoint heads in weight space forces the model to map completely unrelated semantic concepts to the same logit, causing severe, destructive parameter interference. 
The authors do not discuss this conflict. How is it resolved? If they kept the classification heads separate (i.e., multi-head), then they did not merge the entire model, and the heads were not updated. If they merged the single head, the performance collapse ($78.08\%$ ceiling down to $62.16\%$ uniform) is heavily driven by this semantic head conflict rather than "layer-wise task conflicts." The methodology is highly questionable without clarifying how the classification heads are handled.

### 4. Baseline Tuning and Learning Rate Selection
- **RegCalMerge Calibration:** RegCalMerge has several sensitive hyper-parameters (such as the regularization weight $\lambda$). The authors do not state if they tuned $\lambda$ for this specific `vit_tiny` setup, or if they used a default value from the original paper. If the baseline was not properly tuned, the comparison is unfair.
- **Fixed Learning Rate:** The learning rate is fixed at $10^{-3}$ across all methods and all sparsity ratios $p$. However, it is well-known that when restricting optimization to a very sparse subset of parameters (e.g., $p=0.05$, which translates to updating only 3 parameters out of 56), the optimization dynamics change drastically. A highly sparse update likely requires a much larger learning rate to make meaningful progress in 100 steps, while a dense update ($p=1.0$) might require a smaller learning rate to avoid overfitting. Fixing the learning rate across all $p$ is a major confounding factor in the ablation study.

## Reproducibility
The reproducibility of this paper is low:
- No code repository is provided.
- The exact train/val/test splits and the exact 64 calibration images are not specified.
- The random seed is not mentioned, and no error bars or standard deviations are reported.
