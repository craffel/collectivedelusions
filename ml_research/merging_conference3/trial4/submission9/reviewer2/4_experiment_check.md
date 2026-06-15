# 4. Experimental Check

## 1. Major Weaknesses in Experimental Setup and Methodology

### A. Highly Artificial and Unrepresentative "Toy" Evaluation Setting
The entire empirical evaluation of EPM is restricted to:
- A **ViT-Tiny** backbone containing only **5.7 million parameters**.
- Four highly simple, low-resolution datasets (**MNIST, FashionMNIST, CIFAR-10, SVHN**) upsampled from 28x28 or 32x32 to 224x224.

This setup is highly unrepresentative of modern machine learning workflows where model merging is actually applied:
1. **Model Scale:** Model merging is primarily utilized on Large Language Models (LLMs) and Vision-Language Models (VLMs) (ranging from 1B to 70B parameters) because they have the necessary parameter redundancy to support merging without severe representation destruction. In a tiny 5.7M parameter model, there is virtually no redundant capacity, which makes any weight alteration highly destructive.
2. **Domain Realism:** Fine-tuning a single Vision Transformer on a mixture of grayscale handwritten digits, grayscale clothing articles, and low-resolution color images is a highly artificial task mixture. Upsampling MNIST to 224x224 and feeding it to a ViT is a toy setup.
3. **Practical Utility:** Because of the tiny backbone and extreme domain conflicts, the absolute accuracies achieved by the merged model are extremely low (around **46%** joint mean, with CIFAR-10 dropping to **36.98%** and MNIST dropping to **48.07%** from a 98.74% ceiling). A model that gets only 48% on MNIST and 36% on CIFAR-10 is practically useless for any real-world deployment. The physical trade-off of "zero-overhead serving" is not viable if the resulting model performs worse than random guessing on several categories.

### B. Minimax Objective as an Artificial Performance Trade-off
TLC-Tune optimizes a minimax validation metric:
$$\mathcal{S}_{\text{val}} = \min_{k} \text{Acc}_k(\theta_{\text{merged}}) + 0.1 \cdot \text{Mean}(\text{Acc}(\theta_{\text{merged}}))$$
This objective is a zero-sum game that aggressively sacrifices performance on complex, high-performing tasks to boost the performance of trivial tasks:
- Under untuned EPM ($\Lambda = \mathbf{1.0}$), CIFAR-10 accuracy is **68.89%** and SVHN is **59.41%**, but MNIST is only **15.86%**.
- When TLC-Tune is applied, CIFAR-10 accuracy is slashed by over 31% absolute (dropping to **36.98%**) and SVHN drops to **53.28%**, solely to lift MNIST to **48.07%** and FashionMNIST to **46.42%**.
- In any realistic production scenario, sacrificing 31% accuracy on a difficult color object classification task (CIFAR-10) to raise a trivial, grayscale digit recognition task (MNIST) from 15% to 48% (which is still a failing grade of 48%) is highly illogical. 
- The "balanced multi-task performance" is an artificial construct of the minimax objective. TLC-Tune merely acts as a destructive dial that degrades the best-performing representations until they match the low performance of the worst representations.

### C. Refutation of the "Coordinate Exclusivity" Hypothesis
A central claim of the paper is that "coordinate exclusivity" (Soft-EPA) is the primary mechanism that mitigates weight-space interference. However, the authors' own empirical results in Table 2 ($p=0.5$) directly refute this claim:
- The **Standardized TA + Pruning** baseline (which corresponds to EPM with $\gamma = 1.0$ and lambdas=1.0, representing standard Task Arithmetic combined with standardized pruning, i.e., zero coordinate exclusivity) achieves a joint mean accuracy of **44.87%**.
- In contrast, the proposed **EPM with TLC-Tune** (which utilizes Soft-EPA coordinate exclusivity and DCS with $\gamma = 0.4$) achieves a joint mean accuracy of **42.60%**.
- This means that standard average-based weight blending (Task Arithmetic) combined with standardized pruning actually **outperforms** the proposed coordinate exclusivity routing by **2.27% absolute**.
- The authors attempt to justify this by arguing that TLC-Tuned EPM "shields" MNIST, raising its accuracy to 59.15% compared to Standardized TA's 15.77%. However, this "shielding" is merely the result of the minimax objective forcing the parameters to favor MNIST, which drags down SVHN performance to 41.99% (compared to Standardized TA's 67.14%) and CIFAR-10 to 37.86% (compared to Standardized TA's 61.67%).
- This empirical result shows that the proposed coordinate-exclusive routing actually degrades the overall capacity of the network compared to standard averaging, and any "balance" is an artifact of the tuning objective, not the Soft-EPA routing.

## 2. Baseline Comparison Integrity
- **Scale Tuning Fairness:** The authors state that all deterministic baselines (Task Arithmetic, Prune-then-Merge, TIES-Merging, DARE) were tuned across global scale factors $\lambda \in \{0.1, 0.2, 0.3, 0.4, 0.5\}$ on the exact same 128-sample validation split. This ensures a fair scale-tuning setup.
- **Optimizer Mismatch:** As discussed in the Soundness section, evaluating SOTA continuous optimization methods (AdaMerging and ZipMerge) under zero-order (1+1)-ES is a major flaw that renders the comparison with these baselines invalid. A fair baseline comparison would have evaluated AdaMerging and ZipMerge using first-order gradient descent (their native optimization pathway) on the calibration split.
- **Omission of Modern SOTA Merging Methods:** The baselines are restricted to older or simpler methods. Modern weight-space merging and alignment frameworks, such as *Git Re-Basin* (permutation alignment), *RegMean* (regression-based alignment), or *Fisher-weighted merging*, are omitted. Comparing against these would have provided a much stronger baseline suite.
