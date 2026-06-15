# 3. Soundness and Methodology Evaluation

## Clarity of the Description
The mathematical formulation of SVS, BWN, and the scaling cancellation proofs are presented clearly and are easy to follow. The steps for performing Singular Value Decomposition (SVD) on the task vectors and applying the low-rank projection operator are clearly detailed. However, several critical methodological and conceptual limitations are present.

## Evaluation of Methods and Technical Flaws

### 1. The "Known Task Identity" Evaluation Assumption (Critical Flaw)
In Section 4.1, the authors state:
> *"During multi-task evaluation, the output features of the merged visual backbone are routed to the respective task-specific linear heads. This corresponds to a multi-head evaluation setup where task identity is known at test-time to select the correct downstream classification head."*

This is a profound conceptual and methodological flaw that undermines the entire motivation for model merging:
- If the task identity is known at test-time, **there is absolutely no need to merge the backbones in the first place.**
- One can simply maintain the separate, specialized expert backbones and route the test input to the appropriate expert. This would achieve the "Individual Experts" average performance of **88.93%**, completely bypassing the representation degradation of SVS, which drops the accuracy to **74.83%** (a massive $14.1\%$ absolute drop).
- Model merging is valuable precisely when a single, unified model must handle inputs from multiple domains without explicit test-time routing or task-specific metadata. By assuming task routing is available, the authors have evaluated their method in a scenario where model merging provides zero practical utility.

### 2. SVS Density and the Cross-Layer Interference Problem
The paper's core hypothesis is that SVS resolves parameter interference by acting as an analytical noise filter. However, as the authors acknowledge in Section 4.2:
- SVS yields **dense** low-rank update matrices.
- While it filters high-frequency noise in the spectral domain, these dense matrices still overlap entirely in the spatial coordinate basis.
- Consequently, SVS creates severe spatial parameter collisions when combined across multiple cascading layers. This explains why SVS is significantly outperformed by TIES-Merging (**77.98%** vs. **74.83%**), which explicitly zeros out small weights and resolves sign conflicts to eliminate coordinate-basis overlap.
- Therefore, SVS does not actually solve the fundamental problem of cross-layer representation interference in deep networks; it merely applies a continuous projection that fails to prevent localized parameter collisions.

### 3. SVD Computational Complexity & Unverified Scalability Claims
- Computing the exact SVD of a matrix $T_t \in \mathbb{R}^{m \times n}$ has a computational complexity of $\mathcal{O}(\min(m^2 n, m n^2))$. While this may run quickly on a small 86M parameter model (CLIP-ViT-B/32), it scales very poorly to multi-billion parameter Large Language Models (LLMs) with very large hidden dimensions (e.g., $4096 \times 11008$ in LLaMA-7B).
- The authors claim that this bottleneck can be resolved by adapting SVS to use **Randomized SVD** algorithms. However, this claim is **completely unverified**. The authors do not conduct any experiments with Randomized SVD, nor do they evaluate its impact on merging accuracy or runtimes. Leaving this as a purely theoretical hand-wave is a significant methodological gap.

### 4. Highly Questionable "Expert" Baseline in MLP Validation
In Section 4.5, the authors validate the utility of BWN in a non-normalized environment using a standard 3-layer MLP on MNIST and FashionMNIST:
- The reported independent test accuracy of the experts is incredibly low: **77.00%** on MNIST and **69.00%** on FashionMNIST.
- A basic, standard MLP on MNIST easily achieves over **98%** accuracy, and over **85%** on FashionMNIST. 
- Accuracies of $77\%$ and $69\%$ indicate that these "expert" networks are severely underfit, poorly trained, or architecturally defective. 
- Validating an algorithm (BWN) on a broken, toy setup with sub-standard baselines is highly unconvincing and raises questions about whether the observed "activation norm recovery" is merely an artifact of poor initialization or training dynamics rather than a robust, generalizable property.

### 5. Unjustified Test Set Subsampling
- The evaluation uses a subset of **1,000 samples** per dataset instead of the standard, full test sets (which are 10,000 samples for MNIST, FashionMNIST, and CIFAR-10).
- Subsampling to 1,000 samples is completely unnecessary, as evaluating on the full 10k datasets takes only a few seconds.
- This arbitrary reduction raises concerns about potential selection bias, statistical noise, and a failure to report standard deviations across random subsets.

### 6. Arbitrary Tensor Flattening Grouping
- For higher-dimensional tensors (e.g., convolutional kernels), SVS groups the first dimension (output channels) and flattens all other dimensions.
- The authors state: *"grouping by the output channel dimension provided the most stable multi-task accuracy... though fully characterizing the sensitivity... is an important direction."*
- This choice is highly empirical and lacks any mathematical justification. SVD is highly sensitive to the flattening axis, and applying SVS to reshaped tensors without a thorough geometric analysis or ablation of alternative grouping schemes makes the tensor-flattening aspect of the methodology feel like a fragile heuristic.
