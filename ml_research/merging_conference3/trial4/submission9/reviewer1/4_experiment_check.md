# Intermediate Evaluation: 4. Experiment Check

## Critical Evaluation of the Experimental Setup
The experimental setup is designed around a compact **Vision Transformer backbone (ViT-Tiny)** containing 5.7 million parameters. The authors justify this choice by arguing that a compact model lacks redundant parameter capacity, thereby acting as a rigorous test for weight-space conflicts. While this is true, it represents an extreme and unrepresentative setting. Modern weight-space model merging is predominantly applied to massive Large Language Models (LLMs) or Vision-Language Models (VLMs) with billions of parameters. In these overparameterized regimes, the mathematical properties of the parameter space (orthogonal subspaces, gradient physics, capacity redundancy) are fundamentally different from those of a compact 5.7M parameter model. Thus, the empirical findings on ViT-Tiny are highly localized and do not automatically generalize to large-scale foundation models.

The four datasets chosen (MNIST, FashionMNIST, CIFAR-10, SVHN) represent highly disparate and conflicting domains (grayscale digits vs. natural color objects). This creates extreme task conflict, which is useful for evaluating "worst-case" behavior, but is highly unrepresentative of realistic multi-task merging scenarios where practitioners typically merge models within the same modality (e.g., merging different code LLMs, or merging specialized vision models).

---

## Baselines
The baselines selected are comprehensive and represent the standard state-of-the-art in weight-space merging, including Task Arithmetic, AdaMerging, Prune-then-Merge, TIES-Merging, DARE, and ZipMerge. A critical control baseline, Random Tensor Routing (RTR), is also included. All standard baselines were tuned across global scale factors $\lambda \in \{0.1, 0.2, 0.3, 0.4, 0.5\}$ on the same validation split to ensure a fair comparison. 

However, there is an **optimizer mismatch** in the evaluation of AdaMerging and ZipMerge. These baselines were originally designed to be optimized via first-order gradient descent on differentiable cross-entropy losses. By restricting their optimization to a non-differentiable minimax accuracy validation score under a black-box zero-order (1+1)-ES search, the authors created a setup where these 56- and 70-dimensional non-convex search spaces are mathematically expected to fail. While this demonstrates that these baselines are incompatible with zero-order accuracy-aligned search, it is a biased comparison that does not represent their performance under their native, first-order gradient-based optimization pipelines.

---

## Do the Results Support the Claims?
While the authors claim that EPM outperforms other baselines and successfully mitigates representation collapse, a closer inspection of the quantitative results reveals major trade-offs and potential contradictions:

1. **Extreme Performance Degradation:** The individual unmerged expert ceilings are high: MNIST (98.74%), FashionMNIST (91.31%), CIFAR-10 (95.88%), and SVHN (93.72%), with a Joint Mean Ceiling of 94.91%. In contrast, the best-performing merged model, EPM (TLC-Tune), achieves a Joint Mean of only **46.19%** (dense) and **42.60%** (50% sparse). This represents a massive absolute performance drop of **48% to 52%** compared to the expert ceilings. In a practical deployment, a model that performs at ~46% accuracy on simple classification tasks is barely functional, casting doubt on the practical utility of EPM under severe task conflicts.

2. **The Minimax Optimization Zero-Sum Trade-off:**
In Table 1 (dense), untuned EPM ($\Lambda = \mathbf{1.0}$) achieves:
- MNIST: 15.86%, FashionMNIST: 38.31%, CIFAR-10: **68.89%**, SVHN: **59.41%**, Joint Mean: **45.62%**
TLC-Tuned EPM (which optimizes the worst-case floor) achieves:
- MNIST: **48.07%**, FashionMNIST: **46.42%**, CIFAR-10: 36.98%, SVHN: 53.28%, Joint Mean: **46.19%**
This reveals a severe zero-sum trade-off. TLC-Tune severely degrades the accuracy of the complex, high-performing CIFAR-10 expert (dropping from **68.89%** to **36.98%**, a massive **31.9% absolute drop**) just to pull the simple, grayscale MNIST expert up from 15.86% to 48.07%. Sacrificing a complex visual expert to improve a trivial digit classifier is highly undesirable in practical scenarios. The joint mean improvement is negligible (45.62% vs 46.19%). This raises a critical question: does TLC-Tune actually perform meaningful multi-task optimization, or does it merely shuffle capacity between experts in a zero-sum game?

3. **The Exclusivity Contradiction under Sparsity:**
In Table 4 (Sensitivity Analysis of $\gamma$ at $p=0.5$), the authors show that as $\gamma$ increases from $0.0$ (pure coordinate exclusivity) to $1.0$ (standard Task Arithmetic with pruning, zero exclusivity), the Joint Mean accuracy rises monotonically from **41.79%** to **46.14%**. This directly contradicts the core thesis that "coordinate exclusivity" is the primary mechanism driving model merging success. In fact, the standard average-based weight sharing ($\gamma=1.0$) achieves the highest joint average accuracy under 50% sparsity. While the authors argue that Soft-EPA protects the worst-performing task (MNIST) from collapsing, the empirical evidence demonstrates that standard average blending remains superior in terms of joint average performance, undermining the claimed necessity of coordinate exclusivity.

4. **Missing Empirical Evidence for Activation Manifold Claims:**
In Section 4.3, the authors make detailed empirical assertions regarding Centered Kernel Alignment (CKA) similarities and t-SNE visualizations (e.g., "CKA similarity between the merged model and the target experts decays exponentially with layer depth because hard-routing... creates discontinuous representation boundaries... recovering high CKA values and maintaining cohesive clustering of task classes..."). However, **there are no CKA plots, tables, or t-SNE visualizations anywhere in the paper or supplementary files.** Presenting detailed empirical conclusions about activation manifolds without providing the underlying quantitative data or figures is a major scientific flaw.
