# Experimental and Results Check

This section critically evaluates the updated experimental setup, baselines, and results in the paper.

## 1. Quality of the Updated Experimental Setup
The revised experimental setup is highly rigorous and scientifically sound:
*   **Converged Expert Models:** Addressing prior issues, the task-specific expert models have been trained to full convergence (15 epochs). Their individual "Expert Ceiling" performance is highly competitive:
    *   **MNIST:** $93.55\%$
    *   **FashionMNIST:** $81.25\%$
    *   **CIFAR-10:** $87.30\%$
    *   **SVHN:** $50.20\%$
    *   **Joint Mean:** $78.08\%$
    This high baseline performance ensures that subsequent model merging and test-time adaptation experiments are carried out on highly specialized, convergent representations rather than high-variance noise.
*   **The Backbone and Datasets:** Evaluating on a standard compact Vision Transformer (`vit_tiny_patch16_224`) across four diverse vision classification tasks (MNIST, FashionMNIST, CIFAR-10, SVHN) is appropriate. It presents a challenging, high-interference setting.

## 2. Analysis of the Main Results (Table 1)
The main results strongly support the paper's core claims:
*   **The Overfitting-Optimizer Paradox Validated:** Unregularized layer-wise online AdaMerging degrades performance compared to the static baseline ($61.08\%$ vs. $62.16\%$), confirming that unconstrained entropy-minimizing TTA overfits to local test-batch statistics.
*   **Catastrophic Failure of Rigid Subspaces (PolyMerge):** PolyMerge, which restricts coefficients to a 12-parameter quadratic trajectory, collapses catastrophically to a Joint Mean of $46.97\%$ (with MNIST dropping to $13.48\%$, near-random). This is an outstanding result that supports the paper's critique: rigid, pre-defined geometric subspaces cannot handle the complex, non-linear routing relationships required when merging fully converged expert models.
*   **The Power of Sparsity (PG-Merge):** Restricting updates to only the top-$5\%$ most sensitive coordinates ($p=0.05$) represents the optimal regularization regime, yielding the highest Joint Mean of \textbf{62.70\%}. This outperforms unconstrained AdaMerging ($61.08\%$), static Uniform Merging ($62.16\%$), and matches or exceeds SOTA RegCalMerge ($62.35\%$).
*   **Minimalism Vindicated:** PG-Merge ($p=0.05$) achieves SOTA performance without adding the complex spatial penalties ($\lambda$) or class-capacity normalizations used in RegCalMerge. This strongly vindicates the Minimalist persona.

## 3. Areas for Improvement and Missing Analysis

### A. Performance on SVHN Under High Sparsity
While PG-Merge ($p=0.05$) performs exceptionally well overall and achieves the best individual results on MNIST ($63.28\%$) and FashionMNIST ($75.59\%$), its performance on **SVHN** drops to **$32.03\%$**, which is worse than both static Uniform Merging ($33.20\%$) and unconstrained AdaMerging ($35.16\%$). 
*   In contrast, PolyMerge performs exceptionally well on SVHN ($40.43\%$) despite its overall collapse. 
*   This suggests that SVHN—which represents a highly distinct and complex domain (street numbers)—might require more than $5\%$ active coordinates to adapt effectively, or its gradients are noisier. The paper should discuss this interesting trade-off and analyze why SVHN behaves differently under extreme sparsity.

### B. Missing Optimization Curves
The paper would benefit from plotting prediction entropy (adaptation loss) and joint test accuracy over the 100 adaptation steps. This would provide a direct visual verification of the "Overfitting-Optimizer Paradox" in action and clearly show how PG-Merge stabilizes the optimization path compared to unconstrained AdaMerging.
