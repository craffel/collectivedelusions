# Soundness & Methodology Check

## 1. Strengths of the Methodology

### Exemplary Scientific Honesty and Transparency
The methodological soundness is heavily bolstered by the paper's outstanding transparency. Rather than obfuscating the weaknesses of the proposed approach, the authors dedicate entire sections to dissecting:
* **The Exclusivity Contradiction**: They show that Standardized TA + Pruning ($\gamma=1.0$) can achieve a higher joint average accuracy than EPM ($\gamma=0.2$) under 50% sparsity, but show that EPM serves as a crucial spatial shield that lifts the worst-performing task (MNIST) by over 16% absolute accuracy, raising the worst-case floor.
* **Failure under Extreme Sparsity**: They openly document and mathematically explain why EPM underperforms DARE under $p=0.8$, showing that coordinate exclusivity starves the model of representational capacity under high pruning constraints.
* **Optimizer Mismatch**: They explicitly address that evaluating AdaMerging and ZipMerge under zero-order (1+1)-ES creates an optimizer mismatch, as these methods were designed for first-order gradient descent on differentiable losses.

### Theoretical Grounding of Soft-EPA
The derivation of Equation 9 showing that Soft-EPA is a convex/linear combination of pure exclusivity ($\gamma=0$) and Task Arithmetic ($\gamma=1$) is mathematically correct and elegant. It provides clear theoretical intuition for the role of $\gamma$ as a background topological "glue" that preserves activation manifold coherence across layers.

### Rigorous Defense of Scale Decoupling
The decision to decouple standardization for routing decisions from unstandardized values for physical weight updates is highly sound. The authors provide concrete statistics (scale overrides occurring at exactly 13.79% of all coordinates, directional overrides where SVHN dominates MNIST/FashionMNIST) to justify this design. This empirical analysis provides strong evidence that standardizing the physical weights would distort pre-trained activation physics and destroy learned feature scales.

### Dynamic Coherence Scheduling (DCS)
The formulation of DCS ($\gamma(p) = \gamma_0 + (1-\gamma_0) \cdot p^2$) is theoretically robust. It acknowledges that coordinate collision probability decreases quadratically with pruning, allowing the model to transition smoothly from highly exclusive routing under dense conditions to cooperative distributed blending under extreme pruning. This successfully prevents capacity starvation under high sparsity.

---

## 2. Weaknesses and Flaws in Methodology

### Weakness 1: Default Selection of Global vs. Layer-wise Standardization
* **Problem**: Although the authors mathematically formulate **Layer-wise Task Vector Standardization** (Equations 4 and 5) and provide an elegant comparative analysis of scale granularity, they utilize Global Standardization as the primary default in their experiments.
* **Critique**: In deep networks, weights in different layers operate on highly distinct ranges and gradient scales. While they argue that Global Standardization is default because it preserves the relative magnitude ratios across different layers (crucial for keeping the activation manifold coherent), utilizing a single global standard deviation $\sigma_k$ can still lead to specific layers being dominated by a single expert whose updates are localized there. 
* **Implication**: Although the theoretical trade-offs are well-articulated, the lack of extensive empirical comparisons between global and layer-wise standardization in the results leaves a minor gap. If early layers of SVHN have massive updates, Global Standardization might allow SVHN to dominate early attention layers and drown out MNIST.

### Weakness 2: Practical Utility of the Minimax Objective (Zero-Sum Trade-off)
* **Problem**: TLC-Tune optimizes for a balanced minimax validation score (Equation 13) to raise the worst-performing task's accuracy.
* **Critique**: This minimax formulation lifts the worst-performing tasks (MNIST and FashionMNIST) but causes a severe **zero-sum trade-off**:
  * For dense merging ($p=0.0$), TLC-Tuned EPM lifts MNIST from **15.86%** to **48.07%** and FashionMNIST from **38.31%** to **46.42%**.
  * However, this comes at the cost of **collapsing** the harder color tasks: CIFAR-10 collapses from **68.89%** down to **36.98%**, and SVHN drops from **59.41%** to **53.28%**.
  * MNIST is a trivial dataset where a basic linear model easily gets $>90\%$ accuracy. Collapsing a highly complex representation like CIFAR-10 (ceiling 95.88%) by over 30% absolute accuracy just to raise MNIST to 48% is a poor trade-off in practical engineering.
* **Acknowledge authors' response**: The authors do address this by mapping the Pareto frontier (e.g., pointing out that completely untuned EPM ($\Lambda = \mathbf{1.0}$) achieves a highly competitive joint mean of **45.62%** while keeping CIFAR-10 at **68.89%** and SVHN at **59.41%**), and by proposing weighted or constraint-based objectives. Nonetheless, the practical utility of a pure minimax objective remains questionable under wildly disparate task difficulties.

### Weakness 3: Baseline Comparison Bias under Optimizer Mismatch
* **Problem**: SOTA continuous tuning methods (AdaMerging and ZipMerge) are evaluated under (1+1)-ES, leading to "absolute optimization failure."
* **Critique**: These baselines are natively designed for first-order gradient descent on differentiable validation cross-entropy losses. Forcing them to optimize a 56- or 70-dimensional non-convex continuous space using a single-point greedy zero-order search creates an optimizer mismatch.
* **Acknowledge authors' response**: The authors address this by explicitly running a 500-step optimization study showing that AdaMerging and ZipMerge remain completely flat and stuck across all steps (0.1725 and 0.1732 respectively), proving they suffer from absolute optimization failure (under-convergence in high-dimensional spaces under (1+1)-ES) rather than transductive overfitting. They also present a differentiable softmax TLC-Tune formulation for first-order gradients. However, the empirical superiority of EPM over AdaMerging and ZipMerge in the main tables remains valid only under this artificially restricted zero-order search setup.

---

## 3. Soundness Rating: Good
Despite the weaknesses, the methodology is sound. The mathematical derivations are correct, the design decisions (especially scale decoupling and DCS) are rigorously justified through empirical statistics and probability analyses, and the authors are exceptionally self-honest and upfront about their limitations, trade-offs, and optimizer mismatches.
