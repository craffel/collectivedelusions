# Paper Summary

## 1. Objective and Problem Statement
The paper addresses the challenge of **spatial weight-space interference** in weight-space model merging. When task-specific expert models (fine-tuned from a shared pre-trained base model) are consolidated into a single unified network, standard average-based weight interpolation (e.g., Task Arithmetic, TIES-Merging) dilutes task-specific features due to opposing parameter updates. This weight-space conflict leads to representational collapse, especially under high domain shifts. While continuous test-time adaptation (TTA) methods can tune merging coefficients post-hoc, they are highly parameterized, unstable under zero-order search, and prone to the **Overfitting-Optimizer Paradox** (transductive overfitting on small validation calibration sets). The paper aims to design a simple, robust, training-free parameter routing protocol, **Exclusive Parameter Merging (EPM)**, to mitigate weight-space interference at the coordinate level directly during parameter composition.

## 2. Proposed Methodology
The proposed framework consists of two core techniques:
* **Soft Exclusive Parameter Allocation (Soft-EPA)**:
  * Evaluates a coordinate-wise relative routing metric using absolute task-vector updates standardized by their task-specific global standard deviations ($\sigma_k$). This **Task Vector Standardization** prevents experts with large gradient norms (e.g., color datasets like CIFAR-10 or SVHN) from systematically dominating and erasing updates of simpler tasks (like MNIST or FashionMNIST).
  * For each coordinate, the dominant expert is routed at full strength, while updates from non-dominant experts are attenuated by a coherence retention factor $\gamma = 0.2$. This soft assignment acts as a background "glue" to preserve multi-layer activation manifold alignment.
  * To prevent capacity starvation under sparse merging, the paper introduces **Dynamic Coherence Scheduling (DCS)**, where $\gamma(p)$ dynamically scales with the target network sparsity $p$ via a quadratic rule: $\gamma(p) = \gamma_0 + (1 - \gamma_0) \cdot p^2$.
  * A crucial design choice is the **decoupling of scale**: standardization is used exclusively as a decision routing filter, while actual physical weight integration is kept in the original unstandardized weight space to preserve pre-trained activation physics.
* **Task-Level Coefficient Tuning (TLC-Tune)**:
  * Restricts the optimization space to only $K$ global scaling factors (one per expert) to bypass the high-dimensional overfitting paradox.
  * Optimizes these $K$ factors on a tiny validation split (128 samples per task) using a stable, gradient-free (1+1) Evolution Strategy (ES) to maximize a balanced minimax validation score ($\min_k \text{Acc}_k + 0.1 \cdot \text{Mean}(\text{Acc}_k)$).

## 3. Experimental Evaluation
* **Backbone**: Vision Transformer (`vit_tiny_patch16_224`, 5.7M parameters) from the `timm` library.
* **Datasets and Experts**: MNIST, FashionMNIST, CIFAR-10, SVHN.
* **Baselines**: Task Arithmetic (TA), AdaMerging, Prune-then-Merge, TIES-Merging, DARE, ZipMerge, Random Tensor Routing (RTR), and Standardized TA + Pruning.
* **Target Sparsity Levels**: Dense ($p=0.0$), Moderate Sparsity ($p=0.5$), and Extreme Sparsity ($p=0.8$).

## 4. Primary Claims and Empirical Findings
* **Mitigation of Representational Collapse**: Under dense merging ($p=0.0$), TLC-Tuned EPM achieves **46.19%** joint mean accuracy, outperforming Task Arithmetic (40.96%) and TIES-Merging (20.55%) by creating a highly balanced multi-task model.
* **Deconstructing Overfitting vs. Optimization Failure**: A systematic 500-step optimization study shows that high-dimensional block-group-wise tuning methods (AdaMerging, ZipMerge) fail to converge under a zero-order search paradigm because their continuous parameter spaces (56 and 70 dimensions) are too large for (1+1)-ES, remaining completely flat across all 500 steps. In contrast, TLC-Tune's 4-dimensional space converges in 40 steps, proving that their failure is due to absolute optimization failure (under-convergence) rather than transductive overfitting.
* **Self-Honest Limitations**:
  * **Extreme Sparsity Underperformance**: Under $p=0.8$, even with Dynamic Coherence Scheduling (which raises the accuracy to **26.41%** compared to the un-scheduled baseline's **24.11%**), EPM is heavily outperformed by DARE (**40.90%**), which benefits from expectation-value scaling to preserve activation scales.
  * **Optimizer Mismatch**: Acknowledges that SOTA continuous adaptation methods were designed for first-order gradient descent, and their poor zero-order performance is an artifact of this mismatch.
  * **Zero-Sum Minimax Trade-off**: Improving worst-case performance (MNIST/FashionMNIST) significantly collapses complex color experts (CIFAR-10 collapses from **68.89%** down to **36.98%** under TLC-Tune).
