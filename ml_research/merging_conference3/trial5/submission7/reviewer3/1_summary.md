# Intermediate Evaluation: Summary of the Paper

## Main Topic
The paper addresses the challenge of **test-time model merging** (active parameter-space model fusion) where multiple specialized expert networks (fine-tuned from a shared base model) are combined into a single multi-task model on-the-fly. The optimization of layer-wise merging coefficients is performed using unlabeled test-time adaptation (TTA) batches by minimizing prediction entropy.

## Approach
To mitigate the **"Overfitting-Optimizer Paradox"**—where unconstrained optimization of high-dimensional coefficients on tiny, unlabeled calibration streams leads to transductive overfitting and catastrophic representational decay—the authors propose **Pruned Gradient Merging (PG-Merge)**. 

PG-Merge is a non-parametric, training-free approach that:
1. Calculates the gradients of the prediction entropy loss with respect to all layer-wise merging coefficients.
2. Generates a binary sparse gradient mask by sorting the absolute values of the gradients and keeping only the top-$p\%$ (e.g., $5\%$ or $15\%$) largest coordinates, zeroing out the rest.
3. Updates only the selected active coefficients while keeping the remaining $(100-p)\%$ coefficients frozen via a post-update projection step to prevent momentum leakage in adaptive optimizers like Adam.

## Key Findings
1. **Severe Parameter Interference:** Combining specialized experts statically (Uniform Merging) results in a large performance drop ($78.08\%$ expert ceiling vs. $62.16\%$ joint average), indicating substantial task conflicts.
2. **Unconstrained Adaptation Overfits:** Unconstrained online AdaMerging overfits to local unlabeled batch statistics, degrading performance further to $61.08\%$.
3. **Rigid Subspaces Fail:** Restricting adaptation to a low-degree polynomial trajectory (PolyMerge) is overly restrictive and collapses MNIST performance to near-random ($13.48\%$), resulting in a joint mean of $46.97\%$.
4. **PG-Merge Effectiveness:** Restricting updates to only $5\%$ of coordinates ($p=0.05$) yields the best joint multi-task accuracy ($62.70\%$), outperforming both the static uniform baseline ($62.16\%$), unconstrained AdaMerging ($61.08\%$), and the complex regularizer RegCalMerge ($62.35\%$).

## Explicitly Claimed Contributions (with Evidence)
1. **Exposing Redundant Complexity:** The authors claim that complex spatial regularizers (like RegCalMerge) and rigid trajectory restrictions (like PolyMerge) are unnecessarily complex and can be outperformed by simple gradient pruning. *Evidence:* Table 1 shows PG-Merge ($p=0.05$) achieves $62.70\%$ joint accuracy compared to $62.35\%$ for RegCalMerge and $46.97\%$ for PolyMerge.
2. **Dynamic Gradient Sparsity Framework:** PG-Merge is presented as an elegant, training-free, and hyperparameter-lean method. *Evidence:* The mathematical formulation in Section 3 and the ablation study in Section 4.3.
3. **Exhaustive Evaluation:** The authors claim robust validation on a Vision Transformer across four tasks (MNIST, FashionMNIST, CIFAR-10, SVHN). *Evidence:* Table 1 and Table 2 report accuracy across these datasets.
4. **Sparsity "Sweet Spot" Identification:** The authors identify that updating only $5\%-15\%$ of coefficients represents the optimal regularizing regime. *Evidence:* The ablation study in Table 2, showing a monotonic performance decline as $p$ increases from $0.05$ to $1.0$.
