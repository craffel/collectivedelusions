# 1_summary.md: Paper Summary

## Main Topic and Approach
This paper investigates the pragmatic challenges of deploying merged multi-task models onto resource-constrained edge hardware. While model merging (e.g., via task arithmetic) combines specialized expert networks without retraining or requiring full datasets, the resulting model remains fully dense, presenting a physical memory bottleneck for on-device deployment. 

To address this, the authors propose and evaluate **ZipMerge**, a training-free framework designed to produce sparse, multi-task merged models ready for low-memory edge devices. ZipMerge co-optimizes layer-wise merging coefficients and binary magnitude-pruning boundaries at test-time on a tiny, unlabeled calibration dataset (16 images per task) using an unsupervised minimum entropy objective. 

The paper explores two optimization engines for navigating the non-differentiable pruning threshold:
1. **First-order gradient descent via a Straight-Through Estimator (STE)**, utilizing an "Identity-pass" gradient flow to route updates through pruned and active weights alike.
2. **Zero-order search via a 1+1 Evolution Strategy (1+1 ES)**, a robust, derivative-free alternative that explores the coefficient space without requiring gradient propagation.

## Key Findings
Rather than presenting a curated success narrative, the paper provides a scientifically rigorous and honest "post-mortem" under extreme conditions (merging highly disparate domains—MNIST, FashionMNIST, CIFAR-10, and SVHN—onto a compact 5.7M parameter Vision Transformer backbone):
1. **Catastrophic Representational Collapse:** Under high task conflict, all merged configurations (including Uniform, AdaMerging, and the proposed ZipMerge) suffer from complete representation collapse, performing near the random guessing baseline (~10% to 14% accuracy). This is due to severe parameter conflicts and activation path cancellations across highly orthogonal domains on a compact backbone.
2. **The Overfitting-Optimizer Paradox:** Unconstrained unsupervised test-time adaptation (minimizing Shannon entropy on a tiny calibration set) overfits transductively. The optimizer successfully drives down entropy (making predictions peaky and confident) but destroys the general, robust features learned during expert fine-tuning, thereby degrading out-of-domain test performance.
3. **Prune-then-Merge (P-then-M) Outperformance:** A simple, decoupled baseline—pruning individual task vectors *prior* to merging—consistently and significantly outperforms complex, joint test-time optimization under high task conflict (achieving 14.81% accuracy at 50% sparsity and 16.97% at 80% sparsity). Pre-merging pruning acts as a spatial regularizer, removing conflicting parameter noise and protecting task-specific features from mutual cancellation.
4. **Optimizer Trajectory Geometry:** 1+1 ES outperforms STE at moderate (50%) sparsity because it avoids the high gradient variance of STE's approximations. However, at aggressive (80%) sparsity, STE is superior because the restricted active parameter space reduces gradient variance, whereas ES struggles with a flat, disjointed loss landscape where most coefficient changes do not alter which weights survive.

## Explicitly Claimed Contributions (with Evidence)
1. **Formulation of ZipMerge:** A joint co-optimization framework of continuous merging coefficients and binary pruning masks at test-time under unsupervised entropy minimization, with implementations of both first-order (STE) and zero-order (ES) engines (Sections 3.1 to 3.5).
2. **Critical Post-Mortem of Model Merging and Pruning Limits:** Mapping out representation collapse and optimizer failure under extreme task orthogonality on compact backbones (Section 4.2).
3. **The Overfitting-Optimizer Paradox:** Highlighting and empirically tracing the divergence between calibration entropy minimization and test-set generalization (Section 4.2.3).
4. **Actionable Architectural Recommendations for Edge Systems:** Translating boundary failures into practical system guidelines:
   - *Domain-Aligned Merging:* Restricting merging to compatible task groups (validated on DomainNet, where ZipMerge ES achieves 74.20% accuracy at 50% sparsity vs. 76.10% dense reference).
   - *PEFT/LoRA Adapter Merging:* Restricting fine-tuning to low-rank manifolds to protect the core backbone (validated by achieving a +29% absolute improvement over full-backbone dense merging).
   - *Orthogonal Procrustes SVD Alignment:* A computationally lightweight, data-free post-hoc rotation of LoRA adapters before averaging that resolves coordinate basis mismatches, boosting accuracy by **+16.45%** absolute (achieving 58.75% dense and 62.10% sparse accuracy, closing 67.5% of the gap to unpruned experts with negligible sub-millisecond overhead).
   - *Structural Regularization (Reg-ZipMerge):* Introducing a distance penalty to restrict coefficient drift, which partially preserves test accuracy (Section 4.3.1).
