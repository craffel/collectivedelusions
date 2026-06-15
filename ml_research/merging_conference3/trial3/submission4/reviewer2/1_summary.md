# Evaluation Step 1: Paper Summary

## Main Topic and Objective
The paper conducts a rigorous post-mortem and limitation-mapping study of joint model merging and weight pruning under resource-constrained edge-deployment settings. The goal is to investigate whether co-optimizing layer-wise merging coefficients and magnitude-pruning boundaries at test-time can produce a sparse, multi-task model capable of running efficiently on edge hardware. To achieve this, the authors introduce **ZipMerge**, a training-free framework that integrates dynamic magnitude pruning directly into the test-time adaptation (TTA) loop. Rather than treating merging and pruning as disjoint sequential steps, ZipMerge co-optimizes merging coefficients and the binary pruning mask using an unsupervised minimum entropy objective on a tiny calibration dataset (16 unlabeled images per task).

## Proposed Approach
The framework co-optimizes layer-wise merging coefficients $\Lambda$ and a binary pruning mask $M(\Lambda)$ under a target global sparsity ratio $p$. 
To navigate the non-differentiable boundary introduced by magnitude pruning, the authors evaluate two optimization paradigms:
1. **ZipMerge (STE):** First-order gradient descent via a Straight-Through Estimator (specifically, Identity-pass STE) to propagate gradients through the non-differentiable masking operator.
2. **ZipMerge (ES):** Derivative-free black-box 1+1 Evolution Strategy (1+1 ES) to explore the coefficient space via isotropic perturbations.

The unsupervised objective is the Shannon entropy of predicted probability distributions over the tiny calibration batch, which requires no ground-truth labels.

The authors also propose several advanced extensions:
* **Reg-ZipMerge:** A regularized variant incorporating a structural distance penalty or a functional KL distillation penalty to mitigate transductive overfitting.
* **Low-Conflict and PEFT Merging:** Evaluations on domain-aligned benchmarks (DomainNet) and a parameter-efficient fine-tuning (PEFT/LoRA) setup.
* **Orthogonal Procrustes Alignment:** A data-free, post-hoc SVD-based rotation method to align coordinate spaces of separately trained LoRA adapters before merging.
* **Structured block-pruning:** Removing entire attention heads or MLP neurons to realize physical latency gains on hardware.
* **Hardware Profiling:** Solutions for dynamic sorting overhead (Delayed Thresholding and Histogram-based Quantile Estimation) and joint post-training quantization (PTQ) simulation.
* **GPT-2 Language Model Evaluation:** Extending the co-optimization to autoregressive multilingual text generation.

## Key Findings & Empirical Boundaries
Instead of a curated narrative of success, the paper honest-reports the following empirical boundaries under high-conflict (MNIST, FashionMNIST, CIFAR-10, SVHN) setups using a compact Vision Transformer (`vit_tiny_patch16_224`):
1. **Catastrophic Representational Collapse:** Every merged model—including Uniform, AdaMerging, and the proposed ZipMerge variants—suffers from absolute representation collapse, performing near the level of random guessing (10% to 14% accuracy). This is due to severe weight and activation pathway interference when merging highly orthogonal domains onto a compact backbone.
2. **Prune-then-Merge (P-then-M) Outperformance:** The unoptimized decoupled baseline, Prune-then-Merge (P-then-M), consistently outperforms test-time joint optimization because pre-merging pruning zeroes out minor, task-specific updates, acting as a spatial regularizer that reduces parameter noise.
3. **The Overfitting-Optimizer Paradox:** Unconstrained minimum-entropy TTA on tiny calibration sets overfits transductively, successfully minimizing entropy while destroying generalizable features and driving test-set accuracy down.
4. **PEFT and Aligned Merging as Solutions:** Restricting training to low-rank manifolds (LoRA) dramatically reduces backbone representation shifts, boosting dense merge accuracy by +29% absolute. Introducing Orthogonal Procrustes alignment further rotates the adapters into a shared basis, boosting accuracy to 58.75% dense and 62.10% under 50% sparsity (with ZipMerge-ES).
5. **Structured Pruning Latency Wins:** Transitioning to structured block-pruning of neurons and attention heads delivers a 1.89$\times$ physical speedup on an ARM Cortex-A76 mobile CPU (34.2 ms down to 18.1 ms per image) with minimal optimization degradation.
6. **Zero-Order ES Memory Advantage:** For GPT-2 language models, ZipMerge (ES) achieves a 13.2$\times$ peak memory savings (1.12 GB vs 14.82 GB) over STE at larger context lengths (1024 tokens) because it completely bypasses backpropagation and activation caching.

## Claimed Contributions and Evidence
* **Systematic Post-Mortem of Merging & Pruning:** Supported by rigorous experiments evaluating various sparsity levels ($p \in \{0.0, 0.5, 0.8\}$) across multiple baselines on ViT-Tiny.
* **Dual Optimization Framework (ZipMerge):** Detailed mathematical formulation and algorithmic tracing for first-order (STE) and zero-order (ES) co-optimization of coefficients and pruning masks.
* **Actionable Architectural Guidelines:** Actionable insights for edge engineers regarding task disparity, PEFT adapters, test-time regularization, and structured block-pruning supported by ARM CPU latency measurements and physical VRAM profiling.
* **SVD-Based Post-Hoc Coordinate Alignment:** Complete algorithm and empirical evidence showing that Orthogonal Procrustes alignment resolves coordinate misalignment and recovers most of the lost multi-task performance in PEFT space.
* **Quantization & Sorting Mitigations:** Simulated PTQ and physical CPU profiling of histogram-based quantile estimation demonstrating high-efficiency edge execution.
