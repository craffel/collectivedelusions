# 1. Summary of the Paper

## Main Topic and Approach
This paper conducts a rigorous post-mortem and limitation-mapping study of joint model merging and pruning on resource-constrained edge hardware. The authors investigate **ZipMerge** (Post-Merge Joint Weight Pruning and Coefficient Tuning), a framework designed to co-optimize layer-wise merging coefficients and magnitude-pruning boundaries at test-time using an unsupervised minimum entropy objective on tiny calibration sets. The authors explore two optimization paradigms for handling the non-differentiable pruning threshold:
1. **ZipMerge (STE):** Uses a Straight-Through Estimator to propagate first-order gradients.
2. **ZipMerge (ES):** Uses a derivative-free 1+1 Evolution Strategy for zero-order search.

## Key Findings and Boundaries Exposed
The core of the paper is a highly honest, empirical investigation that exposes severe limitations of linear weight-space operations and adaptive test-time compression under extreme domain shift (using MNIST, FashionMNIST, CIFAR-10, and SVHN on a compact Vision Transformer backbone):
1. **Catastrophic Representational Collapse:** Every single merged model configuration (including Uniform, AdaMerging, and the proposed ZipMerge STE/ES) suffers from complete representational collapse, performing at or near the random guessing level (10% to 14% absolute accuracy).
2. **The Overfitting-Optimizer Paradox:** Unconstrained unsupervised test-time adaptation on tiny calibration sets overfits transductively, successfully minimizing the entropy objective while destroying generalizable features and driving test-set accuracy down.
3. **Prune-then-Merge Baseline Outperformance:** The simple, unoptimized decoupled baseline, **Prune-then-Merge (P-then-M)**, consistently outperforms complex, joint test-time optimization under high task conflict because pre-merging pruning acts as a spatial regularizer, removing orthogonal parameter noise.
4. **Noisy Expert Noise Injection Constraint:** The quality of the input expert models is a critical prerequisite; merging a single poorly converged model (like the SVHN expert evaluated at 19.59%) acts as a "poison pill" that collapses the performance of the entire multi-task system.

## Explicitly Claimed Contributions
1. **Framework (ZipMerge):** Formalizes joint co-optimization of layer-wise merging coefficients and dynamic binarized pruning masks.
2. **Rigorous Empirical Post-Mortem:** Conducts an honest, boundary-mapping study that resists "curating a narrative of triumph" and instead exposes fundamental limitations.
3. **Comprehensive Ablations and Extensions:**
   - Evaluates alternative objectives (MMI, soft pseudo-labeling, Likelihood Ratio, CBC loss) to mitigate transductive overfitting.
   - Evaluates alternative backbones (ResNet-18) and high-capacity backbones (ViT-Base).
   - Validates PEFT/LoRA adapter merging (+29% absolute improvement) and Orthogonal Procrustes SVD alignment (+16.45% absolute improvement over unaligned LoRA merge).
   - Evaluates structured block-pruning (achieving a 1.89x physical speedup on mobile CPUs).
   - Demonstrates generalizability to autoregressive language models (GPT-2) and details VRAM/CPU runtime scaling optimizations.
4. **Actionable Architectural Guidelines:** Translates empirical failures into system engineering guidelines (avoid extreme task disparity on compact backbones, leverage PEFT, incorporate explicit test-time regularizations).
