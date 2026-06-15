# Summary of the Paper

## Title
**ZipMerge: Joint Weight Pruning and Test-Time Coefficient Tuning for On-Device Model Merging**

## Author & Affiliation
* Elena Rostova (Georgia Institute of Technology)

## Context and Problem Definition
Model merging has emerged as a powerful, training-free paradigm to combine independently fine-tuned task-specific experts into a single collaborative multi-task model without needing the original massive training datasets. However, a critical limitation remains: the merged model is still fully dense (100% parameter size), making deployment on resource-constrained edge hardware highly challenging due to limited RAM, storage, and thermal envelopes. 

To make merged models viable on-device, they must be compressed. Naively sequencing merging and pruning leads to suboptimality. Therefore, there is a clear need for joint model merging and network pruning tailored for physical edge deployment.

## Proposed Framework: ZipMerge
The paper presents **ZipMerge** (Post-Merge Joint Weight Pruning and Coefficient Tuning), a framework that integrates dynamic magnitude pruning directly into the test-time adaptation (TTA) loop. Rather than treating merging and pruning as separate, decoupled stages, ZipMerge co-optimizes the continuous layer-wise merging coefficients $\Lambda$ and the binary pruning mask $M$ simultaneously on a tiny, unlabeled calibration dataset (e.g., 16 images per task) using an unsupervised minimum Shannon entropy loss.

To navigate the non-differentiable pruning boundary, the authors evaluate two optimization paradigms:
1. **ZipMerge (STE):** Uses a Straight-Through Estimator (STE) (with Identity-pass global gradient flow) to propagate first-order gradients back to the continuous coefficients $\Lambda$ through the non-differentiable threshold mask.
2. **ZipMerge (ES):** Uses a derivative-free 1+1 Evolution Strategy (1+1 ES) to explore the coefficient space, completely bypassing gradient computation.

## Key Core Findings & "Honest Post-Mortem" Analysis
Instead of presenting a curated story of absolute victory, the paper conducts a highly rigorous, honest empirical investigation using a compact Vision Transformer backbone (`vit_tiny_patch16_224`) across four highly disparate tasks (MNIST, FashionMNIST, CIFAR-10, SVHN). The experiments expose critical, fundamental boundaries for linear weight-space operations and adaptive test-time adaptation:

1. **Catastrophic Representational Collapse:** Under high task and domain conflict, every single merged configuration (including Uniform, AdaMerging, and ZipMerge) collapses to the level of random guessing (10% to 14% absolute accuracy). The highly orthogonal visual domains (grayscale digits vs. grayscale clothing vs. natural objects vs. colorful street numbers) create severe representational collisions in the compact backbone.
2. **The Overfitting-Optimizer Paradox:** Unsupervised Shannon entropy minimization on a tiny calibration set overfits transductively. It successfully drives down calibration-set entropy but destroys the general, robust representations learned during expert training, driving test-set accuracy down.
3. **Prune-then-Merge (P-then-M) Outperformance:** The unoptimized, decoupled baseline, where experts are pruned prior to merging, consistently and significantly outperforms joint test-time optimization. Pre-merging pruning acts as a spatial regularizer, zeroing out small, conflicting parameter updates and reducing destructive interference in the shared backbone.
4. **The Noisy Expert Noise Injection Constraint:** Model merging is highly sensitive to the convergence quality of its input experts. A single poorly converged expert (e.g., the SVHN expert achieving 19.59% accuracy) acts as a "poison pill" in weight space, injecting massive high-frequency parameter noise that collapses the performance of other tasks.

## Extensive Analytical Ablations and Validation Studies
To isolate the Algorithmic performance of ZipMerge from the inherent limitations of linear task arithmetic under extreme domain shift, the paper includes a massive suite of rigorous analytical studies:
- **Reg-ZipMerge:** Demonstrates that introducing a structural distance penalty or functional KL divergence distillation penalty successfully mitigates transductive overfitting.
- **Low-Conflict Visual Domains (DomainNet):** Evaluates ZipMerge on a homogeneous subset of DomainNet (Clipart, Painting, Real, Infograph), proving that the ZipMerge algorithm is highly stable and functional (+74.20% joint mean accuracy at 50% sparsity) when task-vector conflicts are low.
- **High-Capacity Backbones (ViT-Base):** Scales the backbone to an 86M parameter ViT-Base, showing that increased capacity mitigates but does not completely resolve representational collapse under extreme conflict, and that ZipMerge continues to outperform post-hoc Uniform merging baselines.
- **PEFT/LoRA-Adapter Merging:** Evaluates merging on low-rank adapters (LoRA). Fusing LoRA adapters on a frozen base model improves the Uniform dense merge from 13.17% to 42.30% Joint Mean.
- **Orthogonal Procrustes SVD Alignment:** Proposes and implements a zero-data, negligible-overhead SVD rotation step that rotates independently learned LoRA adapter weight spaces into a shared coordinate system before averaging. This boosts the Uniform LoRA merge accuracy from 42.30% to **58.75%** (closing over 67.5% of the remaining expert ceiling gap) and achieves **62.10%** when co-optimized with ZipMerge (ES) at 50% sparsity.
- **Trained-from-Scratch MTL Baseline:** Contextualizes merging with joint multi-task training (74.63% Joint Mean ceiling).
- **CNN Backbone Diversity:** Evaluates ResNet-18, confirming that representational collapse and P-then-M superiority are general traits of weight-space operations across both CNNs and Transformers.
- **Statistical Significance & Seed Sensitivity:** Shows extremely tight standard deviations ($\pm$ 0.29% to $\pm$ 0.52%) across random calibration subsets, confirming that the findings are scientifically reliable.
- **Calibration Set Size Sensitivity:** Sweeps $B \in \{8, 16, 32, 64, 128\}$ images per task, demonstrating statistical convergence.
- **Continuous Sweep of global scaling factor:** Shows that scalar scaling merely trades off absolute collapse for task dilution.
- **Hybrid TIES-ZipMerge:** Applies sign-voting and parameter filtering from TIES-Merging before ZipMerge adaptation, showing complementary benefits (+16.50% joint accuracy at 50% sparsity).
- **Generative Language Models (GPT-2):** Evaluates ZipMerge on a 124M parameter GPT-2 model (English and French experts). ZipMerge (ES) achieves a perplexity joint mean of **24.50** at 50% sparsity (compared to 84.60 for Naive Uniform Merge and 38.50 for P-then-M), with a detailed qualitative generation analysis.
- **On-Device Systems Profiling:**
  - **Structured Block Pruning:** Introduces hardware-friendly structured pruning (masking entire attention heads and MLP neuron blocks). Profiles execution latency on an ARM Cortex-A76 mobile CPU, delivering a **1.89x physical speedup** (from 34.2 ms to 18.1 ms) and a **1.91x speedup** on ViT-Base (from 542.4 ms to 284.1 ms) out-of-the-box.
  - **Percentile Sorting Overhead:** Quantifies that linear-time Histogram-based Quantile Estimation and Delayed Thresholding yield up to **17.4x sorting speedups** with zero accuracy loss, bypassing sorting bottlenecks.
  - **Memory/Backpropagation Analysis:** Shows that ZipMerge (ES) reduces peak calibration VRAM from **1.45 GB** (STE) to a mere **180 MB** (ES) by bypassing activation caching, a massive **8.1x memory reduction** essential for edge deployment. At sequence length 1024 on GPT-2, ES yields a **13.2x memory savings** over STE.
  - **Joint Quantization-Pruning (INT8/INT4 PTQ):** Systematically sweeps low-bit precision under 50% sparsity on DomainNet, demonstrating that INT4 joint quantization-pruning is highly robust (71.10% simulated accuracy) and reduces weight storage by **8x**, while highlighting native compiler/NPU layouts and decompression bottlenecks.
  - **Storage-Budget Trade-offs:** Compares merged single-models vs. separate aggressively pruned experts under a fixed 1.2M parameter storage budget, providing clear guidelines on when to deploy which option.
  - **Expert Convergence Ablation:** Confirms that even when all experts are fully converged (using an 82.15% SVHN expert), representational collapse persists, proving it is a fundamental geometric boundary of weight space.

## Key Recommendations
The authors translate these physical system realities into three actionable guidelines for edge engineers:
1. **Avoid Extreme Task Disparity on Compact Backbones:** Limit merging to domain-aligned, semantically compatible tasks.
2. **Leverage Parameter-Efficient Adapters (PEFT):** Focus on merging low-rank adapters fine-tuned on top of robust, contrastive pre-trained foundation models (such as CLIP or DINOv2) to act as a "coordinate anchor."
3. **Incorporate Explicit Test-Time Regularizers:** Utilize structural distance penalties or functional distillation constraints to bound optimization.
