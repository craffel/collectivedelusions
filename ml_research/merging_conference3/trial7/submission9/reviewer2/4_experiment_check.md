# Experimental Evaluation Check: SABLE

## 1. Evaluation of Datasets and Scale
The primary empirical evaluations are conducted in two main settings:
- **Synthetic 14-layer Coordinate Sandbox:** This is a fully synthetic coordinate-space simulation. While mathematically controlled, it does not represent real-world feature distributions, covariate shifts, or natural neural network non-linearities.
- **Physical Image Classification Experiments:** 
  - CNN trained from scratch (3 layers, 10 epochs).
  - Deep MLP trained from scratch (4 layers, 6 epochs).
  - Frozen ResNet-18 feature extraction followed by a 2-layer MLP head.
  - Crucially, all three physical experiments are restricted **exclusively to grayscale MNIST and FashionMNIST** (with 1,000 training samples per task).

**Critical Critique:** MNIST and FashionMNIST are extremely low-resolution, low-complexity toy datasets that have been saturated in the literature for decades. Evaluating a new model-merging technique on these datasets is insufficient for modern machine learning standards. The paper lacks evaluations on full-color, high-resolution natural image datasets (such as CIFAR-100, ImageNet, or VTAB) or modern natural language processing tasks (such as GLUE). While the authors include "Real-World Actionable Blueprints" for ViT-B/16 on VTAB and LLaMA-3-8B on instruction tuning, these are purely conceptual blueprints and lack empirical results. The actual empirical scale is very limited.

## 2. Evaluation of Backbones
The neural backbones used in the physical experiments are extremely small:
- A 3-layer CNN (2 conv layers, 1 dense layer).
- A 4-layer MLP.
- A 2-layer MLP head on top of frozen ResNet-18 features.

There are no experiments fine-tuning or adapting standard-scale, modern deep architectures (such as full ResNets, Vision Transformers, or modern LLMs) where representational alignment and activation scale mismatches are most prominent. This restricts the generalizability of SABLE to modern deep learning workflows.

## 3. Evaluation of Baselines
The baselines evaluated include:
- **Expert Ceiling:** The oracle upper-bound.
- **Uniform Merging:** Static coordinate averaging.
- **Linear Router (Unreg):** A weak parametric baseline trained on only 64 samples.
- **PFSR (No MBH) and PFSR + MBH:** Parameter-space subspace routing.

**Critical Critique:** 
- The comparison against the Linear Router is unfair and represents a "strawman" because the parametric router was trained on an extremely small calibration split (64 samples) with no apparent regularization.
- The paper does not compare SABLE to other PEFT-specific ensembling or merging baselines (e.g., LoraHub, MoE-Adapters, or other dynamic adapter routing techniques). The authors argue that LoraHub is static and MoE-Adapters requires heavy training, but they should still include them as empirical baselines (even if trained on the same data) to demonstrate SABLE's relative performance.

## 4. Analysis of Whether Results Support the Claims
The empirical results partially support the core claims, but introduce notable caveats:

### A. Perfect Robustness to Heterogeneity (0.00% Collapse)
- **Claim:** SABLE is natively immune to heterogeneity collapse.
- **Support:** The results in Table 1 (CNN), Table 3 (ResNet-18), and Table 5 (Sandbox) show identical joint mean accuracies under both homogeneous and heterogeneous streams, proving this claim mathematically and empirically.

### B. Superiority over Systems-Centric Scheduling (MBH)
- **Claim:** SABLE outperforms the complex PFSR+MBH systems pipeline in heterogeneous streams while stripping away stateful wrappers.
- **Support:** In Table 5 (Sandbox), SABLE Late Adaptation achieves **68.10%** vs PFSR+MBH's **67.20%** (+0.90% margin). In Table 1, SABLE Soft ($r=10, M=2$) [Support 16] achieves **69.30%** vs collapsing PFSR's **49.00%**. 
- **Caveat:** SABLE still lags behind the **Expert Ceiling** by a wide margin in all physical evaluations:
  - CNN: SABLE Soft [Support 16] (69.30%) vs. Expert Ceiling (78.40%) — **a 9.10% gap**. SABLE Zero-Data (63.50%) vs. Expert Ceiling (78.40%) — **a 14.90% gap**.
  - Deep MLP: SABLE Soft Single-Pass (65.20%) vs. Expert Ceiling (74.00%) — **an 8.80% gap**.
  - ResNet-18: SABLE Hybrid [Support 16] at $r=16$ (69.30%) vs. Expert Ceiling (74.80%) — **a 5.50% gap**. SABLE Hybrid [Refined Zero] at $r=16$ (61.60%) vs. Expert Ceiling (74.80%) — **a 13.20% gap**.
  This indicates that while SABLE outperforms collapsing parameter-space routing, it still loses a substantial amount of task specialization compared to a true expert model.

### C. Wall-clock Serving Latency and Memory Advantages
- **Claim:** SABLE achieves a 6.8$\times$ latency reduction (12.4 ms vs 84.6 ms) and 36.4% memory savings compared to MBH.
- **Support:** These numbers are reported on an NVIDIA A100 GPU.
- **Caveat:** The authors note in Section 4.5 that naive sequential PyTorch loops would introduce significant CUDA kernel launch overhead that dominates wall-clock time and negates these savings in practice. They state that production deployments must leverage specialized serving frameworks (such as S-LoRA or Punica) to vectorize adapter computations. However, they do not clarify if their own benchmark of 12.4 ms was obtained using Punica/S-LoRA, or if they used a naive loop and somehow bypassed the overhead, or if these are simulated/theoretical wall-clock times. This ambiguity undermines the credibility of the reported benchmarks.
