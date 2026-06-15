# 2. Novelty and Literature Context Check

## Core Novelty Analysis
The main novelty of OmniMerge lies in the introduction of a **stochastic co-optimization** scheme during the test-time adaptation of model merging coefficients. Specifically, the framework introduces:
1. **Stochastic Operator Sampling (SOS):** Uniformly sampling an active PTQ operator from a discrete hardware-relevant pool $\mathcal{Q}$ at each optimization step.
2. **Scale and Zero-Point Noise Perturbation (SZNP):** Dynamic injection of Gaussian noise into the scale and zero-point computation during calibration.

While both model merging and post-training quantization are active fields, the intersection—ensuring that a single merged checkpoint generalizes across *heterogeneous* and *mismatched* post-training quantization backends (e.g., symmetric vs. asymmetric, per-channel vs. per-tensor)—is a highly practical and novel niche.

## The 'Delta' from Prior Work
The paper positions itself relative to several key baselines:
- **AdaMerging (Yang et al., 2023):** Merges weights in FP16/FP32 but suffers under subsequent PTQ discretization.
- **Q-Merge (Anonymous, ICML 2024):** Optimizes merging coefficients directly under a simulated quantization operator using the Straight-Through Estimator (STE). However, Q-Merge is optimized under a single static operator (typically Symmetric Per-Channel).
- **ZipMerge / RegCalMerge (Anonymous, 2025):** Focus on joint pruning/merging or calibration, but still utilize a single static quantization operator during optimization.

OmniMerge’s key delta is the integration of **SOS** and **SZNP** during coefficient search. Instead of fitting coefficients to a single discrete rounding grid, OmniMerge forces the coefficients to be robust to a variety of grids and noise patterns, producing hardware-invariant merging weights.

## Scholarly Attribution & Literature Representation
Applying a rigorous scholarly perspective, there is a minor attribution and framing tension in the paper:
1. **Vulnerability Demonstration vs. Resolution:** 
   - In the **Introduction** (Section 1), the authors state: *"In this work, we demonstrate that optimizing merging coefficients strictly under a single quantization operator (as done by Q-Merge) induces a phenomenon we term Cross-Schema Performance Degradation."*
   - In the **Related Work** (Section 3), the authors state: *"The robustness audit conducted in [qmergeaudit] highlighted that such single-schema optimization makes the model highly vulnerable to catastrophic cross-schema overfitting when deployed onto hardware running mismatched compilers."*
   - There is a slight overlap here. The vulnerability (and the phenomenon itself) was already audited and published by `qmergeaudit` (2025). Therefore, the primary contribution of this work is **not** the discovery of the "Cross-Schema Performance Degradation" phenomenon, but rather the **resolution** of this known bottleneck via a training-free co-optimization framework (OmniMerge). The authors should adjust their introduction to more precisely attribute the first demonstration of this vulnerability to `qmergeaudit`, thereby clarifying that their work focuses on providing the first systematic algorithmic solution to this audited gap.
2. **Prior Work in Multi-Hardware/Robust Quantization:**
   - While the paper cites standard PTQ techniques (e.g., AdaRound, BRECQ, SmoothQuant, AWQ) and test-time adaptation surveys, it could benefit from situating itself within the broader context of robust and adaptive quantization across multiple hardware accelerators (such as HAQ, HAWQ, or recent adaptive quantization frameworks like SigmaQuant). Integrating this context would enrich the scholarly framing of the paper.
   - The use of noise injection during optimization is a well-established regularizer in machine learning. While novel in the context of test-time model merging, comparing or drawing parallels to existing weight-noise or scale-noise methods in standard Quantization-Aware Training (QAT) would strengthen the connection to historical quantization literature.
