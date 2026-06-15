# 4. Experimental Evaluation and Empirical Check

This file provides a critical evaluation of the experimental setup, baselines, and whether the results support the paper's claims.

## Strengths of the Experimental Setup
1. **Diverse Baselines:** The paper compares against a comprehensive set of baselines: Task Arithmetic (static), unconstrained AdaMerging (first-order), AdaMerging with conventional regularizers ($L_2$ and Total Variation), RegCalMerge (state-of-the-art regularized TTA), and PolyMerge (of varying degrees, $d \in \{0, 1, 2, 3\}$).
2. **Robust Multi-Seed Evaluation:** The simulated experiments are evaluated across 15 independent random seeds (seeds 42 to 56), which is highly rigorous and helps rule out optimization flukes.
3. **Hardware Profiling:** The authors conduct a detailed, hardware-grounded profile of step latency, SRAM memory overhead, and DRAM weight-reconstruction bandwidth, which provides a realistic look at the on-device tradeoffs.

## Major Empirical Gaps and Critical Weaknesses

### 1. Heavily Reliant on Continuous Numerical Simulations
While the paper presents results for a 12-layer Vision Transformer (ViT-B/32), **these experiments are conducted entirely within an analytical, simulated loss landscape sandbox (Model I and Model II)** rather than on actual, physical Vision Transformer weights. Model II is modeled as a non-convex Rastrigin-like loss landscape calibrated with some properties of CLIP. While the simulation is continuous and multi-seeded, there is a massive difference between an analytical mathematical function (with 12 parameters) and a true high-dimensional deep learning weight space. This heavily weakens the practical significance of the simulated results.

### 2. Physical Validation is Limited to "Toy" Architectures
To anchor the simulation, the authors perform physical validations. However, these are restricted to:
* A 3-layer MLP with **108K parameters** on MNIST and FashionMNIST.
* A 5-layer CNN with **250K parameters** on MNIST, FashionMNIST, and KMNIST.

These models are tiny toy networks. The authors do not evaluate FlatMerge on actual, physical Vision Transformer weights, even though pre-trained models like CLIP ViT-B/32 (86M parameters) are widely available and can be merged and adapted on standard CPU/GPU hardware. This represents a major gap between the claimed utility (edge deployment of Vision Transformers) and the actual physical verification.

### 3. The "Adaptation-Decline" Paradox: Static Baseline Beats FlatMerge on CNN
A highly critical look at the physical 5-layer CNN results (Table 4) reveals a devastating weakness: **unadapted, uniform Task Arithmetic (static blending) outperforms FlatMerge across all noise levels, including clean conditions!**
* **At $\gamma = 0.0$ (Clean):** Task Arithmetic achieves **$58.20\%$** accuracy, while ZO-FlatMerge only achieves **$48.57\%$** (a drop of nearly $10\%$ absolute).
* **At $\gamma = 1.0$ (Moderate):** Task Arithmetic achieves **$40.67\%$**, while ZO-FlatMerge achieves **$29.20\%$** (a drop of over $11\%$ absolute).
* **At $\gamma = 2.0$ (Heavy):** Task Arithmetic achieves **$24.60\%$**, while ZO-FlatMerge achieves **$19.77\%$**.
* **At $\gamma = 3.0$ (Extreme):** Task Arithmetic achieves **$17.77\%$**, while ZO-FlatMerge achieves **$16.07\%$**.

This means that on actual, physical CNN weights, **adaptation actually hurts performance compared to doing nothing (Task Arithmetic)**. While ZO-FlatMerge is indeed much more robust than standard AdaMerging and PolyMerge (which catastrophically collapse to near random guessing, $\approx 14\% - 17\%$, due to the constant-prediction failure), it still fails to outperform the simple, static Task Arithmetic uniform baseline. 

This raises a serious question regarding the practical utility of the entire framework: **Why would an edge-device practitioner deploy an active test-time adaptation loop (requiring repeated DRAM weight reconstructions, forward-pass perturbations, and latency overhead) if they can achieve $10\% - 11\%$ higher classification accuracy by simply using a static uniform merging coefficient?** This empirical finding contradicts the claim of "pragmatic utility" on physical deployments and suggests that unsupervised test-time entropy minimization remains fundamentally sub-optimal compared to static merging on real-world convolutional weights, even when regularized with flatness-aware constraints.
