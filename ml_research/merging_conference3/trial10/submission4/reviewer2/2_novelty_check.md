# 2. Novelty Check

## Key Novel Aspects of the Paper
The primary novelty of the paper lies in addressing and resolving the mathematical and systems-level bottlenecks of **dynamic low-precision model ensembling**. While model merging and neural network quantization are individually mature fields, their intersection in the context of dynamic coordinate-space ensembling is unexplored. The key novel aspects include:
1. **Dynamic Low-Precision Routing Stabilization:** First-of-its-kind application of quantization-aware optimization (via STE) and error-feedback systems (EF-Smooth and AEF) to stabilize dynamic routing parameters and intermediate representation vectors on discrete integer grids.
2. **First-Order Noise-Shaping for Ensembling Paths:** Modeling the rounding error of ensembling coefficients as a digital filter problem. Specifically, EF-Smooth is formalized as a first-order finite impulse response (FIR) high-pass noise-shaping filter that diffuses rounding errors across layers.
3. **Activation Error Integration (AEF):** Resolving the "Small-Step Quantization Bottleneck" through a closed-loop error integration system. This is backed by a formal mathematical proof (Theorem 3.1) showing that AEF bounds the cumulative activation rounding error to a constant value ($\frac{s_{\text{act}} \sqrt{D}}{2}$) independent of the network depth $H$.
4. **Permutation-Invariant Single-Pass Apportionment (PI-SPA):** Applying Hamilton's apportionment method to project continuous weights onto a discrete 4-bit simplex grid ($\sum q_k = 15$) on specialized edge vector hardware, bypassing the costly $O(K \log K)$ sorting bottleneck while preserving permutation invariance.

## The 'Delta' from Prior Work
- **Delta from Static Model Merging (e.g., Model Soups, Git Re-Basin):** Static merging methods average models offline, producing a single frozen weight configuration that cannot adapt to non-stationary queries. QA-Merge enables dynamic, sample-by-sample coordinate blending during inference, which is particularly robust on streaming multi-task queries.
- **Delta from Float32 Dynamic Ensembling (e.g., SABLE, ChemMerge, Momentum-Merge):** These methods assume high-precision Float32 coordinates. Under 8-bit activation and 4-bit weight constraints, they suffer from "Quantization Collapse" (as shown in Figure 1, dropping to static uniform performance). QA-Merge provides the necessary mathematical correction layers (QCC, EF-Smooth, and AEF) to fully recover the Float32 ensembling ceiling.
- **Delta from Standard Quantization (PTQ/QAT) and STE:** Traditional QAT and STE are applied to standard linear or convolutional layers within a single network. QA-Merge applies STE to coordinate-space parametric routers, enabling end-to-end backpropagation through non-differentiable routing rounding boundaries.
- **Delta from Mixed-Precision Routing:** Unlike methods that route tokens to different bit-widths (e.g., DMR), QA-Merge routes representations across discrete expert weights while keeping the execution entirely in low-precision integer-only registers.

## Characterization of Novelty
The novelty of this work is **significant and highly creative**. 

While some component techniques are inspired by or adapted from other fields (e.g., STE is from deep learning QAT; error feedback is from digital signal processing/sigma-delta modulation; Hamilton's method is from political science apportionment), the authors do not merely apply these techniques blindly. Instead, they synthesize and tailor them specifically for coordinate-space ensembling:
- They model the ensembling trajectory mathematically, proving rigorous error bounds (Theorem 3.1).
- They design PI-SPA to make Hamilton's apportionment branchless and permutation-invariant, which is crucial for SIMD instruction pipelines.
- They establish a formal noise-shaping analysis of EF-Smooth (Appendix A.1), showing that perfect error feedback ($\beta=1.0$) yields a telescoping sum that bounds cumulative blending weight errors.

This rigorous theoretical grounding lifts the paper beyond an incremental empirical heuristic, offering a deep mathematical understanding of why the proposed feedback loops prevent representational drift.
