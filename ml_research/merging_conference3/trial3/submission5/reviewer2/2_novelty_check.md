# Novelty and Delta Assessment: Q-PolyMerge

## 1. Assessment of Key Novel Aspects
The primary novelty of Q-PolyMerge lies in the **combination of continuous polynomial parameterization with quantization-aware test-time model merging**. 
- While model merging (Task Arithmetic, TIES-Merging, RegMean) and test-time adaptation (AdaMerging) are known, and post-training quantization is standard, the integration of these three domains specifically to solve the on-device memory and overfitting constraints is a distinct contribution.
- The concept of restricting layer-wise parameters to a low-degree polynomial curve of normalized layer depth acts as a powerful regularizer (low-pass filter). While weight parameterization using polynomials or splines is an established mathematical tool, applying it to *merging coefficients across layers* during test-time entropy minimization under weight rounding is a creative and elegant solution to the "Overfitting-Optimizer Paradox".

## 2. The 'Delta' from Prior Work
The paper positions itself relative to several key baselines:
- **AdaMerging (Yang et al., ICLR 2024):** Traditional test-time adaptation optimizes layer-wise merging coefficients in full precision (FP16/FP32). The delta is that AdaMerging assumes a continuous weight space and suffers from severe representation and alignment noise when subsequent low-bit quantization is applied. Furthermore, unconstrained layer-wise optimization overfits on compact calibration sets.
- **Q-Merge Baseline:** An unconstrained quantization-aware merging approach that optimizes coefficients directly under the quantization operator. The delta is that Q-Merge optimizes independent layer-wise parameters (56 parameters in the ViT-Tiny setup). This high dimensionality leads to overfitting (transductive noise fitting) on tiny streams, yielding jagged, physically unstable trajectories. Q-PolyMerge reduces the parameter space by over 78% and regularizes the trajectories.
- **Static Quantization-Aware Merging (TVQ, E-PMQ, 1bit-Merging):** 
  - *TVQ (ICCV 2025)* focus on task vector compression for storage savings but does not adapt coefficients at test-time.
  - *E-PMQ (May 2026)* utilizes expert weights as anchors during post-merge quantization but is a static, offline method requiring training statistics.
  - *1bit-Merging (Feb 2025)* compresses task vectors to 1-bit with dynamic routing but does not address continuous layer-wise coefficient regularization during test-time adaptation.
  - *The Delta:* Q-PolyMerge is the first to address **on-device active SRAM reduction** and **test-time stream overfitting** by projecting coefficients onto a low-degree continuous polynomial subspace.

## 3. Characterization of Novelty
The novelty is **incremental but highly significant from a systems and pragmatic perspective**. 
- It is *incremental* because it builds directly on the mathematical formulations of AdaMerging, Straight-Through Estimation, and standard 1+1 Evolution Strategies.
- It is *significant* because it successfully bridges the gap between academic full-precision model merging and actual physical edge-hardware constraints. By constraining the search space to a 12-parameter polynomial subspace, it enables zero-order search to converge rapidly and reliably with a >95% reduction in volatile memory (SRAM), resolving a critical deployment bottleneck that previously made test-time adaptation on microcontrollers physically impossible.
- The inclusion of orthogonal Chebyshev scaling pathways and a fully-integerized operator execution blueprint shows a high degree of technical maturity and foresight regarding hardware-in-the-loop deployment.
