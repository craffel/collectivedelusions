# 2. Novelty and Delta from Prior Work

## Key Novel Aspects
The primary novelty of OmniMerge lies in identifying and addressing the **generalization gap of merged model coefficients across heterogeneous hardware quantization standards**. 
Specifically:
- **Identifying Cross-Schema Performance Degradation:** While prior work on quantization-aware merging recognized that quantization introduces discretization errors, they assumed a single, static target hardware operator. This paper is the first to systematically investigate and identify the boundary-overfitting issue that occurs when coefficients optimized under one operator are deployed on mismatched compilers.
- **Multi-Schema Stochastic Co-Optimization (SOS):** Instead of optimizing blending coefficients for a single target schema, OmniMerge stochastically samples active quantization operators at each step of test-time optimization. This is a novel application of stochastic operator sampling as a parameter-space regularizer for model merging.
- **Scale and Zero-Point Noise Perturbation (SZNP):** OmniMerge injects multiplicative and additive Gaussian noise directly into the scale and zero-point parameters of the simulated quantization grid. While noise injection is widely used in other areas of deep learning, applying it to smooth the rugged, non-differentiable loss landscape of the merging coefficients under post-training quantization is a clever and effective adaptation.

## Delta from Prior Work
The paper positions OmniMerge in the context of three main areas:
1. **Unquantized Model Merging (e.g., Task Arithmetic, Model Soups, AdaMerging, TIES-Merging, DARE):**
   - *Delta:* These methods operate entirely in high precision (FP32/FP16) and completely ignore the severe discretization errors introduced by post-training quantization.
2. **Quantization-Aware Model Merging (e.g., Q-Merge, ZipMerge, RegCalMerge):**
   - *Delta:* These existing frameworks optimize blending coefficients under a single, static quantization operator (typically Symmetric Per-Channel) using STE. OmniMerge is the first to demonstrate that this single-schema optimization induces catastrophic cross-schema overfitting, and it explicitly resolves this by co-optimizing across multiple schemas stochastically.
3. **Post-Training Quantization (PTQ) and Test-Time Adaptation (TTA) (e.g., AdaRound, BRECQ, Tent, EATA):**
   - *Delta:* While standard TTA methods (like Tent) adapt model parameters to unseen test streams by minimizing entropy, they do not consider model merging coefficients or downstream hardware-quantization mismatch. OmniMerge adapts the ensembling coefficients rather than backbone weights, and integrates multi-schema co-optimization to guarantee hardware-invariant robustness.

## Characterization of Novelty
The novelty of this paper is best characterized as **incremental but highly significant and pragmatic**. 

- **Pragmatic Insight:** The core insight—that edge-deployment hardware heterogeneity makes single-schema optimization brittle—is highly practical and of great value to MLOps practitioners.
- **Methodological Synthesis:** The individual components of the methodology are not entirely new to machine learning:
  - Stochastic Operator Sampling is structurally equivalent to stochastic dropout or random data/operator augmentation during training.
  - Scale/Zero-point Noise Perturbation is an adaptation of jittering/variational noise injection to smooth discontinuous loss landscapes.
  - Test-time entropy minimization combined with consensus regularization is adapted from TTA (Tent) and AdaMerging.
- **Value of Synthesis:** However, the synthesis of these techniques into a unified, training-free, and metadata-free co-optimization framework for model merging is a highly creative, well-motivated, and elegant solution. The proposed method successfully solves a real-world edge deployment constraint without requiring expensive retraining, hardware metadata, or inference-time overhead.
