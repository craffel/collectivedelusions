# Novelty Check: Scale-Aligned Quantized Activation Blending (SA-QAB)

## 1. Key Novel Aspects
The primary novelty of SA-QAB lies in its application of **activation-space blending** to the problem of **low-bit quantized multi-task model merging**.
- Prior multi-task model merging techniques (such as weight averaging, TIES-Merging, and Q-Merge) operate strictly in parameter space. Under low-bit quantization, this causes severe activation distortion and performance collapse.
- Prior activation-space blending frameworks (such as SPS-ZCA and SABLE) operate in full-precision floating-point formats, rendering them impractical for low-power edge microcontrollers.
- SA-QAB is the first framework that adapts activation blending to a mixed-precision integer-only pipeline (INT4 base model + INT8 task adapters), resolving representation scale contraction and routing stability entirely on the integer manifold.

## 2. Delta from Prior Work
- **From Parameter-Space Merging (TIES, DARES, Q-Merge):** Instead of statically collapsing expert parameters into a single weight matrix before inference, SA-QAB keeps the base weights and expert adapters decoupled during quantization. It executes them in their native integer formats and blends their activations dynamically at runtime. This avoids weight-space interference and is completely immune to mixed-batch (heterogeneous) task collapse.
- **From High-Precision Activation Blending (SPS-ZCA, SABLE):** Prior methods assume FP16/FP32 precision. SA-QAB introduces three new components to tolerate low-bit integer constraints:
  - *Decoupled Heterogeneous Quantization (DHQ)* to aggressively compress the heavy base weights while maintaining expert precision.
  - *Quantization Scale Recovery (QSR)* to dynamically correct low-bit scale contraction on-the-fly via pre-computed ratios.
  - *Quantized Zero-Shot Centroid Alignment (Q-ZCA)* to execute routing via low-overhead integer cosine similarity.
- **From standard GMM OOD Detection:** Introduces a lightweight systems mitigation—**Zero-phase Component Analysis (ZCA) Pre-whitening**—which is fused offline into the preceding weights, allowing a simple diagonal GMM to perform as well as a full-covariance GMM on highly correlated features, with zero runtime overhead.

## 3. Characterization of Novelty
The novelty of this paper is characterized as **incremental to moderate, and heavily systems-driven**. 
- Algorithmically, the work is a synthesis of several well-established techniques:
  - Activation blending and early-stage centroid-based routing (derived directly from SABLE and SPS-ZCA).
  - Post-training uniform quantization (standard PTQ literature).
  - Out-of-distribution detection using GMMs (standard statistical literature).
  - Outlier-aware channel scaling or scaling calibration (analogous to SmoothQuant and other scale-correction methods).
  - Quantization-aware fine-tuning using Straight-Through Estimators (a classic, decades-old technique).
- The value of the paper lies not in introducing a radically new mathematical primitive, but in the **engineering integration and thorough systems co-design** required to make these disparate pieces work together on resource-constrained hardware.
- **Critical Perspective:** From an elegant, minimalist standpoint, the novelty is somewhat fragmented. Rather than addressing the root cause of quantization difficulty in merged parameter spaces with a simple, unified mathematical abstraction, the paper constructs a complex multi-stage pipeline of "patches":
  1. A mixed-bitwidth quantization strategy (DHQ) to isolate parameters.
  2. A scaling calibration step (QSR) to fix scale drift.
  3. A specialized routing metric (Q-ZCA) with fast fixed-point approximations to avoid microcontroller divisions.
  4. An auxiliary GMM to reject OOD samples.
  5. A ZCA pre-whitening matrix fused into the weights to fix the GMM's diagonal assumption.
  6. A 5-epoch QAT training phase to recover the accuracy lost to quantization.
  While each piece is well-reasoned, the cumulative complexity of this stack is substantial, which somewhat detracts from the elegance of "training-free, forward-only" model merging.
