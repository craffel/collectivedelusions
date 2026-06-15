# 2. Novelty and Relation to Prior Work

## Concept and Theoretical Novelty
The paper presents a significant and highly original conceptual leap in the field of multi-task parameter-efficient adapter ensembling. While prior non-parametric activation blending methods (such as SABLE and SPS-ZCA) suffer from the **Early-Feature Loss Trade-Off**—where they must freeze and leave early layers unadapted to avoid executing the backbone twice (the Routing Paradox)—PEAR elegantly resolves this paradox by routing inside the **first structural layer** (the frozen Patch Embedding layer, Layer 0). 

By performing routing at Layer 0, PEAR allows task-specific adapters (like LoRAs) to remain active across **100% of the network depth**. This is a highly novel perspective: rather than viewing early representations as too low-level to guide routing, PEAR shows that they are highly informative, particularly when properly calibrated.

Furthermore, the paper identifies and formalizes the **Global Average Color Routing Paradox** in real-world deployments. To resolve it, the paper proposes the **Early-Layer Routing Compromise** (shifting the routing boundary to Layer 1 or 2). This is a highly practical and intellectually honest insight: it shows that while pure Layer 0 routing works in a synthetic sandbox, real-world images require slightly deeper features (incorporating local self-attention structure) to resolve color-based representational overlap, while still leaving over 83% of the network fully adaptable with minimal latency/FLOPs trade-offs.

Crucially, the paper introduces two novel conceptual solutions to address systems and training-serving misalignments:
1. **Early-Layer Freezing during Training (ELFT):** A simple yet highly original training-serving alignment strategy. By freezing the early routing blocks during the expert adaptation training phase, ELFT completely eliminates the representational mismatch that arises when bypassing early-layer adapters during test-time routing.
2. **Adaptive Task-Specific Thresholding:** An elegant solution to the security-selectivity trade-off in Out-of-Distribution (OOD) rejection. Instead of setting a static global threshold, it scales the rejection boundary dynamically based on each task's expected representational density ($d_k$), ensuring noisier manifolds are not prematurely rejected while maintaining tight security on clean manifolds.

## Algorithmic Contributions
PEAR combines several elegant, non-parametric, closed-form algorithmic techniques:
1. **Zero-Shot Patch Centroids (ZPC):** Establishes reference task coordinates in the frozen Patch Embedding layer using a tiny, low-data calibration split ($B_{\text{cal}} = 64$) with zero training or gradient steps.
2. **Unit-Norm Calibration (UNC):** Leverages cosine similarity on a unit-norm hypersphere to ensure scale invariance across diverse visual manifolds, neutralizing representation magnitude drift.
3. **Intra-Task Dispersion Calibration (IDC):** Standardizes similarities by dividing by the expected in-distribution dispersion factor of the calibration set, resolving asymmetric task representation densities.

These components are combined into a temperature-scaled Softmax to obtain sample-specific routing weights. The simplicity of these closed-form formulations is aligned with the **Minimalist** philosophy (Occam's razor), avoiding complex parameterized gating layers or heavy scheduling algorithms.

## Delineation from Prior Work
The paper thoroughly positions PEAR in the context of three main literature branches:
- **Static Weight Merging (e.g., Task Arithmetic, TIES-Merging, DARE):** Unlike static methods that average parameters off-line and suffer from representation collapse when merging heterogeneous experts, PEAR performs dynamic, sample-wise activation ensembling at runtime. A major recent survey citation, Yang et al. (2024, ACM Computing Surveys), is integrated to position the work clearly within the broad field of model merging.
- **Parametric Routing / MoE (e.g., Linear Router):** Standard MoEs require training from scratch. Low-data parametric routers (like Linear Routers) overfit to the small calibration set and suffer from **Vectorization Collapse** under batch-independent streams ($B=1$). PEAR is parameter-free, calibration-robust, and maintains consistent performance across all stream configurations.
- **Non-Parametric Activation Blending (e.g., SABLE, SPS-ZCA, PFSR + MBH):** 
  - **SABLE SOTA** is forced into "Late Adaptation," leaving layers 0-9 unadapted to bypass the routing paradox. PEAR allows all 12 layers to be adapted, capturing early task-specific features.
  - **SPS-ZCA** restricts adapters to mid-to-late layers, creating a training-test mismatch. PEAR adapts 100% of the layers.
  - **PFSR + MBH SOTA** requires heavy scheduling (Micro-Batch Homogenization) and multiple sequential backbone passes ($O(K)$ latency), destroying real-time guarantees. PEAR executes in a single parallel forward pass with flat $O(1)$ sequential latency and zero dynamic memory buffers.

## Summary of Novelty Assessment
The novelty of this work is **excellent**. It identifies clear structural bottlenecks in existing state-of-the-art methods (the Early-Feature Loss Trade-Off and Vectorization Collapse), and proposes a minimalist, closed-form solution that resolves both. The conceptual and empirical bridge between the synthetic sandbox and real-world Vision Transformers (empirically discovering and resolving the color routing paradox, proposing and validating ELFT and Adaptive Thresholding) is highly original and adds immense value to the literature.
