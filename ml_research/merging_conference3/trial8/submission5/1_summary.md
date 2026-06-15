# 1. Summary of the Paper

## Core Problem and Motivation
This paper addresses the critical systems and representational challenges of **dynamic, parameter-efficient multi-task expert model ensembling** on resource-constrained edge devices. Specifically, it focuses on orchestrating specialized Low-Rank Adaptation (LoRA) adapters on-the-fly under heterogeneous request streams.

Existing state-of-the-art non-parametric ensembling methods suffer from a severe **Early-Feature Loss Trade-Off** to resolve the **Routing Paradox** (the requirement to execute the model twice to perform routing, which doubles serving latency). For instance, SABLE uses **Late Adaptation**, freezing and leaving early layers (typically Blocks 0 to 9 out of 12) unadapted, which discards crucial task-specific features learned in those early blocks and limits representation capacity.

Conversely, classical low-data parametric routers suffer from **Vectorization Collapse** (severe accuracy degradation when deployed under sample-by-sample vectorized stream regimes with a batch size $B=1$) because their decision boundaries overfit to low-data calibration splits ($B_{\text{cal}} = 64$).

## Proposed Solution: PEAR
To resolve these issues, the paper proposes **PEAR (Patch-Embedding Activation Routing)**, a training-free, non-parametric, closed-form routing framework designed for Vision Transformers.
- **Routing Layer 0 (Patch-Embedding) / Early-Layer Routing:** PEAR extracts representations immediately from the frozen Patch Embedding layer (Layer 0) or slightly deeper layers (Layer 1 or 2, the *Early-Layer Routing Compromise*) to bypass the **Global Average Color Routing Paradox**.
- **Zero-Shot Patch Centroids (ZPC):** Computes reference coordinates for each specialized task in the Patch Embedding space offline using only 64 samples per task, requiring zero trainable parameters.
- **Unit-Norm Calibration (UNC):** Formulates similarity using cosine similarity on a unit-norm hypersphere to ensure scale invariance across diverse visual manifolds.
- **Intra-Task Dispersion Calibration (IDC):** Normalizes similarity scores by expected in-distribution calibration variance to resolve asymmetric representation densities.
- **Single-Pass Activation Blending (SPS):** Performs layer-wise, sample-specific LoRA activation blending on-the-fly across **100% of the network depth** (all layers adapted), maintaining flat $O(1)$ sequential latency complexity and zero dynamic memory buffers.
- **OOD Rejection and Fallback:** Incorporates a robust Out-of-Distribution (OOD) rejection threshold $\gamma_{\text{OOD}}$ which defaults to a **Static Uniform Weight Merging Fallback** ($\alpha_{k,b} = 1/K$) or a **Hard Edge Rejection** fallback ($\alpha_{k,b}=0$) using a dedicated task-agnostic **Generalist Classification Head** to avoid logit nullification.
- **Adaptive Task-Specific Thresholding:** Scales the OOD threshold dynamically as $\gamma_{\text{OOD}, k} = \eta \cdot d_k$ (where $d_k$ is the Intra-Task Dispersion Calibration factor) to resolve the security-selectivity trade-off across task manifolds of varying dispersion densities.
- **Early-Layer Freezing during Training (ELFT):** A training-serving alignment strategy that freezes the first $l_{\text{route}}$ blocks during fine-tuning to completely eliminate representational mismatch at the routing boundary block.

## Experimental Framework
The authors evaluate PEAR under two distinct paradigms:
1. **12-Layer PyTorch Synthetic Representation Sandbox:** A representation sandbox modeling standard Vision Transformer layers with feature dimension $D=192$ under a challenging **Overlapping Subspace Layout** (where each task occupies a 96-dimensional subspace with a 64-dimensional overlap with neighboring tasks). Here, PEAR is tested under three stream configurations (Homogeneous, Heterogeneous batch, and Heterogeneous vectorized $B=1$), as well as non-linear (GeLU) and highly optimized (low-noise) regimes. SVHN is deliberately configured as a high-noise stress-test (19.68% expert ceiling) to evaluate robustness.
2. **Real-World Empirical Validation on Actual Images:** Evaluated on MNIST, Fashion-MNIST, CIFAR-10, and SVHN using a pre-trained ImageNet $\mathtt{vit\_tiny\_patch16\_224}$ backbone. Both routing accuracy and end-to-end multi-task LoRA classification are validated.

## Key Findings
- In the overlapping subspace layout, PEAR achieves a consistent **59.34%** Joint Mean accuracy across all batch and streaming configurations, outperforming SABLE SOTA (**55.30%**) by **+4.04%** absolute accuracy.
- PEAR completely eliminates Vectorization Collapse under $B=1$ vectorized streams, whereas the Linear Router drops to **52.36%**.
- Real-world experiments on actual images confirm that while pure Layer 0 routing is limited by color-based representational overlap (57.81% accuracy), routing at Layer 1 or Layer 2 (the *Early-Layer Routing Compromise*) resolves this, achieving **95.31%** routing accuracy and outperforming an explicitly trained pre-backbone CNN router (**91.02%**).
- End-to-end real-world multi-task LoRA ensembling on all four tasks across all 12 blocks of the backbone confirms that PEAR achieves **55.08%** Joint Mean accuracy (recovering the vast majority of the **66.80%** Expert Ceiling), outperforming SABLE SOTA by **+15.24%** absolute accuracy ($55.08\%$ vs. $39.84\%$) and static Uniform Merging by **+20.70%** absolute accuracy ($55.08\%$ vs. $34.38\%$).
- Activating ELFT (Early-Layer Freezing during Training) successfully aligns training and serving, allowing PEAR + ELFT to recover an outstanding **85.10%** of its corresponding Expert Ceiling (recovering 53.52% out of 62.89%).
- Integrating Adaptive Task-Specific Thresholding ($\eta = 0.85$) maintains a secure MNIST FAR of $5.47\%$ and SVHN FAR of $17.19\%$, while preserving in-distribution SVHN accuracy at $13.60\%$ (compared to $10.00\%$ under a global threshold).
- Systems analysis shows PEAR L0 adds only **3.15%** CPU latency overhead ($0.95$ ms) relative to a full backbone pass, while PEAR L2 adds a sequential timeline delay of **20.78%** ($6.26$ ms) on ViT-Tiny and **17.59%** ($36.09$ ms) on ViT-Base but enables 100% layer adaptability with zero extra computational FLOPs.
