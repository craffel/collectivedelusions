# Intermediate Evaluation 1: Paper Summary

## Main Topic and Practical Context
This paper addresses a highly critical system-level challenge in multi-task edge serving: how to dynamically orchestrate and ensemble task-specific, parameter-efficient adapters (such as Low-Rank Adaptation, or LoRA) on a sample-by-sample basis under heterogeneous, vectorized query streams (batch size $B=1$). In practical edge deployments, serving separate fine-tuned models for different tasks is memory-prohibitive. Dynamic ensembling frameworks resolve this by combining specialized adapters on the fly. However, existing methods suffer from a severe architectural trade-off:
- **The Routing Paradox and Early-Feature Loss:** To route an input, current non-parametric ensembling methods (e.g., SABLE) evaluate representations in deeper layers. This forces them to keep the early layers (up to 10 out of 12 blocks) completely unadapted ("Late Adaptation") to avoid running the backbone twice, which discards crucial task-specific features learned early on.
- **Vectorization Collapse of Parametric Routers:** Traditional parametric gating networks (e.g., linear classification routers) trained on low-data calibration splits fail when deployed under sample-by-sample streams ($B=1$) because they overfit to the calibration data and depend on batch-average objectives.

To resolve these practical serving and efficiency bottlenecks, the paper introduces **PEAR (Patch-Embedding Activation Routing)**, a strictly training-free, parameter-free, closed-form routing framework that evaluates routing coefficients at the very first projection layer—the frozen Patch Embedding layer (Layer 0)—or slightly deeper (Layer 1 or 2) using an "Early-Layer Routing Compromise."

---

## Technical Approach
PEAR operates dynamically and non-parametrically through three core calibration phases:
1. **Zero-Shot Patch Centroids (ZPC):** Extracts class-wise reference anchors in the Layer 0 (or early layer) representational space offline from a tiny calibration split ($B_{\text{cal}} = 64$ samples per task), introducing zero trainable parameters.
2. **Unit-Norm Cosine Projection:** Computes the cosine similarity of incoming query representations against these ZPC anchors on a unit-norm hypersphere, ensuring strict scale invariance.
3. **Intra-Task Dispersion Calibration (IDC):** Normalizes raw similarity scores by dividing by the expected in-distribution calibration similarity factor, resolving representational density and variance asymmetries across different task manifolds.

The resulting calibrated similarity scores are passed through a temperature-scaled Softmax to compute sample-specific routing weights. These weights scale and blend the activations of the parallel specialized LoRA paths across 100% of the transformer layers (or blocks $l \ge l_{\text{route}}$) in a single parallel forward pass, achieving flat $O(1)$ latency complexity with zero dynamic memory buffering.

To address the **Global-Average-Color Routing Paradox**—the mathematical reality that spatially average-pooling Layer 0 tokens is equivalent to a linear projection of global average color/brightness, leading to representation bleed in real-world semantic datasets—the authors propose the **Early-Layer Routing Compromise**. This shifts the routing boundary slightly deeper (to Layer 1 or 2), capturing local self-attention features and spatial structure. Furthermore, they align the training and serving paths using **Early-Layer Freezing during Training (ELFT)**, freezing the early blocks used for routing to prevent representational mismatch.

---

## Key Findings and Claims
The paper's claims are backed by rigorous evaluations both in a controlled 12-layer synthetic representation sandbox in PyTorch and on actual real-world images using a pre-trained $\mathtt{vit\_tiny\_patch16\_224}$ backbone:
- **Elimination of Vectorization Collapse:** PEAR maintains stable, sample-wise ensembling performance under vectorized streams ($B=1$), where parametric linear routers collapse (achieving $59.34\%$ vs. $52.36\%$ Joint Mean accuracy in the synthetic sandbox).
- **Resolution of Early-Feature Loss Trade-Off:** By ensembling early and enabling adaptation across 100% of the network depth, PEAR outperforms SABLE SOTA by **+4.04%** absolute accuracy under overlapping task manifolds ($59.34\%$ vs. $55.30\%$) and by **+1.74%** under highly optimized expert regimes ($96.10\%$ vs. $94.36\%$).
- **Real-World ViT Validation:** Real-world experiments on MNIST, Fashion-MNIST, CIFAR-10, and SVHN confirm that PEAR L0 suffers from the Color Routing Paradox ($57.81\%$ accuracy). However, shifting routing to Layer 1 or 2 (Early-Layer Routing Compromise) resolves it, reaching **$95.31\%$** real-world routing accuracy and outperforming an explicitly trained parametric Tiny CNN router ($91.02\%$) while introducing zero trainable parameters.
- **End-to-End Adapter Ensembling on Real Images:** End-to-end LoRA ensembling on actual images shows PEAR achieves **$55.08\%$** Joint Mean accuracy, outperforming SABLE SOTA ($39.84\%$) by **$+15.24\%$** and Static Uniform Merging ($34.38\%$) by **$+20.70\%$**. Incorporating ELFT recovers **$85.10\%$** of the corresponding expert ceiling.
- **Flat $O(1)$ Serving Latency:** Systems-level profiling confirms PEAR maintains a constant execution latency as the number of experts $K$ increases, with PEAR L2 incurring only a minor sequential latency delay of $6.26$ ms (or $20.7\%$ relative overhead on ViT-Tiny, which scales down to $17.59\%$ on ViT-Base) before the routing decision is finalized. Crucially, the cached early-layer representations are fully re-used, resulting in virtually zero extra computational FLOPs overhead.

---

## Explicitly Claimed Contributions
1. **Conceptual Formalization:** Formalizes the Early-Feature Loss Trade-Off in late-adaptation non-parametric ensembling and Vectorization Collapse in low-data parametric routing.
2. **Framework Design:** Introduces PEAR, a strictly parameter-free, training-free, closed-form routing framework that dynamically blends specialized adapters.
3. **High-Fidelity Sandbox Validation:** Demonstrates state-of-the-art ensembling performance across homogeneous and heterogeneous stream configurations (with $B=256$ and $B=1$) inside a PyTorch representation sandbox under overlapping task layouts.
4. **Real-World Deployment Proof:** Successfully bridges the simulation-to-real-world gap on real-world datasets with a pre-trained ViT, identifying and resolving the Global-Average-Color Routing Paradox through the Early-Layer Routing Compromise and ELFT, and profiling latency and compute overheads.
