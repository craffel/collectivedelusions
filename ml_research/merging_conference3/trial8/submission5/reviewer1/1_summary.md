# Intermediate Evaluation 1: Summary of the Paper

## Main Topic and Motivation
The paper addresses the challenge of **dynamic multi-task expert model ensembling** (specifically parameter-efficient adapters like LoRA) for resource-constrained edge devices. Current approaches suffer from two main limitations:
1. **The Routing Paradox / Early-Feature Loss Trade-Off:** Non-parametric methods like SABLE perform routing in deep layers, requiring the first several layers (often 10 out of 12) to remain unadapted to avoid running the backbone twice (which would double latency). This discards task-specific early features.
2. **Vectorization Collapse:** Parametric routers (e.g., linear classification routers) trained on low-data calibration splits overfit and degrade significantly under vectorized streaming regimes (batch size $B=1$), where batch-average normalization is unavailable.

## Proposed Approach: PEAR
To resolve these limitations, the paper introduces **PEAR (Patch-Embedding Activation Routing)**, a parameter-free, closed-form ensembling framework that shifts the routing operation to the base model's first structural projection layer—the frozen Patch Embedding layer (Layer 0). 

PEAR consists of the following components:
1. **Early Representation Extraction (Layer 0):** Spatially average-pooling patch tokens from the frozen Patch Embedding layer to obtain a global representation vector $z_b$.
2. **Zero-Shot Patch Centroids (ZPC):** Extracting task-specific class centroids offline from a tiny calibration set ($N_{cal} = 64$) without training any parameters.
3. **Unit-Norm Cosine Projection:** Evaluating maximum cosine similarity between query representations and task-specific class centroids to project raw features onto a unit-norm sphere.
4. **Intra-Task Dispersion Calibration (IDC):** Normalizing raw cosine similarities by an expected dispersion factor $d_k$ calculated offline from the calibration set to resolve asymmetric task manifold density.
5. **Temperature-Scaled Softmax & OOD Rejection:** Applying a soft routing weights computation using temperature-scaled softmax, with a static or adaptive Out-of-Distribution (OOD) rejection threshold leading to a uniform weight-merging fallback.
6. **Dynamic Activation Blending:** Propagating the computed routing weights through all subsequent layers of the network in a single forward pass, maintaining flat $O(1)$ sequential complexity.

## Key Findings and Claims
- **Synthetic Sandbox Evaluation:** In a simulated 12-layer PyTorch representation sandbox, PEAR achieves a Joint Mean accuracy of **59.34%** under overlapping task subspaces, outperforming SABLE SOTA (**55.30%**) by **+4.04%** and L2-regularized Linear Router (**52.36%**) by **+6.98%** in vectorized streaming ($B=1$).
- **The Global-Average-Color Routing Paradox:** Real-world experiments using a pre-trained $\mathtt{vit\_tiny\_patch16\_224}$ backbone reveal that pure Layer 0 routing acts as a global average color router and achieves only **57.81%** joint routing accuracy due to representational overlap.
- **Early-Layer Routing Compromise:** Shifting the routing boundary slightly deeper to Layer 1 or Layer 2 resolves the color routing paradox, achieving up to **95.31%** joint routing accuracy and outperforming a trained, lightweight pre-backbone CNN router (**91.02%**).
- **End-to-End Real-World LoRA Validation:** PEAR L2 ensembling achieves **55.08%** classification accuracy on a heterogeneous test set of real images, recovering **82.46%** of the **66.80%** Expert Ceiling and significantly outperforming SABLE SOTA (**39.84%**).
- **Early-Layer Freezing during Training (ELFT):** Freezing early layers during adapter fine-tuning eliminates the representational mismatch at the boundary layer, recovering **85.10%** of the corresponding Expert Ceiling.
- **Systems Latency:** Relative sequential latency delay of PEAR L1 and L2 is $11.92\%$ and $20.78\%$ on ViT-Tiny, which scales down to $9.80\%$ and $17.59\%$ on ViT-Base.

## Explicitly Claimed Contributions
1. Formal identification of the *Early-Feature Loss Trade-Off* in non-parametric routing and *Vectorization Collapse* in parametric routing under batch-independent streaming regimes ($B=1$).
2. PEAR, a training-free, non-parametric, closed-form routing framework that evaluates cosine similarity over class centroids at Layer 0.
3. Demonstration of PEAR's effectiveness and robustness across homogeneous, heterogeneous, and vectorized streams inside a 12-layer synthetic representation sandbox in PyTorch.
4. Empirical bridging of the simulation-to-real-world gap on real images from MNIST, Fashion-MNIST, CIFAR-10, and SVHN, proposing the *Early-Layer Routing Compromise* and *ELFT* to resolve semantic representation and boundary mismatch trade-offs.
