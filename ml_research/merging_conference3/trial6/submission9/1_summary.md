# 1. Summary of the Paper

## Overview
The paper proposes **Cross-Attention Multi-Expert Routing (CAM-Router)**, a classical, spatially aware dynamic weight routing framework designed for multi-task model merging on compact Vision Transformer (ViT) backbones. The paper aims to address three fundamental limitations of existing dynamic model-merging routers:
1. **Vulnerability to Spatial Occlusion:** Conventional routers collapse spatial token dimensions via immediate global average pooling (GAP) before mapping to task routing weights. This discards localized spatial feature cues, making routing decisions highly vulnerable to background noise, occlusions, and crop corruptions.
2. **Task Heterogeneity Collapse:** Under mixed-task batching, average pooling token features across heterogeneous samples (e.g., MNIST and SVHN in the same batch) averages out task-specific representation signatures, causing the predicted coefficients to collapse back to uniform compromises.
3. **Softmax-driven zero-sum competition:** Traditional methods use Softmax normalization to scale routing coefficients, forcing an artificial zero-sum competitive bottleneck among specialized task-experts.

## Proposed Methodology (CAM-Router)
To overcome these limitations, CAM-Router retains the un-pooled sequence of patch tokens from the early layers of the transformer backbone (specifically, after the first self-attention block, utilizing the static pre-trained base model $W_{base}^{(1)}$ to resolve the "First-Block Paradox"). Its main components are:
1. **Trainable Task-Expert Queries ($Q \in \mathbb{R}^{K \times D}$):** Learned query embeddings (one per task expert) that serve as specialized templates for recognizing domain-specific features.
2. **Multi-Head Cross-Attention (MHCA):** The task queries attend directly to the spatial patch tokens, enabling the router to dynamically extract localized representations.
3. **Independent Bounded Sigmoidal Gating:** Replaces Softmax with independent Sigmoids scaled by a maximum bound $\lambda_{max} = 0.3$ to allow the non-competitive concurrent activation of multiple experts.
4. **Decoupled Historical Gating (DHG):** For batched inference, it computes per-sample routing coefficients and tracks them via an exponential moving average (EMA) over historical steps to insulate individual routing weights from unrelated concurrent task images in a batch.
5. **GPU Acceleration Proposals:** The authors outline conceptual engineering designs, including coefficient quantization caching and custom Triton kernels for fused gated matrix multiplication.

## Key Claims & Findings
- **High Joint Performance:** On a 14-layer compact Vision Transformer (`vit_tiny_patch16_224` backbone) across four tasks (MNIST, FashionMNIST, CIFAR-10, SVHN), CAM-Router achieves a Joint Mean Accuracy of **53.07%**, representing a **+11.10%** absolute improvement over the Static Uniform baseline (41.97%) and significantly outperforming standard average-pooling routers such as BSigmoid-Router (28.70%) and QWS-Merge (24.90%).
- **Occlusion Robustness:** Under systematic spatial occlusions (up to 80% patch masking), CAM-Router remains robust, maintaining a stable Joint Mean Accuracy of **50.57%** (vs. BSigmoid-Router which remains at 28.70%).
- **Task Heterogeneity Resilience:** Under large mixed-task batch sizes (up to $B=256$), CAM-Router maintains a Joint Mean Accuracy of **54.30%**, while average-pooling-based BSigmoid-Router collapses to 28.12%.
- **Lightweight Overhead:** CAM-Router adds only **~0.15M parameters** (approx. 2.61% overhead) to the 5.7M parameter ViT-Tiny backbone.
