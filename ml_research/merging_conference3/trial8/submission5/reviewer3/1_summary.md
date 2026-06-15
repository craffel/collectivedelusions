# Paper Summary: PEAR (Patch-Embedding Activation Routing)

## Main Topic and Problem Statement
The paper addresses the challenge of serving specialized multi-task expert adapters (e.g., Low-Rank Adaptation (LoRA) experts) dynamically and efficiently on resource-constrained edge devices. 
Specifically, the authors target two main limitations of existing model-merging and routing paradigms:
1. **The Routing Paradox and Early-Feature Loss Trade-Off:** Existing non-parametric ensembling methods (like SABLE) compute routing weights in the later layers of the network to avoid executing the base model multiple times. To resolve this "Routing Paradox," they enforce "Late Adaptation," freezing and leaving early blocks (e.g., blocks 0–9 in a 12-block ViT) completely unadapted. This discards early-layer task-specific features and restricts model capacity.
2. **Vectorization Collapse of Parametric Routers:** Lightweight parametric gating networks (e.g., Linear Routers) trained on small calibration sets suffer from severe overfitting. Under batch-independent stream regimes where batch size $B=1$, their decision boundaries collapse, degrading performance to the level of static uniform weight merging.

---

## Proposed Approach (PEAR)
PEAR (Patch-Embedding Activation Routing) is proposed as a parameter-free, closed-form ensembling framework that performs dynamic, sample-wise ensembling at the frozen Patch Embedding layer (Layer 0) of a Vision Transformer. It enables specialized expert adapters to remain active across 100% of the network depth (all blocks adapted) with flat $O(1)$ sequential depth complexity.

The PEAR framework consists of the following mathematical and algorithmic steps:
1. **Early Representation Extraction (Layer 0):** Spatially average-pooling patch tokens immediately after the frozen Patch-Embedding layer to obtain a translation-invariant representation vector $z_b = \frac{1}{N} \sum_{i=1}^N Z_{b, i, :} \in \mathbb{R}^D$ at virtual zero extra compute.
2. **Zero-Shot Patch Centroids (ZPC):** Establishing reference coordinates for task domains in the Layer 0 space by calculating the class-wise means of a tiny, offline calibration set: $\mu_{k, c} = \frac{1}{|\mathcal{C}_{k, c}|} \sum_{s \in \mathcal{C}_{k, c}} z_s^{(0)}$.
3. **Unit-Norm Cosine Similarity:** Evaluating the raw similarity of a query $z_b$ relative to task expert $k$ on a unit-norm hypersphere to ensure scale invariance: $s_{k, b} = \max_{j} \frac{z_b \cdot \mu_{k, j}}{\|z_b\|_2 \|\mu_{k, j}\|_2}$.
4. **Intra-Task Dispersion Calibration (IDC):** Standardizing similarity scores by dividing raw similarities by the expected in-distribution dispersion factor $d_k$ calculated during offline calibration: $u_{k, b} = s_{k, b} / d_k$.
5. **Temperature-Scaled Softmax and OOD Rejection:** Converting similarities into routing weights $\alpha_{k,b}$ via a temperature-scaled Softmax: $\alpha_{k, b} = \frac{\exp(u_{k, b}/\tau)}{\sum_j \exp(u_{j, b}/\tau)}$. If the maximum similarity is below a threshold, the system falls back to a Static Uniform Weight Merging Fallback ($\alpha_{k, b} = 1/K$) or a Hard Edge Rejection.
6. **Dynamic Activation Blending:** Propagating activations through each block $l$ while scaling LoRA expert adapters sample-wise in a single parallel pass: $h_b^{(l)} = h_b^{(l-1)} W_{\text{base}}^{(l)} + \sum_{k=1}^K \alpha_{k, b} (h_b^{(l-1)} A_k^{(l)} B_k^{(l)})$.

---

## Key Findings and Claims
* **Overcoming Vectorization Collapse:** Under true batch-independent streams ($B=1$), PEAR maintains a consistent Joint Mean accuracy of **59.34%** in the synthetic sandbox, entirely avoiding the collapse of Linear Routers (which drop to 52.36%).
* **Preserving Capacity via Full-Depth Adaptability:** By routing at Layer 0, PEAR allows experts to adapt all 12 blocks, outperforming the late-adaptation SABLE SOTA (which adapts only 2 blocks) by **+4.04%** absolute accuracy.
* **Resolving the Global-Average-Color Routing Paradox:** On actual real-world images from MNIST, Fashion-MNIST, CIFAR-10, and SVHN, standard PEAR L0 acts as a color router (yielding 57.81% routing accuracy). However, shifting the routing boundary to Layer 1 or Layer 2 (the *Early-Layer Routing Compromise*) resolves this, achieving up to **95.31%** real-world routing accuracy, outperforming an explicitly trained pre-backbone CNN router (91.02%).
* **End-to-End Real-World LoRA Validation:** On real images using a pre-trained ImageNet $\mathtt{vit\_tiny\_patch16\_224}$ backbone, PEAR recovers **82.46%** of the Expert Ceiling in standard training, and **85.10%** of the ceiling under Early-Layer Freezing during Training (ELFT), outperforming SABLE by up to **+15.24%** absolute accuracy.
* **Flat $O(1)$ Sequential Complexity:** Computing ensembling weights at Layer 0 ensures constant sequential depth complexity, bypassing the linear latency penalties of sequential execution baselines.

---

## Explicitly Claimed Contributions
1. **Identification of Trade-offs:** Formalizing the *Early-Feature Loss Trade-Off* in late-adaptation non-parametric routers and *Vectorization Collapse* in low-data parametric routing.
2. **The PEAR Framework:** A training-free, non-parametric, closed-form ensembling framework routing in the Patch Embedding layer (Layer 0).
3. **Sandbox Validation:** Demonstrating SOTA dynamic ensembling across homogeneous, heterogeneous, and vectorized streams in a high-fidelity 12-layer PyTorch representation sandbox.
4. **Real-World and Systems Bridge:** Evaluating PEAR on actual images using a pre-trained $\mathtt{vit\_tiny\_patch16\_224}$ backbone. Establishing the *Early-Layer Routing Compromise* to resolve the color routing paradox, proposing *ELFT* to mitigate representational boundary mismatch, and validating $O(1)$ execution speeds with detailed latency and hardware scaling analysis.
