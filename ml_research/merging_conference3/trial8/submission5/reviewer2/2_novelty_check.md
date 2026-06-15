# Intermediate Evaluation 2: Novelty and Delta Analysis

## Conceptual Novelty and Delta from Prior Art
PEAR represents a significant conceptual shift in how dynamic multi-task ensembling is performed on vision backbones. Instead of attempting to design more complex parametric gating models or sophisticated batch scheduling mechanisms, PEAR focuses on a systems-level question: *where* in the architecture should routing occur to maximize parameter efficiency, minimize latency, and preserve model capacity?

The delta between PEAR and existing state-of-the-art ensembling methods is clear and practical:

| Feature / Dimension | Static Merging (e.g., TIES, DARE) | Parametric Routers (e.g., Linear Router) | PFSR with MBH | SABLE SOTA (Late Adaptation) | **PEAR (Ours)** |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Routing Mechanism** | None (Static) | Trained Gating Network | Final Classification Heads | Mid-to-Late Activation Similarity | **Early-Layer Activation Similarity** |
| **Trainable Gating Parameters** | 0 | $O(K \times D)$ (trained on calibration) | 0 | 0 | **0 (Parameter-Free Calibration)** |
| **Adapted Network Depth** | 100% (Merged) | 100% | 100% | ~16% (Blocks 10–11 only adapted) | **100% (Blocks 0–11) or ~83% (Blocks 2–11)** |
| **Inference Latency Complexity** | $O(1)$ | $O(1)$ | $O(K)$ or heavy scheduling delay | $O(1)$ | **$O(1)$ (Single-pass parallel execution)** |
| **Vectorization Robustness ($B=1$)** | Yes (No routing) | No (**Vectorization Collapse**) | No (Requires micro-batching) | Yes | **Yes (Robust to stream heterogeneity)** |
| **Training-Serving Alignment** | Yes | Yes | Yes | Yes | **Yes (via ELFT / Early-Layer Routing)** |

---

## Technical and Algorithmic Contributions
PEAR combines several elegant, training-free algorithmic steps to establish stable, sample-wise boundaries:
1. **Zero-Shot Patch Centroids (ZPC) at Layer 0:** While previous methods align activations at late blocks (SABLE), PEAR proves that the very first projection (Layer 0, Patch Embedding) contains highly discriminative spatial/semantic representation spaces that can serve as reliable multi-task routing anchors when computed as class-wise centroids.
2. **Intra-Task Dispersion Calibration (IDC):** This is a key technical novelty. Different task manifolds exhibit vastly different representational densities (e.g., concentrated MNIST vs. highly dispersed SVHN). Dividing cosine similarities by the expected in-distribution dispersion factor ($d_k$) standardizes the similarity scales, resolving the voting bias where concentrated tasks systematically outvote more diverse ones.
3. **Adaptive Task-Specific Thresholding:** Instead of a fragile, static global Out-of-Distribution (OOD) threshold, PEAR scales the rejection boundary with each task's expected representational density ($\gamma_{\text{OOD}, k} = \eta \cdot d_k$). This elegantly resolves the security-selectivity trade-off, enabling robust OOD protection under heterogeneous noise regimes without manual hyperparameter tuning.

---

## The "Early-Layer Routing Compromise" & ELFT
A major highlight of the paper's novelty is the identifying and empirical solving of the **Global-Average-Color Routing Paradox** on real images. 
- *The Paradox:* Spatially average-pooling Layer 0 tokens is equivalent to a linear projection of global average color, which lacks semantic discriminative power in real-world datasets and causes representation bleed.
- *The Novel Solution (Early-Layer Routing Compromise):* By shifting the routing boundary slightly deeper (to Layer 1 or 2), PEAR leverages the local self-attention features and spatial structure learned in the first few blocks. This increases routing accuracy on real images from $57.81\%$ (Layer 0) to **$95.31\%$** (Layer 2).
- *Training-Serving Alignment (ELFT):* To eliminate the subtle representational mismatch that arises because early blocks are executed using unadapted base weights while fine-tuned experts were trained with active early adapters, the authors propose **Early-Layer Freezing during Training**. Freezing blocks $l < l_{\text{route}}$ during expert fine-tuning completely neutralizes this boundary discrepancy.

---

## Characterization of Novelty
The novelty of this work is **highly significant and of great practical utility**. 

Rather than proposing purely theoretical improvements or complex gating networks that require heavy training, PEAR represents an elegant systems-level design that is extremely attractive for real-world production deployment on resource-constrained edge devices. By shifting the routing operation to early layers, the framework achieves the ideal combination of **parameter-free execution, flat $O(1)$ latency, 100% (or 83%) layer adaptability, and absolute robustness to vectorized streams ($B=1$)**. The integration of Intra-Task Dispersion Calibration (IDC), Adaptive Task-Specific Thresholding, and ELFT demonstrates a deep, systems-aware understanding of the practical challenges in multi-task edge serving.
