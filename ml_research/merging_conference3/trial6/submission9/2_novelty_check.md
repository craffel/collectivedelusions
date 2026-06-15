# 2. Novelty and Originality Check

## Conceptual Novelty
The paper introduces several highly interesting and conceptually fresh ideas that address genuine, practical bottlenecks in the emerging sub-field of weight-space model merging:
1. **Preservation of Spatial Token Sequences:** While almost all existing dynamic weight routers (including QWS-Merge, AdaMerging, and BSigmoid-Router) immediately collapse spatial tokens via global average pooling (GAP) before routing, CAM-Router's proposal to retain the full sequence of patch tokens is conceptually fresh and well-motivated.
2. **Trainable Task-Expert Queries via Cross-Attention:** Introducing a learned query vector per task-expert ($Q \in \mathbb{R}^{K \times D}$) to attend to spatial tokens via Multi-Head Cross-Attention (MHCA) is a novel import of established transformer attention mechanisms into the routing layer of model merging. This enables localized and spatially adaptive feature extraction.
3. **Decoupled Historical Gating (DHG):** The idea of tracking and smoothing predicted routing coefficients over a sliding window using an exponential moving average (EMA) represents an interesting heuristic for stabilizing weight trajectories and mitigating task heterogeneity conflicts in high-throughput deployments.

## Incremental Elements and Overlapping Work
Despite these strengths, several components of the framework are highly derivative of, or identical to, prior work:
1. **Independent Bounded Sigmoidal Gating:** The authors claim independent bounded gating as a core contribution to resolve the "zero-sum competitive bottleneck of traditional Softmax constraints." However, they explicitly acknowledge that this was already established by **BSigmoid-Router** in concurrent/recent literature. The mathematical scaling via $\lambda_{max} = 0.3$ is also a direct adoption of the BSigmoid-Router baseline without modification.
2. **First-Block Paradox Resolution:** Using the static base model's first transformer block to generate token features before the merging coefficients are known is a standard paradigm in layer-wise or dynamic routing frameworks (e.g., L3-Router), although the paper articulates this constraint clearly.
3. **Task Vectors and Weight Averaging:** The actual weight-space assembly mechanism ($W_{merged} = W_{base} + \sum \alpha_k V_k$) is standard task arithmetic, identical to the formulations in Task Arithmetic (Ilharco et al., 2022) and subsequent dynamic model merging papers.

## Summary of Novelty
The core novelty of the paper lies in **spatially aware, query-based routing** for parameter fusions. This represents a solid, non-trivial extension of dynamic model merging. However, the gating mechanism (Bounded Sigmoid) and the overall model assembly formula are directly borrowed from recent work, making the contribution more of a creative combination of existing blocks rather than an entirely new mathematical formulation from scratch.
