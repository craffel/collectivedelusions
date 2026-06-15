# Assessment of Novelty and Delta

## Assessment of Key Novel Aspects
The paper introduces several distinct system and algorithmic combinations for modular dynamic PEFT serving:
- **Sample-Wise Activation-Space Blending (SPS) in a Single Pass:** Blending active adapter activations sample-wise on-the-fly to serve mixed-task batches in a single forward pass, rather than splitting the batch or statically merging weights in weight-space.
- **Early-Layer Centroid Routing (ZCA):** Routing queries based on early-layer representation centroids (Layer 3) instead of final-layer classification heads or penultimate features.
- **Intra-Task Dispersion Calibration (IDC):** Calibrating cosine similarity coordinates by scaling them inversely by the expected in-distribution similarity to address varying task manifold densities.
- **Low-Dimensional GMM Shield:** Using GMMs fit over low-dimensional routing coordinates for out-of-distribution detection, rather than using high-dimensional representation features or auxiliary networks.

## The 'Delta' from Prior Work
The paper positions itself relative to several categories of prior work:
1. **Static Weight-Space Model Merging (e.g., Task Arithmetic, TIES, DARE):** The delta is that SPS-ZCA is dynamic, adapting to mixed-task streams sample-by-sample without causing "heterogeneity collapse" (degradation due to conflict in static parameter compromises).
2. **Dynamic Merging / MoE Routing (e.g., PFSR, MBH, AdapterFusion):** 
   - Compared to **MBH**, which splits batches on-the-fly and executes them sequentially (scaling compute/memory cost $O(K)$), SPS-ZCA processes everything in a single parallel pass ($O(1)$) through activation blending.
   - Compared to **PFSR**, which uses cosine similarity against task-specific classification heads in the late-stage penultimate space, SPS-ZCA uses early-stage (Layer 3) task-centroid projections, resolving the temporal circular dependency (routing paradox) and remaining robust to domain/vocabulary shifts.
3. **Systems-Level Dynamic Serving (e.g., S-LoRA, Punica):** Unlike these heavy scheduling and paging frameworks for server-class GPU clusters, SPS-ZCA is training-free, compiler-friendly, and operates purely in activation space inside shared neural layers, making it suitable for edge devices.

## Characterization of Novelty
The overall novelty is **moderate to significant**, primarily arising from a creative, systems-ML co-designed combination of existing concepts:
- *Activation blending* itself is not entirely new (conceptually related to AdapterFusion or Mixture-of-Experts), but executing it sample-wise inside standard LoRA layers without retraining is a highly practical execution pattern.
- *Early-layer routing* is a clever, simple way to resolve the temporal circular dependency of layer-by-layer dynamic adapters.
- *Calibration techniques (UNC, IDC)* and *Coordinate GMM* represent incremental but valuable additions that systematically address common failure modes (scale drift, asymmetric manifold variance, OOD noise) in nearest-centroid systems.

However, from a **theoretical perspective**, the novelty is somewhat limited. The paper does not propose fundamentally new mathematical concepts or learning paradigms. Instead, it adapts standard geometric operations (cosine similarity, Softmax, Gaussian Mixture Models, vector normalization) and wraps them in a Systems-ML engineering framework. The contribution is primarily a pragmatic and system-level advancement rather than a foundational theoretical breakthrough.
