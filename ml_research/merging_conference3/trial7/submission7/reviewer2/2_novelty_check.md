# 2. Novelty Check

## Assessment of Key Novel Aspects
The paper introduces three main novel aspects to the domain of dynamic model merging:
1. **Shifting routing to early layers with Downstream-Only Merging:** Rather than ensembling/merging full weights across all layers, ELATI frozen-routes early layers and dynamically merges only subsequent downstream weights. This effectively resolves the "two-pass latency penalty" that has plagued previous dynamic model ensembling methods like PFSR.
2. **Early-Layer Representative Mapping (ELRM):** The use of unsupervised, offline-profiled activation centroids from a tiny calibration split as projection keys. This allows similarity-based routing at early layers where downstream task-specific classification heads cannot be accessed.
3. **Downstream-Only Micro-Batch Homogenization (DO-MBH):** Restructuring the batch processing pipeline to run early layers once for the entire heterogeneous batch, partitioning at Layer 2, and merging only downstream layers sequentially.

## The 'Delta' from Prior Work
- **Static Model Merging (Task Arithmetic, TIES, DARE):** These methods average weights once offline. They suffer from static representation conflicts and lack on-the-fly sample adaptability. ELATI is dynamic and adapts to incoming sample streams.
- **Penultimate Dynamic Merging (PFSR + MBH):** PFSR extracts features at Layer 13 (penultimate), necessitating a complete, redundant forward pass of the backbone to compute coefficients before the actual prediction pass. ELATI shifts routing to Layer 2, saving 11 layers of execution in Pass 1.
- **Linear/Probing Routers:** Standard intermediate probing approaches train parametric classifiers (linear probes) on intermediate activations. ELATI uses a non-parametric, training-free centroid similarity projection that is highly robust to overfitting under data-scarcity (64 samples).

## Characterization of Novelty
The overall novelty of this paper is characterized as **incremental to moderate**.

### Major Novelty Elements:
- The concept of **downstream-only weight ensembling** based on an early activation cut-off is a clever, highly practical systems contribution. It bridges the gap between early probing/exit networks and parameter-space model ensembling.

### Limitations in Conceptual Novelty:
- **Unsupervised Centroids:** Projecting representations onto task centroids is equivalent to a classical Nearest Centroid Classifier (or prototype-based learning) applied to intermediate embeddings. While data-efficient, this is a well-established representation learning concept.
- **Early Probing:** Shifting classifiers to early layers of deep networks to extract domain or semantic properties has been extensively studied in linguistic and vision probing literature (e.g., Alain & Ollivier, 2016; Tenney et al., 2019). The technical transition to model merging is logical and straightforward rather than a paradigm shift.
- **Micro-Batch Homogenization:** The dynamic grouping mechanism is heavily derived from PFSR + MBH (Chronopoulou et al., 2023). The novelty here is strictly limiting it to the "downstream-only" tail.
