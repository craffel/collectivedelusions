# Novelty and Delta Check: SABLE

## 1. Key Novel Aspects of the Work
SABLE introduces several design choices to test-time dynamic model merging:
- **Activation-Space Merging via the Distributive Law:** While previous test-time merging techniques (e.g., PFSR) operate in parameter space ($W_{\text{merged}} = W_{\text{base}} + \sum \alpha_k V_k$), SABLE shifts ensembling to activation space ($Y = X W_{\text{base}} + \sum \alpha_k X V_k$). This enables per-sample ensembling coefficients within a batch, natively preventing batch-level averaging collapse.
- **Mid-Layer Routing & Late Adaptation:** Solves the hierarchical mismatch between low-level representations and final-layer class heads (the "Representational Alignment Paradox") by leaving early layers unadapted and computing cosine similarity routing from intermediate base-network activations on-the-fly.
- **Refined Zero-Data Centroids:** Improves data-free routing by applying L2-normalization to classification weight vectors before averaging, mathematically preventing vector cancellation in post-hoc centroid construction.
- **Layer-Dependent Hybrid-Rank Protocol:** Combines strict low-rank adapters in high-dimensional hidden layers with full-precision final output layers to bypass bottleneck capacity limits while maintaining global parameter efficiency.

## 2. 'Delta' from Prior Work
- **Delta from Parameter-Space Routers (PFSR, etc.):** Traditional parameter-space routers compute a single set of merged weights for the entire batch. In mixed-task streams, they must average routing coefficients across the batch dimension, which causes "heterogeneity collapse." SABLE's ensembling is computed sample-wise during the activation-space forward pass, making it natively immune to batch heterogeneity.
- **Delta from Systems scheduling wrappers (Micro-Batch Homogenization, MBH):** MBH addresses heterogeneity collapse by intercepting query streams, dynamically buffering queries, sorting them, and partitioning them into homogeneous micro-batches. SABLE completely eliminates this systems-level scheduling wrapper, replacing it with a pure, stateless mathematical formulation in activation space. This removes queue latency, state tracking, and buffering overhead.
- **Delta from PEFT-specific ensembling (LoraHub, MoE-Adapters):** LoraHub is static at test-time and requires target-task calibration splits. MoE-Adapters trains a parametric routing network, requiring heavy multi-task training and showing sensitivity to out-of-distribution queries. SABLE is completely non-parametric, requires zero training or calibration data, and dynamically adapts on-the-fly using frozen parameters.

## 3. Characterization of Novelty
The novelty of SABLE can be characterized as a **highly pragmatic, creative, and conceptually elegant combination of existing ideas** (PEFT/LoRA, non-parametric coordinate projection, and activation-space ensembling) to resolve a major practical system bottleneck (heterogeneity collapse).

Mathematically, the core of SABLE relies on the distributive property of matrix multiplication ($X(W_{\text{base}} + \sum \alpha_k V_k) = X W_{\text{base}} + \sum \alpha_k (X V_k)$). This is a well-known linear algebraic identity and is conceptually straightforward. However, the true significance of the novelty lies in the system-level insight and the rigorous structural design to make this ensembling feasible, fast, and robust under realistic, uncalibrated, and real-time streaming constraints. Specifically:
- Recognizing and resolving the **Representational Alignment Paradox** via Late Adaptation.
- Defining and addressing the **Capacity Bottleneck** of low-dimensional layers via the Hybrid-Rank Protocol and identifying the **Low-Rank Regularization Paradox**.
- Devising a highly principled, data-free **Refined Zero-Data Centroids** mechanism to make the routing completely autonomous.

Thus, while the mathematical foundation is a direct application of the distributive law, the comprehensive architectural framework and practical insights represent a **significant, high-impact advancement** for multi-tenant serving of parameter-efficient models.
