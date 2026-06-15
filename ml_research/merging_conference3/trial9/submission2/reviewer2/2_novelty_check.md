# 2. Novelty Check

## Key Novel Aspects
* **Dynamic Budget-Driven Ensembling Control:** The introduction of a real-time system resource coefficient $C_{\text{budget}} \in [0, 1]$ to dynamically scale the active expert capacity ($M$) and adjust the gating threshold ($\theta$) in microsecond-scale execution.
* **Early-Layer Coordinate GMM Safety Shield:** Utilizing low-dimensional cosine similarity coordinates extracted from Layer 3 task centroids to fit a 2-component diagonal Gaussian Mixture Model that flags and filters OOD inputs before downstream experts are executed.
* **Hierarchical Macro-Domain GMM Routing (HMD-GMM):** A two-level routing architecture that groups $K$ tasks into semantic macro-domains, scaling OOD filtering and expert routing to larger registries (up to $K=24$) without coordinate manifold overlap or covariance singularity.

## Delta from Prior Work
* **Parameter-Space Model Merging (TIES, DARE):** These SOTA merging methods are fundamentally static, collapsing all experts into a single set of weights offline. RB-TopM is dynamic, preserving specialized task weights and routing activations sample-by-sample, which secures up to an **8.7% accuracy margin** over static merging on heterogeneous evaluation streams.
* **Dynamic Activation Blending (SABLE, SPS-ZCA):** These techniques run all $K$ parallel expert pathways for every input query, assuming static, infinite serving resources. RB-TopM introduces dynamic resource-constrained pruning to save compute and DRAM weight transfer bandwidth.
* **Quantized Activation Blending (Q-SPS):** Q-SPS uses static, hardcoded thresholds to skip expert execution. RB-TopM dynamically adjusts capacity and pruning thresholds on-the-fly in response to real-time OS interrupts (e.g., thermal throttling, battery drain) and stream-quality variations.

## Characterization of Novelty
The novelty of this work is **moderate-to-significant** from a systems-engineering perspective but **incremental** from a theoretical perspective. 

* **Systems/Pragmatic Novelty (Significant):** The integration of real-time hardware state monitoring with activation-space gating represents a highly practical and novel bridge between TinyML systems and dynamic deep learning ensembling. The Hierarchical HMD-GMM formulation is a clever, highly scalable engineering solution to handle covariance singularities and manifold overlaps in large registries.
* **Theoretical Novelty (Incremental):** The underlying building blocks—Zero-Shot Centroid Alignment, diagonal GMMs, hard-thresholding operators, and linear interpolation—are well-established in the literature. The novelty lies in their combination and the heuristic formulation of the control loop, rather than any new fundamental learning algorithms or mathematical paradigms.
