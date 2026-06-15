# Intermediate Evaluation 2: Novelty Check

## Assessment of Key Novel Aspects
The paper introduces several novel concepts and techniques within the domain of non-parametric multi-task adapter ensembling:
1. **Early-Layer Routing in ViTs:** Recognizing that the Patch Embedding layer (Layer 0) or very early layers of a Vision Transformer can serve as zero-cost, aligned feature extractors for task routing, avoiding the need to run the backbone twice.
2. **Intra-Task Dispersion Calibration (IDC):** Standardizing cosine similarity values across diverse task manifolds by dividing them by expected in-distribution calibration similarities ($d_k$). This is a simple but effective technique to balance asymmetrical representation densities.
3. **Adaptive Task-Specific Thresholding:** Scaling the OOD rejection threshold dynamically relative to a task's expected dispersion factor ($\gamma_{OOD, k} = \eta \cdot d_k$) to resolve the security-selectivity trade-off without manual tuning.
4. **The Early-Layer Routing Compromise & ELFT:** Shifting the routing boundary slightly deeper (e.g., to Layer 1 or 2) to resolve representational overlap (the "Global-Average-Color Routing Paradox") on actual images, combined with freezing those early blocks during fine-tuning (ELFT) to prevent boundary representational mismatch.

## Delta from Prior Work
- **Delta from SABLE (SOTA):** 
  SABLE performs "Late Adaptation" by routing at Layer 10 and leaving Layers 0--9 completely unadapted, which severely restricts expert adaptation capacity. PEAR aims to route at Layer 0 (or Layer 1/2) and keep all remaining layers ($100\%$ or $\ge 83\%$ of blocks) fully adapted.
  However, under the *Early-Layer Routing Compromise* and *ELFT* (which are required for real-world images), PEAR must freeze the first 1-2 layers during training and keep them unadapted during serving. While this unadapted section is much smaller than SABLE's (2 layers vs. 10 layers), it relies on the **exact same paradigm** of freezing early layers during training and serving. The difference is quantitative (number of layers) rather than qualitative (a fundamentally new mechanism).
- **Delta from SPS-ZCA:** 
  SPS-ZCA aligns activations at specific layers but restricts adapters to mid-to-late blocks. PEAR proposes an early-layer centroid matching with dispersion calibration to enable deeper adaptation.
- **Delta from PFSR / MBH:** 
  PFSR performs routing at the final classification heads and relies on Micro-Batch Homogenization (MBH) to schedule heterogeneous batches. PEAR is completely scheduling-free and processes samples fully independently in a single parallel pass.

## Characterization of Novelty
The novelty of this paper is characterized as **incremental but highly practical**. 
While the individual components (cosine similarity, centroid matching, freezing early layers) are heavily derived from prior work like SABLE and SPS-ZCA, the combination is well-reasoned and specifically tailored to Vision Transformers. 

However, there is a clear **gap between the conceptual sales pitch and the practical execution**:
- The paper heavily pitches PEAR as a "strictly parameter-free, closed-form routing framework designed for the frozen Patch Embedding layer (Layer 0)... allowing expert adapters to be activated and blended across 100% of the network depth."
- But the real-world experiments (Section 4.6) show that Layer 0 routing **completely fails** on actual images due to the "Global-Average-Color Routing Paradox," achieving only **57.81%** accuracy.
- To resolve this, the authors must compromise and route at Layer 1 or Layer 2, freezing those layers (ELFT) during training. This directly contradicts the core claim of "routing inside the Patch Embedding layer (Layer 0)" and "adapting 100% of the layers."
- When routing at Layer 2 and using ELFT, PEAR is conceptually identical to SABLE, simply with $l_{route}=2$ instead of $l_{route}=10$. The conceptual delta is therefore minor, and the authors overstate their novelty by claiming "100% layer adaptability" when they actually freeze early layers in their best-performing real-world pipeline.
