# Evaluation Part 2: Novelty and Delta from Prior Work

## Key Novel Aspects
1. **Minimalist Framing / Boundary Probing:** Rather than seeking state-of-the-art results, the primary novelty of this work lies in its "deconstructive audit" framework. It systematically evaluates the absolute lower bound of parameter capacity in test-time adaptive model merging (exactly $K=8$ parameters).
2. **Ray-Scaling Simplex Projection:** The paper employs a non-orthogonal ray-scaling projection onto the convex barycentric simplex, arguing that it preserves directional ratios and avoids the hard sparsification inherent to Euclidean orthogonal projections.
3. **Mean-Field Proximity Penalty:** The introduction of a closed-form geometric penalty that pulls task-specific coefficients towards a stable, uniform barycentric centroid.

## Delta from Prior Work
- **From Static Merging (Task Arithmetic, TIES):** BPAM optimizes merging weights on unlabeled test-time streams, unlike static methods. However, BPAM-Static uses only 8 parameters, making its parameter footprint comparable to static methods.
- **From High-Capacity Adaptive Merging (AdaMerging, SyMerge, FoldMerge):** AdaMerging uses layer-wise coefficients (1,264 parameters), FoldMerge uses a 4-layer normalizing flow (2.6M parameters), and SyMerge uses low-rank adapters. BPAM reduces weight-space parameters by 99.3% to 99.99%, optimizing only $K$ global task scalars. 
- **Deconstructive Audits (SAIM, Layer-wise Audit):** This work builds upon the deconstructive philosophy of the SAIM Audit and the Layer-wise Model Merging Sanity Check, extending the analysis of transductive overfitting and spatial parameter redundancy.

## Characterization of Novelty
The novelty is primarily **conceptual and diagnostic (incremental/diagnostic)**. Rather than proposing a new performance-oriented technique, it acts as a "sanity check" or a diagnostic baseline to map the exact performance threshold where layer-wise degrees of freedom become necessary. The mathematical components (convex combinations, simplex projection, and $\ell_2$ centering penalties) are standard, but their combination to analyze parameter boundaries in model merging represents a valuable conceptual contribution to the community.
