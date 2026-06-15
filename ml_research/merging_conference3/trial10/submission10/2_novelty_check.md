# 2. Novelty and Originality Check

This section evaluates the originality and novelty of the contributions made in the paper, focusing on the conceptual framing, proposed algorithms, and mathematical analysis.

### 1. Conceptual Framing: "Minimalist Deconstruction" (High Originality)
The paper's most refreshing aspect is its "deconstructionist" approach. While the machine learning community frequently adopts increasingly complex dynamical formulations—such as modeling parameter ensembling as biochemical ODEs or utilizing learned first-order state-space models—this paper successfully deconstructs these methods to show that their core performance resides in a much simpler mathematical concept: localized recursive filtering (specifically, discrete first-order 2D bilinear IIR filters).

Revealing that a simple, training-free 2D autoregressive recursive filter can match or outperform biochemical and PAC-Bayesian formulations is a highly valuable conceptual contribution that promotes simplicity, interpretability, and execution efficiency in edge-serving environments.

### 2. Methodological Originality: 2D-STEM (Moderate Originality)
*   **Comparison with SABLE:** SABLE performs stateless nearest-centroid routing at each layer and sample. 2D-STEM introduces spatial and temporal recurrences, converting this stateless routing into a stateful, smooth sequence.
*   **Comparison with Momentum-Merge:** Momentum-Merge uses spatial (depth-wise) EMA but resets the state for every new sample. 2D-STEM extends this by keeping a sequence-level temporal state and coupling both spatial and temporal smoothing in a unified 2D update.
*   **Comparison with PAC-Kinetics:** PAC-Kinetics propagates state temporally but forces the same routing coefficients across all layers, losing depth-wise representational alignment and centroid calibration. 2D-STEM resolves this by enabling independent layer-wise centroids and propagating state across both dimensions. Furthermore, 2D-STEM is training-free, whereas PAC-Kinetics requires training complex state-transition matrices offline using PAC-Bayesian bounds.
*   **Comparison with ChemMerge:** ChemMerge models routing trajectories as biochemical reaction kinetics governed by ODEs integrated online. 2D-STEM replaces continuous-time ODE solvers with a single-line 2D bilinear update.
*   **Novelty Level:** Moderate. The basic mathematical formulation of a 2D bilinear EMA is a classical signal-processing technique (an AR(1) model in 2D space). However, its adaptation to multi-task expert ensembling across backbone depth and stream time is highly creative and novel in this context.

### 3. Adaptive Temporal Gating (ATG-PL) and Power-Law Sharpening (Moderate-to-High Originality)
The authors acknowledge that using stream-level similarity to dynamically scale temporal momentum is an adaptation of the *Adaptive Online Kinetics* framework from PAC-Kinetics. However, the following aspects of their gating are highly original and technically sound:
1.  **Power-Law Sharpening ($\gamma \ge 2$):** The identification of the upward bias in cosine similarity when computed on non-negative probability vectors (which prevents momentum from collapsing to zero during task switches in overlapping manifolds) is mathematically sharp and original. The proposed power-law exponent ($\gamma = 3$) elegantly resolves this bias-variance trade-off.
2.  **Coordinate-Prior Spatial Boundary Condition:** The theoretical identification of spatial momentum cancellation at the first adapted layer under a raw-weight boundary is highly nuanced. Designing the Coordinate-Prior boundary using early frozen layers to activate spatial smoothing at the first adapted layer is an excellent, original contribution.
3.  **Top-$k$ Coordinate Masking:** This Appendix-level sparse extension is a clever, parameter-free way to ensure that transition detection scales to extremely large expert pools ($K \ge 50$) with $O(1)$ similarity complexity.

### Summary Verdict on Originality
The paper does not necessarily introduce entirely new primitives; rather, it combines classical signal processing concepts (2D bilinear filters/IIR filters) and centroid-based routing in a highly creative, mathematically elegant way. The deconstruction of more complex baselines, combined with the elegant resolution of the boundary condition and coordinate bias (ATG-PL), makes the work highly original and of significant interest to the edge-serving and parameter-efficient ensembling communities.
