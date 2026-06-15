# Novelty and Delta Check

## Key Novel Aspects
1. **SVD Spectral Audit in Model Merging:** The application of Singular Value Decomposition (SVD) and the definition of the Collinearity Ratio ($\rho_{collinear}$) to analyze the spatial dimensionality of learned dynamic routing trajectories is a novel diagnostic tool in the model-merging literature.
2. **Bounded Sigmoid (BSigmoid) Router with Frozen Gaussian Projection:** Combining independent sigmoidal gates with a non-parametric random projection (motivated by the Johnson-Lindenstrauss Lemma) to minimize parameter capacity and optimize under scarce calibration data.
3. **Rigorous Formulation of the Batch-Averaged Paradox:** Codifying and naming the "Batch-Averaged Multi-Task Inference Paradox" (Section 3.5), exposing the systemic limitations of dynamic model merging on heterogeneous vs. homogeneous batches.

## The 'Delta' from Prior Work
* **Rebuttal of "Layer-Averaging Collapse":** The primary delta is a direct, critical rebuttal of a highly influential recent theoretical result (cited as `[anonymous]`) that mathematically proved "Layer-Averaging Collapse." The paper identifies that the prior theoretical proof is dependent on over-simplified, linear representation-space "sandboxes" (14-layer linear networks) and low-conflict task settings. By introducing non-linear deep backbones (DeepMLP-12 and TinyCNN-4) and high-conflict suites, this work establishes the boundaries where the collapse theorem fails.
* **Router Gating:** Standard dynamic merging (e.g., Gu et al., 2024; Yadav et al., 2024) relies on Softmax gates or discrete Top-$k$ routing. This paper introduces independent Bounded Sigmoid gates to avoid the zero-sum gradient competition inherent in Softmax.

## Characterization of Novelty
The novelty is characterized as **Incremental to Moderate**. 

* **The Diagnostics:** The SVD diagnostic on routing weights is an interesting, elegant, and simple adaptation of classic linear algebra, but it is conceptually straightforward.
* **The Architecture:** The BSigmoid router is a relatively minor variation of standard gating mechanisms. Frozen random Gaussian projections are well-established (random projection theory and Johnson-Lindenstrauss are classical ML techniques), so applying them here to restrict routing capacity is a practical engineering choice rather than a fundamental algorithmic breakthrough.
* **The Empirical Scope:** While the paper claims to adopt a "rigorous physical empirical evaluation," it is heavily constrained to a highly simplified, toy setting (Split-MNIST on tiny, under-parameterized networks). Thus, the empirical validation of this novelty is highly restricted, and the broader generalizability of the "delta" to modern deep learning systems remains speculative.
