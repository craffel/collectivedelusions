# 3. Soundness and Methodology

## Clarity of the Description
The methodology is exceptionally well-structured and described with rigorous scientific precision. Key positive aspects of the presentation include:
1. **Mathematical Grounding:** Each mathematical formulation—including node/edge potentials, the Boltzmann distribution, exact bidirectional message propagation, on-the-fly backward recurrence, and the Dobrushin contraction theorem—is explicitly detailed with consistent notation.
2. **Step-by-step Algorithms:** Section 3 systematically breaks down the Predict-then-Smooth pipeline, the forward-backward equations, scale-normalization steps, and marginal weights assembly.
3. **Appendix Code Integration:** Appendix A contains a self-contained, clean, and production-grade PyTorch implementation of the `QPathMergeController` class, which aligns perfectly with the equations in Section 3.
4. **Transparent Discussion of Limitations:** The paper does not hide limitations; it explicitly discusses the "reality gap" of using ResNet-18 as a proxy, few-shot calibration constraints, and Oracle sub-optimality due to signature perturbation effects.

---

## Appropriateness of Methods
The proposed methods are highly appropriate for resolving the accuracy-stability dilemma:
1. **Belief Propagation on 1D MRF:** Mapping layer-to-layer ensembling to a 1D chain MRF is mathematically elegant. 1D chain graphical models can be solved exactly and efficiently using sum-product belief propagation (the Forward-Backward algorithm), which avoids the need for approximate or iterative inference.
2. **Decoupled Spatial Smoothing:** Solving the MRF entirely within the depth lattice of a single sample eliminates sample-to-sample state tracking, resolving the temporal lag and hysteresis of previous stateful models.
3. **On-the-Fly Truncated Backward Horizon:** Truncating the backward pass to a small constant horizon ($H = 4$) is highly appropriate because it reduces complexity from $O(L^2 K^2)$ to $O(L H K^2)$, restoring linear complexity while maintaining near-oracle smoothness. This is a brilliant engineering optimization backed by robust theory.
4. **Dobrushin Contraction Theorem:** Using this contraction mapping property to formally prove the exponential decay of truncation error is highly appropriate, grounding the $H=4$ heuristic in solid mathematical theory.
5. **Linear Extrapolation:** Using local potential slope trends (\texttt{LinearExtrap}) is an effective and appropriate solution to break power-iteration degeneracy, enabling the backward pass to anticipate downstream task switches without requiring a full forward trial pass.

---

## Potential Technical Flaws or Limitations
While the methodology is highly sound, a rigorous evaluation highlights a few areas of discussion:
1. **Centroid Calibration Prerequisite:** The method relies on pre-calibrated expert activation centroids or channel signatures. Although the authors demonstrate that this calibration is highly sample-efficient (requiring only 1 to 4 samples) and robust to distribution shifts due to cosine scale-invariance, it remains a necessary offline step.
2. **Static transition leakage $M$ and temperature $\tau$:** The edge potentials use a static transition leakage $M = 0.10$, and the temperature parameter $\tau = 0.5$ is set globally. Although the authors propose layer-specific leakage scheduling and learned dynamic edge potentials as speculative extensions, these are not fully implemented or evaluated in the main results.
3. **Power-Iteration Degeneracy of QPathMerge-Single:** The speculative assumption of constant future potentials ($\psi_{l'} = \psi_l$) reduces the backward recurrence to a power iteration. Under this assumption, the backward beliefs do not contain genuine future information, but rather a smoothed reflection of the current layer's potentials. Although the authors successfully resolve this via \texttt{LinearExtrap}, the raw single-pass version without extrapolation exhibits this mathematical degeneracy.

---

## Reproducibility
The reproducibility of this paper is **excellent**:
1. **Self-Contained PyTorch Code:** The authors provide the exact code in Appendix A for the `QPathMergeController` module, which can be directly dropped into any PyTorch model pipeline.
2. **Explicit Parameter and Calibration Settings:** The paper lists all specific parameters used, including hidden dimensions ($D=192$), active layers ($4$ to $14$), temperature ($\tau = 0.5$), transition leakage ($M = 0.10$), truncated horizon ($H = 4$), and ResNet modulation strength ($\lambda = 0.25$).
3. **Clear Evaluation Workloads:** The datasets (ImageNet classes downloaded programmatically), query lengths (200 samples), and augmentation parameters are fully specified, enabling independent replication.
