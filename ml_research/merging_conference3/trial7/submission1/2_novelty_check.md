# 2. Novelty and Related Work Check

## Originality and Concept Novelty
The paper exhibits **exceptional originality** by challenging a dominant theoretical consensus in the weight-space model-merging literature. The "Layer-Averaging Collapse" (rank-1 collapse) theorem had previously steered the community away from layer-wise dynamic routing toward global, single-layer routers under the assumption that layer-wise routing is mathematically redundant. By critically auditing this claim, the paper provides a much-needed course correction.

The main novel concepts and insights introduced are:
1. **SVD Spectral Audit & Collinearity Ratio ($\rho_{collinear}$):** The paper introduces a highly elegant spectral diagnostic to measure the actual dimensional concentration of learned layer-wise coefficients. This represents a solid, statistically robust method to mathematically audit the dimensionality of routing trajectories.
2. **Deconstruction of Sandbox Assumptions:** The paper pinpoints the exact mathematical reasons why prior proofs of rank-1 collapse do not hold in physical systems—specifically, they assume perfectly collinear base representations and strictly contractive linear Jacobians, which are strongly violated in deep hierarchical networks learning cross-domain tasks.
3. **The Normalization Paradox:** The conceptual deconstruction of the element-wise Sigmoid gating mechanism shows that scale-stabilizing sum-to-1 constraints (essential to prevent exponential signal decay) mathematically re-introduce the very competitive zero-sum constraint that Sigmoids were intended to bypass.
4. **The Batch-Averaged Multi-Task Inference Paradox:** This is a highly original conceptual contribution that identifies a fundamental logical and systems-level dilemma in dynamic merging. It exposes that dynamic model-merging is either logically redundant (under homogeneous batches) or functionally degraded to static merging (under mixed batches).
5. **Decoupled Gradient Paths Hypothesis:** The explanation of decoupled gradient paths for the Bounded Sigmoid (BSigmoid) router provides a highly original explanation of how element-wise independent Sigmoids avoid joint optimization collapse during backward propagation under few-shot calibration, even though their forward pass outputs are normalized.
6. **PEFT-Level Adapter Collapse Hypothesis:** The appendix presents an original hypothesis that routing over low-rank adapters (like LoRA) will exhibit an even deeper multi-dimensional specialization (lower SVD Collinearity Ratios) due to targeted spatial specialization and reduced inter-layer interference.

## Positioning Relative to Prior Work
The paper is exceptionally well-situated within the context of existing literature:
- **Static Model Merging:** It accurately positions itself relative to classic arithmetic averaging (Model Soups), Task Arithmetic, and advanced alignment techniques (TIES-merging, Git Re-Basin, REPAIR, and ZipIt!). It provides a clear, mathematically sound justification for why advanced permutation alignment or sign-conflict pruning methods are redundant when experts are fine-tuned from a shared initialization (residing in the same local loss basin).
- **Dynamic Model Merging & MoE:** It contextualizes its methods relative to dynamic merging frameworks (OFS-Tune, Zero-Vision, resolving task conflicts) and standard Mixture-of-Experts (MoE).
- **Representation Analysis:** It connects its SVD diagnostics and cosine similarity heatmaps to landmark representation analysis frameworks (SVCCA, Projection-Weighted CCA, and CKA).

## Completeness of References
The bibliography is comprehensive and covers all necessary foundational blocks:
- Standard Mixture-of-Experts: Shazeer et al. (2017), Fedus et al. (2022).
- Static Model Merging: Wortsman et al. (2022), Ilharco et al. (2022), Yadav et al. (2023) (TIES), Matena & Raffel (2022) (Fisher-weighted), Ainsworth et al. (2022) (Git Re-Basin), Jordan et al. (2022) (REPAIR), Stoica et al. (2023/2024) (ZipIt!).
- Dynamic Model Merging: Gu et al. (2024), Yadav et al. (2024).
- Representation Similarity: Raghu et al. (2017) (SVCCA), Morcos et al. (2018) (PWCCA), Kornblith et al. (2019) (CKA).
- The "Layer-Averaging Collapse" reference is cited as an anonymous/under-review work `[anonymous]`, which is appropriate for a paper that is auditing recent/concurrent assertions in active conferences.

No major missing baselines or references are identified. The literature review is thorough, fair, and exceptionally precise.
