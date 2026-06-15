# Novelty Check: GranMerge

## 1. Positioning relative to Prior Work
The paper positions itself as a systematic study of the "Generalization-Granularity Trade-off" in model merging. It builds on and contrasts itself with several existing lines of work:
*   **Task Arithmetic (Ilharco et al., 2022):** Uses manually-tuned, coarse-grained global scales.
*   **AdaMerging (Yang et al., 2024):** Introduces test-time adaptation using layer-wise coefficients (intermediate granularity).
*   **PolyMerge/SplineMerge (Jung et al., 2025/2026):** Restricts layer-wise profiles using polynomials or splines to mitigate overfitting.
*   **RegCalMerge (Jin et al., 2026):** Mitigates transductive overfitting using spatial and depth regularizers like Elastic Spatial Regularization (ESR).

## 2. Assessment of Originality
Rather than introducing a minor heuristic tweak, the primary novelty of the paper lies in its **systematic, multi-dimensional empirical deconstruction of structural granularity**. Specifically, it:
1.  **Defines a unified hierarchical spectrum:** Nesting 5 distinct granularities (Global, Layer-wise, Block-wise, Component-wise, and Tensor-wise) which were previously studied in isolation or not at all (e.g., component-wise and tensor-wise scales).
2.  **Examines the interaction between structural granularity and optimizer family:** Comparing first-order (Adam) and zero-order (1+1 ES) optimization trajectories in parameter-blending spaces, revealing unique behaviors and offering alternative mathematical interpretations of zero-order robustness in high dimensions.

## 3. Conceptual Contribution: The "Generalization-Granularity Trade-off"
The deconstruction of this trade-off provides extremely valuable conceptual vocabulary to the model merging community. The paper formalizes how test-time adaptation of a large number of blending parameters on a small calibration batch leads to a transductive shift, where the model minimizes prediction entropy by creating "confident but incorrect" predictions, destroying generalizable features.

## 4. Novelty Limitations and Overlaps
While the empirical study is exceptionally thorough, some of the core insights are pre-figured by prior work:
*   The fact that optimizing layer-wise or fine-grained coefficients on small batches leads to transductive overfitting is the primary motivation of both **PolyMerge** (which uses splines to reduce parameters) and **RegCalMerge** (which uses ESR). Thus, the existence of this overfitting and the use of ESR to solve it are not entirely novel.
*   The comparison of Adam vs. ES in model merging was partially touched upon in other literature, though the paper provides a much more structured comparison across all five granularities.
*   However, the paper's primary value is diagnostic and empirical rather than prescriptive. It is an honest, high-signal deconstruction of the failure modes of adaptive weight blending under low-resource constraints, which represents a highly valuable, novel conceptual contribution.
