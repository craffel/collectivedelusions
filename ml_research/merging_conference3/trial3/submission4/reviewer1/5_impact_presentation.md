# 5. Impact and Presentation

## Major Strengths
1. **Outstanding Scientific Honesty:** The paper stands out for refusing to "curate a narrative of triumph." By rigorously analyzing and reporting the failure of highly complex test-time co-optimization, it provides a highly valuable cautionary lesson for real-world practitioners.
2. **Deconstruction of Complexity:** It clearly demonstrates that a simple, unoptimized, decoupled baseline (Prune-then-Merge) outperforms complex, test-time adaptive co-optimization under extreme task conflict. This is a powerful validation of minimalist software and machine learning design.
3. **Rigorous and Extensive Ablation Suite:** The authors do not stop at the failure of ZipMerge; they systematically isolate and address every potential confounder (expert convergence, backbone architecture, language models, calibration sample selection, structured block-pruning latency, and PEFT/LoRA merging).
4. **Introduction of Orthogonal Procrustes Alignment:** The mathematical formulation and empirical validation of post-hoc SVD-based coordinate rotation (+16.45% absolute improvement with negligible overhead) represents a highly elegant, simple, and high-yield contribution.

## Areas for Improvement
* **Framing and Title:** The paper is presented as "ZipMerge: Joint Model Merging and Pruning...", but ZipMerge itself is shown to be highly over-engineered and ineffective compared to simpler baselines. The framing could be shifted even further to emphasize the "Post-Mortem and Limitation-Mapping" in the main title (e.g., "A Post-Mortem of Joint Model Merging and Pruning on Edge Hardware: Exposing the Limits of Test-Time Co-Optimization").
* **Complexity of Proposals:** While the authors identify that unconstrained entropy TTA collapses, some of their proposed regularized objectives (like the self-supervised CBC loss) introduce even more complexity. Highlighting that simpler structural distance penalties (like Reg-ZipMerge) or PEFT manifolds are the primary, most robust solutions is crucial.

## Presentation Quality
* **Excellent Layout and Narrative:** The narrative flows logically from deployment constraints to method formulation, honest empirical results, detailed post-mortem, and extensive ablations.
* **Mathematical Precision:** All formulas, algorithms, and schedules (linear, cubic, cosine) are mathematically detailed and easy to parse.
* **Visual Aids:** Figures and tables are clear and directly reinforce the core findings (e.g., the representational collapse curves in Figure 1).

## Potential Impact and Significance
This paper is highly significant for both researchers and edge-AI practitioners. It steers the community away from overly complex, unconstrained on-the-fly test-time optimization loops that overfit and collapse, and redirects attention toward simpler, structurally robust solutions (such as pre-merging spatial filtering, PEFT adapter constraints, and lightweight coordinate rotation). It establishes a vital, honest baseline that will save future engineers immense computational resources and development time.
