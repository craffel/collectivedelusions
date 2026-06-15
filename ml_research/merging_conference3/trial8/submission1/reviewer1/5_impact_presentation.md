# 5. Impact and Presentation

## Major Strengths
1. **Mathematical Elegance and Rigor:** The use of the Beltrami-Klein model and its Einstein midpoint formula to resolve the non-associativity of sequential Möbius additions is a highly clever and mathematically sophisticated algebraic formulation.
2. **Clear and Cohesive Narrative:** The paper is extremely well-structured, easy to follow, and articulates its motivations and core ideas (like the geometric limits of Euclidean ensembling and representation-level hierarchies) with great clarity.
3. **Comprehensive Baselines:** The authors compare their method against a wide variety of static and dynamic model merging techniques, including state-of-the-art Euclidean methods like SABLE and SPS-ZCA.
4. **Detailed Parametric Ablations:** The paper includes valuable ablation studies on the impact of curvature $c$, out-of-distribution rejection threshold $\gamma_{\text{OOD}}$, and a crowded "Overlapping Subspace" regime.

---

## Areas for Improvement
1. **Addressing the "Small-Norm" Boundary Contradiction:** The authors must reconcile the fact that LoRA updates have small norms ($\|E\|_2 \ll 1$), which places their projections near the Poincaré origin. Since hyperbolic space is locally Euclidean near the origin, the claimed benefits of "exponential volume growth near the boundary" to prevent crowding are physically inactive.
2. **Real-World Empirical Validation:** The authors must move beyond a synthetic "Analytical Coordinate Sandbox" to evaluate HyperMerge on real-world deep neural networks (such as LLaMA or ViT) with real-world datasets (e.g., merging GLUE or image classification adapters), to demonstrate actual practical utility.
3. **Resolving the Quantitative Discrepancy:** The authors need to explain why Table 1 reports **83.40%** joint mean accuracy for $c=0.1$, while Table 2 reports **89.30%** for the exact same curvature value ($c=0.1$). This is a major reporting inconsistency.
4. **Softening Grandiose Claims:** The manuscript frequently uses hyperbolic phrasing (e.g., "shatters these geometric limits," "paradigm-shifting approach," "completely redefines the geometric substrate") that is not supported by the empirical evidence, as flat-space Euclidean baselines (SABLE) consistently outperform HyperMerge.
5. **Clarifying Baseline Similarities:** The authors should clarify that "absolute immunity to stream heterogeneity" is not a unique geometric property of HyperMerge, but rather a standard characteristic of any sample-wise activation ensembling method (as shown by SABLE and SPS-ZCA also having 0.00% collapse).

---

## Overall Presentation Quality
- **Rating: Excellent.**
- **Details:** The layout, formatting, table structure, and language are exemplary. The mathematical notation is clean and consistent throughout the paper. Section 3.8 (distortion analysis) and Section 3.9 (numerical safeguards) show a high degree of diligence in addressing the edge cases of hyperbolic computations.

---

## Potential Impact and Significance
- **Rating: Low to Moderate.**
- **Details:** While the mathematical contribution is elegant, the practical impact is limited. Because HyperMerge is outperformed by a simpler Euclidean baseline (SABLE) and is mathematically equivalent to flat Euclidean ensembling in the small-norm limit, deep learning practitioners have no incentive to adopt this method over much simpler, faster, and more performant flat-space ensembling methods. Its main significance is theoretical, serving as a formal mathematical exploration of non-linear ensembling operators.
