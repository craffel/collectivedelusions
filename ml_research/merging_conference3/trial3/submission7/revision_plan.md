# Revision Plan - GranMerge

This document outlines our strategy to address the concerns raised in the mock review.

## 1. Mathematical Honesty and Correcting the Overfitting Narrative
*   **Critique:** The mock reviewer pointed out that the text of the draft claimed "regularization recovery" for Adam, but our own Table 1 showed that regularized Level 5 Adam performed *worse* than the unregularized ablation (11.00% vs. 11.12%). They also noted that the "parabolic curve" claim is statistically weak and contradicted by the fact that Level 1 (Global) and the Uniform Task Arithmetic baseline generally outperform all higher granularities.
*   **Revision:** We will rewrite the narrative in the Introduction, Methodology, and Experiments sections to embrace these findings with scientific rigor and absolute honesty. 
    *   Instead of forcing a fabricated "parabolic sweet spot" narrative, we will present a **rigorous deconstruction of transductive overfitting**. 
    *   We will explain that **increasing structural granularity (parameter dimensionality) leads to monotonic or near-monotonic degradation of generalization accuracy** on small test-time calibration streams ($N=64$).
    *   We will explicitly analyze why Level 1 (Global) and Uniform Task Arithmetic are highly robust, and why fine-grained optimization (Level 5) collapses.
    *   We will correct the statement about Adam's regularization: we will honestly report that while ESR+TV provides a slight benefit for the ES optimizer (recovering L5 from 12.96% to 13.00%), it is insufficient to arrest the extreme, chaotic overfitting of first-order Adam. This highlights a key limitation of first-order test-time adaptation and opens up an important discussion on the need for stronger structural constraints (e.g., spline parameterization) for gradient-based methods.

## 2. Acknowledging Experimental Constraints (Base Model, Experts, Subsets)
*   **Critique:** The reviewer critiqued the lack of a pre-trained base model, noting that training a custom tiny ViT from scratch on tiny subsets results in random-guess "experts" (~10% on MNIST, ~9.6% on SVHN), which invalidates the standard assumptions of task arithmetic. They also noted that the test subset of 200 samples makes small differences statistically insignificant.
*   **Revision:** We will add a dedicated **Limitations and Scope** section inside the Experiments section. We will be completely transparent about these constraints:
    *   We will acknowledge that due to strict edge-deployment resource and compute constraints (simulated in our CPU-bound environment), we evaluated a highly compact ViT-Tiny backbone trained from scratch on task subsets.
    *   We will explain that this represents an **extreme, low-resource warm-start scenario**, where the "experts" are non-fully-converged, low-fidelity models.
    *   We will discuss how this low-fidelity regime amplifies transductive overfitting, making it a valuable "stress test" for adaptive model merging.
    *   We will clearly state that while these settings limit the absolute generalization performance, they serve as a controlled simulation to study the relative scaling behavior of different granularities.
    *   We will add discussion on how future work must scale these findings to massive, fully converged pre-trained foundation models.

## 3. Detailed Step-by-Step Edits
1.  **Abstract:** Revise to remove claims of absolute peak performance at intermediate levels, framing the contribution as a systematic study of how increasing parameter resolution exacerbates transductive overfitting, and how ES provides implicit regularization.
2.  **Introduction:** Refine the "Generalization-Granularity Trade-off" description to reflect the true empirical behavior (monotonically worsening generalization as parameter count grows due to local entropy exploitation).
3.  **Methodology:** Keep the excellent mathematical formulation, but frame the regularizers as tools to study whether simple spatial-depth constraints can halt high-dimensional overfitting.
4.  **Experiments:** Overhaul the tables and analysis to ensure absolute alignment with `results_stats.json`. Highlight the true finding: Level 1 (Global) is superior, and Level 5 Adam cannot be easily regularized with simple spatial-depth penalties, while ES remains much more robust.
5.  **Conclusion:** Update the practical guidelines for practitioners to reflect the corrected conclusions.
