# Intermediate Review File 2: Novelty Check of the Revised Paper

## 1. Technical Novelty of the Proposed Method
From an algorithmic standpoint, the technical novelty of **Sparse Task Arithmetic (STA)** is minimal. Each of its constituent mathematical and computational steps is drawn directly from existing, well-known merging and compression methodologies:
- **Task Vector Extraction:** Reuses the standard formulation from Task Arithmetic (Ilharco et al., 2022).
- **Layer-wise Magnitude Pruning:** Retaining parameter updates based on their absolute values is a foundational pruning technique (Han et al., 2015). In the context of model merging, it is identical to the first step of TIES-Merging (Yadav et al., 2023).
- **Rescaling (R-STA):** Dividing by the survival density ($100/s$) is mathematically identical to the scale preservation step in DARE (Yu et al., 2024), where remaining values are divided by $(1-p)$.
- **Tuned STA:** Tuning the scaling coefficient $\lambda$ is a standard baseline procedure in all linear model merging literature (Wortsman et al., 2022; Ilharco et al., 2022).

Thus, STA represents a simplified, component-subtracted version of prior work (TIES-Merging and DARE), containing zero new mathematical operations or architectural blocks.

## 2. Conceptual and Methodological Novelty
Despite its low technical novelty, the paper has strong conceptual and methodological novelty as a **deconstructive critique**.
- **Challenging Community Assumptions:** The paper systematically deconstructs a widely accepted assumption—that coordinate-wise sign voting and sign consensus enforcement are necessary to prevent parameter interference during sparse merging.
- **Isotropic Noise Filtering Perspective:** Reinterpreting magnitude pruning not as a "sign-conflict resolver" but as an isotropic noise-filtering process is an intellectually engaging conceptual contribution. It shifts the focus from resolving sign conflicts back to weight-space dynamics and gradient noise.
- **Tail-Bias & Variance Distortion Analysis (Section 4.3):** In the revised paper, the authors introduce a new analysis of the "tail-bias" of magnitude pruning vs. the "uniform variance" of stochastic dropout. This provides an elegant, scholarly explanation for why Rescaled STA fails at low densities while DARE succeeds. It represents a solid conceptual contribution to our understanding of sparsified weight spaces.

## 3. Key Limitations to Novelty and Contextualization
While the critique is conceptually valuable, several limitations remain in how the authors position their work relative to concurrent literature:
- **Omission of DARE-TIES:** In the original DARE paper (Yu et al., 2024), DARE's stochastic dropping is presented as a preprocessing step that can be combined with *either* Task Arithmetic (DARE-TA) *or* TIES-Merging (DARE-TIES). The authors' DARE baseline in their codebase is DARE-TA (direct addition without sign consensus). By omitting the sign-consensus version of DARE (DARE-TIES), the authors compare against a weaker baseline.
- **Standard Pruning-in-Merging Literature:** The idea of applying magnitude pruning prior to model merging is not new. Decoupled "Prune-then-Merge" pipelines have been explored (e.g., ZipMerge, 2025; and other works). The authors should more clearly position STA relative to these works, acknowledging that pruning before merging is an established concept.
