# Evaluation Task 2: Novelty Check and Related Work Positioning

## 1. Characterization of Novelty
The novelty of this paper is best characterized as **incremental but highly pragmatic and technically clever**. The individual building blocks (polynomial depth constraints for model merging and flatness-aware optimization/randomized smoothing) are not entirely novel on their own, but their synergistic combination in a highly compact, backpropagation-free coefficient space at test-time represents a strong, practical engineering advance.

* **Conceptual Novelty (Fair-to-Good):** The core idea of identifying the vulnerability of unsupervised TTA to test-time sensor/input noise (Noise-Entropy Collapse) is an insightful extension of the "Overfitting-Optimizer Paradox" described in AdaMerging. Seeking flat loss valleys (SAM) to enhance generalization is a well-established concept.
* **Technical Novelty (Good):** The specific combination of a polynomial subspace constraint with zeroth-order randomized smoothing *in the coefficient space* is highly elegant. Applying flatness-aware optimization to just 12 blending parameters—completely bypassing the high-dimensional model weight space and backpropagation—is a very creative and practical adaptation of flatness theory for resource-constrained edge accelerators.

---

## 2. The "Delta" from Prior Work

### Delta from AdaMerging (ICLR 2024)
* **AdaMerging:** Optimizes $L \times K$ layer-wise blending coefficients independently using standard first-order gradient descent with Shannon prediction entropy minimization.
* **FlatMerge's Delta:** Instead of unconstrained, point-wise first-order optimization, FlatMerge restricts the optimization to a polynomial subspace (filtering high-frequency layer noise) and minimizes the expectation of the loss over randomized coefficient perturbations (preventing low-frequency transductive drift) using a backpropagation-free, zeroth-order gradient estimator.

### Delta from PolyMerge (Missing Citation in BibTeX)
* **PolyMerge:** Introduced the exact low-degree polynomial depth parameterization of layer blending coefficients ($\lambda^l_k$) to enforce depth-wise smoothness and reduce optimization dimensions. However, it assumes clean, ideal test data and uses standard first-order optimization or evolutionary search.
* **FlatMerge's Delta:** Under realistic input noise, PolyMerge remains vulnerable to low-frequency transductive drift where the learned polynomial trajectory overfits the systematic bias of corrupted inputs. FlatMerge addresses this residual vulnerability by adding flatness-aware randomized smoothing directly within this compact polynomial space, guiding adaptation toward robust entropy valleys.

### Delta from Sharpness-Aware Minimization (SAM) (Missing Citation in BibTeX)
* **SAM:** Reformulates the training objective to seek flat loss valleys in the high-dimensional network weight space, which requires double-backward passes and caching massive activation maps.
* **FlatMerge's Delta:** Rather than applying SAM to millions of network weights, FlatMerge applies flatness theory to the compact ($12$-parameter) polynomial coefficient space using zeroth-order randomized smoothing. This completely eliminates backpropagation and activation memory caching during test-time adaptation.

---

## 3. Scholarly Critique: Critical Bibliographic Deficiencies

As a scholarly assessment, the paper exhibits **highly severe bibliographic and technical presentation errors** where foundational papers are cited in the text but completely omitted from the bibliography (`references.bib`). This is a critical failure of scholarship that results in undefined citation warnings and a broken bibliographic chain:

1. **Missing Reference: `polymerge`**
   * **In-Text Citations:** Cited in Section 1 (page 2), Section 2.3 (page 3), Section 3.1 (page 4), Section 3.3 (page 5), and throughout the experiment sections.
   * **BibTeX File:** Completely missing from `submission/references.bib` under the key `polymerge` or any other key.
   * **Impact:** The work builds directly on PolyMerge's polynomial depth parameterization (adopting its exact mathematical formulation in Equation 3). Failing to include the formal bibliographic entry for PolyMerge is a severe scholarly oversight and a major technical presentation flaw.
2. **Missing Reference: `sam`**
   * **In-Text Citations:** Cited in Section 2.4 (page 3) and Section 3.4 (page 5) under `\cite{sam}`.
   * **BibTeX File:** Completely missing from `submission/references.bib` under the key `sam` (Foret et al., ICLR 2021).
   * **Impact:** Sharpness-Aware Minimization (SAM) is the foundational basis for FlatMerge's flatness-aware optimization formulation. Leaving out this core reference is highly unprofessional and damages the paper's scholarly rigor.

### Positioning within the Broader Literature
The related work section does a reasonable job of describing classic model merging (Task Arithmetic, RegMean, TIES-Merging, Fisher-Merging), test-time merging (AdaMerging, RegCalMerge, SyMerge), and subspace adaptation. However, the authors should expand their discussion of recent test-time calibration and merging works to better situate FlatMerge:
* They cite `regcalmerge` (arXiv 2025) and `jung2025symerge` (ICML 2026), which is excellent.
* However, they should more clearly articulate how FlatMerge's zeroth-order backpropagation-free approach compares conceptually to other test-time quantization and memory-efficiency frameworks in the bibliography (e.g., `tvq2025`, `onebitmerging2025`, `epmq2026`). Situating FlatMerge within this recent trend of post-merge quantization and edge-efficient adaptation would significantly strengthen the literature positioning.
