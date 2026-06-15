# 2. Novelty and Literature Delta Check

An in-depth scholarly assessment of the novelty claims made by the authors, the actual "delta" from prior work, and how well the submission positions itself in the existing literature.

## Evaluation of Claims of Primacy
The authors claim that HyperMerge is **"the first work to introduce hyperbolic geometry and non-linear Möbius algebra to the domain of modular deep learning and dynamic model ensembling, breaking the traditional flat Euclidean barrier."**

From a rigorous literature perspective, this claim of absolute primacy is **overstated and inaccurate**:
1. **Hyperbolic Adapters and PEFT:** Works such as *HypLoRA: Hyperbolic Fine-Tuning for Large Language Models* (Yang et al., 2024/2025) have already explored applying hyperbolic geometry to low-rank adaptation (LoRA) by projecting representations to the Lorentz manifold, applying low-rank adaptations, and mapping back. Similarly, *Training-Free Dual Hyperbolic Adapters* (Zhang et al., 2024) has integrated hyperbolic manifolds with adapters.
2. **Hyperbolic Routing and Experts:** *MoSLoRA: Parameter-Efficient Fine-Tuning of LLMs with Mixture of Space Experts* (2025/2026) directly proposes using heterogeneous geometric spaces (hyperbolic, spherical, Euclidean) as experts in an MoE-like LoRA framework with dynamic routing. Additionally, *HELM: Hyperbolic Large Language Models via Mixture-of-Curvature Experts* (2025) explores Mixture-of-Experts operating in different curvature spaces.

While HyperMerge focuses on *dynamic test-time activation-space ensembling/merging*, these existing papers clearly establish the use of hyperbolic geometry, Möbius algebra, and routing in modular deep learning and PEFT. The authors fail to cite or discuss these highly relevant works, leading to false claims of absolute novelty.

## The Real "Delta" from Prior Work
The true, legitimate contribution of HyperMerge lies in:
* **Beltrami-Klein Symmetric Blending (BKSB):** Specifically applying the Beltrami-Klein model's Einstein midpoints to perform *permutation-invariant* and *single-pass* activation-space blending of LoRA updates. Existing hyperbolic routing schemes (like MoSLoRA) typically route to separate discrete expert branches rather than performing a symmetric continuous weighted ensembling of intermediate updates using Lorentz-weighted Einstein midpoints on the manifold.
* **Hyperbolic Centroid Alignment (HCA):** The closed-form computation of task barycenters on a calibration split by transforming Poincaré coordinates to Klein coordinates, weighting by Lorentz factors, and projecting back to the Poincaré Ball.

This specific algebraic application to activation blending is a neat mathematical contribution, but it must be framed as an extension of hyperbolic PEFT/MoE rather than a pioneering first.

## Bibliographic Discrepancies and Unused References
A scholarly check of `references.bib` against the main text reveals several issues:
1. **Omissions of Concurrent/Prior Work:** As noted above, key publications like *HypLoRA* (Yang et al.), *MoSLoRA* (2025/2026), and *HELM* (2025) are completely missing from both the bibliography and the related work.
2. **BibTeX Clutter / Uncited References:** There are numerous references in `references.bib` that are never cited in the text. For example, `matena2021merging` ("Merging models with fisher information") is a highly relevant static model merging paper in the bibliography but is completely omitted from the text's Related Work section. Similarly, multiple general-purpose Transformer papers (such as `vaswani2017attention`, `radford2019language`, `devlin2018bert`, etc.) and PEFT papers (such as `zhang2023adalora`, `valipour2023dylora`, etc.) are in the `.bib` file but never referenced in any `.tex` file.
3. **Imprecise Baseline Citations:** Under Section 4.2, the baseline `SABLE (Early Routing)` and `SABLE (Late Adaptation)` are cited as `\cite{stoica2023zipit}` (ZipIt!). However, ZipIt! is a static feature-aligned parameter-merging method. While SABLE may build on ZipIt! or be a variant, SABLE itself is an activation blending method that is distinct from standard ZipIt!. The citation of Stoica et al. (2023/2024) for SABLE without further clarification of its origin is confusing and biblically imprecise.

## Characterization of Novelty
The novelty of the proposed method is **incremental to moderate**. While the introduction of hyperbolic geometry and Klein Einstein midpoints to activation ensembling is mathematically elegant, the core concepts of hyperbolic adapters, hyperbolic routing, and dynamic activation-space ensembling are already present in literature. The failure to cite and position the paper within the actual landscape of hyperbolic PEFT and MoE represents a major scholarship gap.
