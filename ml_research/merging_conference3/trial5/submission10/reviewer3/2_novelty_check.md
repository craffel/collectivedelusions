# 2. Novelty and Literature Positioning Check

## Key Novel Aspects
1. **Chaos-Theoretic Parameter Merging:** Modeling the layer depth of a deep network as discrete temporal steps in a chaotic Coupled Map Lattice (CML) to determine dynamic parameter-space merging coefficients. This represents a highly creative and original conceptual bridge between non-linear classical physics (specifically spatial-temporal chaos) and weight-space neural network fusion.
2. **Gated Coupled Map Lattice (G-CML) Formulation:** Modifying the standard recurrence of a chaotic system with a learned, contractive layer-wise gate to stabilize gradients. This allows the system to operate on the "edge of chaos" during initial optimization phases and transition to stable, localized attractor basins at inference.
3. **Annealed Chaos-to-Order Merging:** Developing a hybrid annealing schedule that transitions the discrete update map from a chaotic Logistic Map (for global exploration) to a contractive Tanh-gated map (for local exploitation). This is a highly novel strategy that directly addresses the "Gated Chaos Paradox."

## The "Delta" from Prior Work
- **From Static Merging (e.g., Task Arithmetic, TIES-Merging, DARE, FoldMerge):** Unlike static methods that apply a uniform, fixed set of merging coefficients across all inputs and layers, ChaosMerge computes layer-specific, input-dependent scaling coefficients dynamically, significantly reducing parameter-space representational interference.
- **From Dynamic/Adaptive Merging (e.g., AdaMerging, Linear Routers, QWS-Merge):** Traditional dynamic routers either rely on unconstrained linear projection layers (which easily overfit in low-data regimes due to having nearly $30\times$ more parameters) or use wavefunction phase superposition (QWS-Merge). ChaosMerge enforces a highly regularized, physically bounded discrete dynamical system prior.
- **From Classical CMLs and Recurrent Neural Networks:** Unlike standard physical CMLs (Kaneko, 1992) which exhibit severe gradient explosion when run recursively, or traditional RNNs (LSTMs, GRUs) which operate in high-dimensional feature spaces, G-CML uses low-dimensional sphere-projected input features to perturb a tightly constrained, low-dimensional (e.g., $K$-dimensional) parameter-space lattice.

## Characterization of Novelty
- **Conceptual Novelty: Significant.** Applying spatio-temporal chaotic lattices and attractor steering to the parameter-merging space is highly original and opens up an exciting new perspective.
- **Technical Novelty: Moderate-to-High.** Individually, the Logistic Map, Coupled Map Lattices, gating (skip-connections), random sphere projections, and task centroids are well-established. However, their combination, the rigorous Lyapunov stability analysis, and the formulation of the Annealed Chaos-to-Order training scheme show high technical sophistication and a strong understanding of both physical systems and deep learning.

## Literature Positioning and Citation Integrity Check
As a scholarly reviewer, we must closely examine the bibliography and text integration. We identify two major flaws in the submission's positioning and literature care:

1. **Broken and Leaked Internal Citation Keys:**
   - In `02_related_work.tex` (Line 13), the paper cites `\cite{trial2_submission3}` for the *Overfitting-Optimizer Paradox*.
   - In `04_experiments.tex` (Lines 27 and 179), the paper cites `\cite{trial3_submission2}` for *OFS-Tune* and the *Overfitting-Optimizer Paradox*.
   - **Critical Mismatch:** Neither `trial2_submission3` nor `trial3_submission2` is defined in the `references.bib` file. This represents a severe lack of care during final drafting and exposes the internal file-naming conventions of an automated benchmark pipeline. In the bibliography, the corresponding keys are actually `ofstune` and `regcalmerge`, meaning the compilation will produce broken question marks `[?]` or uncompiled citation keys in the PDF.

2. **Abuse of `\nocite{*}` to Inflate the Bibliography:**
   - The authors include `\nocite{*}` before `\bibliography{references}` in `submission/example_paper.tex`, which automatically dumps every entry from `references.bib` into the paper's references.
   - While the bibliography lists highly relevant contemporaneous works such as `regcalmerge` (2025), `polymerge` (2025), `saim` (2025), `ofstune` (2025), `zipmerge` (2025), and `foldmerge` (2025), the authors **fail to discuss or contextualize almost all of them in the main text** (with the sole exception of a brief mention of FoldMerge). 
   - A scholarly reviewer expect papers to actively discuss and differentiate themselves from closely related contemporaneous works—especially those they include in their bibliography—rather than silently appending them to the references page to appear well-read.
