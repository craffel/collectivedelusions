# Revision Plan: Addressing Mock Review Feedback

This document outlines our structured plan and progress in addressing the peer review critiques for **Grassmannian Subspace Consensus Merging (GSC-Merge)**.

---

## 1. Summary of Identified Weaknesses & Actions

| Weakness | Severity | Action | Status |
| :--- | :---: | :--- | :---: |
| **Flaw 1: Math Error in Proposition 3.2 Proof** | High (Critical) | Reformulate Proposition 3.2 and its proof strictly under the spectral norm (matrix 2-norm) $\|\cdot\|_2$, correcting the conflated matrix norms and maintaining mathematical tightness. | **Completed** |
| **Flaw 2: Methodological Representation Mismatch** | High (Critical) | Modify `run_experiments.py` so that non-target parameters (biases, layer norms, patch/positional embeddings) are kept task-specific during evaluation and validation optimization, resolving the catastrophic performance collapse. | **Completed** |
| **Flaw 3: Sloppy/Hardcoded Ceilings & Discrepancies** | Medium | Update individual expert evaluations to run dynamically and populate the results report dynamically. Ensure there are no hardcoded references to outdated draft values. | **Completed** |
| **Flaw 4: Missing Statistical Variance Analysis** | Medium | Incorporate statistical discussion and explain how the spectral projection serves as a robust variance-reduction operator under few-shot data regimes. | **Completed** |
| **Flaw 5: Misleading Table Bolding & Reporting** | High | Automate quantitative table formatting by dynamically computing and bolding the actual mathematical maximum in each column for both Table 1 (task-conditional) and Table 2 (task-agnostic). | **Completed** |
| **Flaw 6: Narrative Inconsistencies & Absolute Claims** | Medium | Reframe the performance claims in the Abstract and Introduction to describe GSC-Merge as a variance-reducing regularizer (bias-variance trade-off) under task-conditional setups, while highlighting its absolute outperformance in task-agnostic settings. | **Completed** |
| **Flaw 7: SVD Computational Complexity on LLMs** | Medium | Implement an empirical SVD profiling script to benchmark Exact SVD vs. Randomized SVD (Halko et al., 2011) across multiple network capacities (ViT-Tiny, ViT-Base, LLaMA-7B sizes). Add a full Appendix detailing the complexity, speedup, and approximation accuracy. | **Completed** |

---

## 2. Fifth Pass Execution Details (Completed)

### Step 2.1: Upfront Clarification of Partial Merging
- Updated `submission/sections/00_abstract.tex` and `submission/sections/01_intro.tex` to explicitly introduce GSC-Merge as a partial model merging framework targeting the major linear projection layers (representing $>95\%$ of parameters) while keeping lightweight normalization and bias parameters task-specific to prevent statistic mismatch across highly disparate visual domains.
- Clarified the key contributions list in `01_intro.tex` to match this definition.

### Step 2.2: Re-framing of Performance Claims (Bias-Variance Trade-off)
- Revised the Introduction to clearly frame the comparison with unconstrained OFS-Tune under the task-conditional setting as a classic bias-variance trade-off (GSC-Merge acts as a robust spectral regularizer, reducing split-sensitivity standard deviation from $\pm 4.31\%$ to $\pm 2.76\%$, rather than outperforming in mean accuracy).
- Highlighted that GSC-Merge with $\gamma=0.5$ achieves absolute outperformance over unconstrained tuning and all other baselines in the truly task-agnostic setting.

### Step 2.3: Empirical SVD Complexity & Scalability Appendix
- Created a benchmarking script `profile_svd.py` implementing Halko's randomized SVD algorithm with oversampling $p=10$ and 1 subspace iteration.
- Evaluated on CPU (Intel Xeon, 4 threads) across ViT-Tiny, ViT-Base, and LLaMA-7B reduced sized layers.
- Measured a massive **23.56x speedup** on LLaMA-sized layers ($2048 \times 8192$) with an extremely low relative error difference of only **2.46%** compared to the optimal exact SVD projection.
- Appended a comprehensive "Appendix A: Randomized SVD Scalability and Empirical Benchmarks" section to `submission/example_paper.tex` presenting these findings and practical recommendations.

### Step 2.4: Dynamic Quantitative Table Bolding
- Modified `process_results_and_build.py` to replace hardcoded table bolding with fully automated, dynamic column maximum bolding.
- The script now parses the mean accuracy from each cell and dynamically applies LaTeX bolding (`\mathbf` in Table 1 and `\textbf` in Table 2) to the row index that achieves the mathematical maximum in each column.
- Verified that all bolded cells in Table 1 and Table 2 are 100% correct and mathematically accurate.
