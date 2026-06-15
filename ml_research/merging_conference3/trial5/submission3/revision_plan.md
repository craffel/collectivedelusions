# Robust Linear Routing (RLR) Revision Plan

This document outlines our systematic plan to address all weaknesses, critical flaws, and areas for improvement identified by the mock reviewers. By resolving these issues, we elevate the scientific rigor, transparency, and academic integrity of our submission.

---

## 1. Addressing Critical Flaws & Major Weaknesses (Current Revision Round)

### Critical Flaw 1: Discrepancy Between Claimed Unweighted Loss and Codebase Task-Balanced Weighting
* **Critique:** The paper's mathematical presentation (Section 3.3, Equation 5) claims RLR minimizes a clean, unweighted uniform multi-task calibration loss. However, the codebase implementation computed task-balancing weights inversely proportional to initial expert performance, creating a critical math-code contradiction.
* **Revision Action:** 
  1. We have updated all relevant Python codebase files (`run_experiments.py`, `run_seed_sweep.py`, and `run_seed_sweep_fast.py`) to use a clean, unweighted uniform multi-task calibration loss formulation (`task_weights = np.ones(4)`).
  2. This aligns the codebase implementation perfectly with the mathematical claims in Section 3.3, removing all heuristics, proxies, and math-code contradictions, which enforces our Minimalist research philosophy.
  3. We are running the experiments again with this unified unweighted calibration loss to ensure all results, plots, and tables are fully updated and accurate.

### Major Weakness 2: Clear Framing of RLR's Empirical Necessity in Homogeneous Settings
* **Critique:** The 5-seed sweep results (Section 4.3) show that the unregularized classical Linear Router ($91.53\% \pm 0.41\%$) and RLR ($91.46\% \pm 0.42\%$) are statistically indistinguishable in homogeneous settings, demonstrating that unregularized routing is already highly robust. The paper should make this clearer and position RLR's regularizations as specialized stabilizers for heterogeneous mixed-batch streams and OOD shifts.
* **Revision Action:**
  1. Updated `00_abstract.tex` and `01_intro.tex` to explicitly frame the classical unregularized Linear Router as already highly robust under standard homogeneous environments when trained using standard practices.
  2. Positioned RLR's weight decay and temperature scaling as specialized stabilizers for heterogeneous mixed-batch streams and out-of-distribution shifts rather than essential tools to prevent homogeneous SVHN collapse.

---

## 2. Addressing Scholarly & Presentation Improvements

### Area 1: Contextualization of Gating Regularization in Mixture-of-Experts (MoE) Literature
* **Critique:** Frame and contextualize gating/routing network regularizations (such as load-balancing losses and entropy/temperature scaling) with foundational sparse MoE literature (e.g., Shazeer et al., 2017; Fedus et al., 2022).
* **Revision Action:**
  1. Updated `02_related_work.tex` (Section 2.3) to provide a deep scholarly contextualization of gating regularizations, citing sparse MoE foundational pillars that prevent expert collapse and stabilize routing.
  2. Appended corresponding high-quality BibTeX records for `shazeer2017outrageously` and `fedus2022switch` to the end of `submission/references.bib`.

### Area 2: Diagnosing the Cause of Prior Work's Reported Collapse
* **Critique:** Add a quick diagnostic checklist or configuration comparison table (Vance's settings vs. Ours) to help future researchers set up stable classical routing baselines.
* **Revision Action:**
  1. Inserted a new table (Table 2: `tab:diagnostic`) in Section 4.2 (`04_experiments.tex`) comparing collapse-prone configurations (deep representation source, high learning rates, over-optimization, lack of L2 decay or temperature scaling) from prior work against our stable, robust configuration.
  2. Provided a detailed textual walkthrough of these three critical factors to assist future baseline setup.

### Area 3: Scaling to High-Dimensional Parameter Spaces (LLMs)
* **Critique:** Expand the future work discussion in Section 5 to outline concrete pathways or cite insights on scaling classical regularized gating to high-dimensional spaces (LLMs).
* **Revision Action:**
  1. Substantially expanded the future work and LLM discussion in `05_conclusion.tex`.
  2. Outlined three concrete scaling pathways: Sequence-Level Pooled Routing Signals, Routing over Low-Rank Adapter (LoRA; citing Hu et al., 2022) Experts, and leveraging Linear Mode Connectivity properties in high-dimensional weight spaces.
