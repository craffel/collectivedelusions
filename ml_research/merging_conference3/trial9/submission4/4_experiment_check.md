# Experimental Validation and Empirical Rigor

This file evaluates the empirical validity of the paper's claims, analyzes the baseline robustness, and details the boundaries where Momentum-Merge's parsimonious assumptions are challenged.

---

## 1. Empirical Environment: The Sandbox Limitation
The experiments are situated entirely within the synthetic **Analytical Coordinate Sandbox (ICS)**. While this environment is highly structured and provides precise, coordinate-aligned task manifolds that allow for deep, controlled studies of cascading noise, it has **limited ecological validity**:
* No real-world large language models (e.g., LLaMA, Mistral) or vision backbones (e.g., ViT-Base) are evaluated.
* No standard downstream multi-task benchmarks (e.g., GLUE, MMLU, VTAB) are used.
* The "representation noise" modeled as isotropic Gaussian noise is highly synthetic and may not reflect the structured, semantic representation drift observed in physical deep architectures.

While the sandbox is sufficient for a proof-of-concept, the absolute empirical claims must be caveated as "sandbox-validated."

---

## 2. Evaluation Rigor & Multi-Seed Synchronization
To the authors' credit, the empirical methodology inside the sandbox is highly rigorous in two ways:
1. **Multi-Seed Evaluation:** All main results are evaluated across 10 independent random seeds, generating fresh coordinate manifolds and shuffled serving streams in each seed, providing reliable statistical means and standard deviations.
2. **Perfect RNG Synchronization:** As observed in `run_experiments.py` and `test_ablation.py`, the authors lock the RNG state prior to executing each ensembling method on a test sample. This guarantees that every method is evaluated on the exact same serving sequence and experiences the *exact same layer-wise noise realizations*. This synchronized comparative evaluation is exceptionally rigorous and represents high scientific standards.

---

## 3. Asymmetric Noise Stress-Test: ChemMerge's Dynamic Advantage
In Section 4.4.2 of the main text, the paper discusses task-asymmetric noise scales. Our empirical analysis of the asymmetric noise sweeps reveals a critical boundary where Momentum-Merge's constant-inertia assumption is challenged:
* Under **Extreme Noise Asymmetry** (where one task has massive representation noise while others are clean), **ChemMerge's state-dependent kinetics** provide a minor joint classification accuracy buffer of $+0.15\%$ to $+0.30\%$ absolute compared to Momentum-Merge.
* This occurs because ChemMerge's reaction rates scale with activation similarities. When a task is clean, it allows rapid routing plasticity (low inertia); when a task is noisy, it applies heavy smoothing (high inertia).
* However, Table 2 and Section 4.4.2 demonstrate that this minor accuracy buffer comes at a **catastrophic cost in routing stability**: ChemMerge's routing jitter surges rapidly from **0.0193** (Symmetric) to **0.0260** (Extreme Asymmetry), whereas Momentum-Merge (Advanced) maintains a near-zero routing jitter of **0.002955** (an over 8.8$\times$ reduction in routing oscillations).
* This prominent boundary analysis shows that constant-inertia EMA remains the superior engineering choice, as the tiny accuracy gain of dynamic kinetics is heavily outweighed by the massive increase in ensembling oscillations.

---

## 4. Summary of Empirical Claims vs. Ground Truth

| Claim in Paper | Simulated Empirical Ground Truth | Verdict / Critique |
| :--- | :--- | :--- |
| **Momentum-Merge (Base) achieves 74.85% classification accuracy.** | MM (Base) achieves **74.85%** classification accuracy in Table 1. | **Verified.** The text and Table 1 are now synchronized. |
| **Momentum-Merge (Advanced) achieves 74.98% classification accuracy.** | MM (Advanced) achieves **74.98%** classification accuracy in Table 1. | **Verified.** The text and Table 1 are now synchronized. |
| **Momentum-Merge matches or exceeds SOTA ChemMerge.** | MM Base (74.85%) beats ChemMerge (74.71%), and MM Advanced (74.98%) beats ChemMerge (74.71%). | **Verified.** However, both calibrated control baselines (SABLE + LC at 77.24% and ChemMerge + LC at 76.60%) outperform MM Advanced. |
| **Complex biochemical metaphors are entirely redundant.** | ChemMerge's dynamic reaction kinetics offer a minor $+0.15\%$ to $+0.30\%$ accuracy buffer under extreme noise asymmetry. | **Overstated.** While ChemMerge's dynamic inertia is slightly beneficial for accuracy under extreme heterogeneity, it causes catastrophic jitter surges. |
| **Momentum-Merge is robust and training-free.** | MM (Advanced) accuracy collapses from **74.98% down to 71.20%** when calibration samples drop to 8. | **Vulnerable.** MM is highly sensitive to calibration scarcity due to "recurrence trapping" of early initial routing errors. |
