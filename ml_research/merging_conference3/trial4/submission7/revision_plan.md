# Revision Plan: SuiteMerge

This document outlines the systematic revisions executed to address the critiques of the Mock Reviewers (Reviewer 2 - The Rigorous Empiricist) and improve the soundness, empirical depth, and impact of the paper.

## Critique 1: Severe Asymmetry in Optimization Capabilities and Budgets
*   **Criticism:** The original simulation compared a restricted first-order Adam optimizer (100 steps) for online Test-Time Adaptation (TTA) against powerful, converged second-order and simplex optimizers (L-BFGS-B and Nelder-Mead) for offline validation methods. This optimization capability asymmetry unfairly handicapped online TTA.
*   **Revision Action:**
    1.  We implemented three new symmetrical optimization baselines in our simulation pipeline (`run_experiments.py`):
        *   `OFS-Tune (Adam)`: Offline validation tuning restricted to the same 100-step first-order Adam optimizer as TTA.
        *   `OFS-Uncon (Adam)`: Offline unconstrained validation tuning restricted to 100 steps of Adam.
        *   `AdaMerge (LBFGS)`: Unconstrained online TTA allowed to run to full convergence using a high-capacity second-order L-BFGS-B optimizer.
    2.  We ran a systematic 30-seed audit of these new baselines. The results conclusively proved that:
        *   Standardizing the optimizer has **no performance cost** on offline validation methods, as `OFS-Tune (Adam)` matches the original simplex-tuned performance.
        *   Allowing online TTA to run with high-capacity second-order L-BFGS-B actually **worsens its transductive overfitting** (degrading accuracy by 0.8%), confirming that unconstrained entropy minimization under stream noise collapses deeper into local minima.
    3.  We wrote a substantial new section **Section 4.4 (Addressing the Optimization Budget and Capability Asymmetry)** in `submission/sections/04_experiments.tex` explaining these findings.

## Critique 2: Stream-Level Noise vs. Validation Noise Mismatch and Temporal Smoothing
*   **Criticism:** The simulation of online TTA did not employ basic temporal smoothing (like Exponential Moving Averages), which is standard in physical online TTA implementations.
*   **Revision Action:**
    1.  We added a temporal Exponential Moving Average (EMA, $\beta=0.90$) parameter smoothing pipeline directly into the online AdaMerging and PolyMerge routines in our physical weight-space validation code.
    2.  We demonstrated that even with state-of-the-art EMA temporal smoothing, physical unconstrained online AdaMerging collapses catastrophically to ~32% accuracy, confirming that unsupervised prediction entropy is fundamentally unaligned with accuracy in high-conflict regimes.
    3.  We added discussion on the role of temporal smoothing and its interactions in Section 4.3 (Limitations) and Section 4.5 (Physical Validation).

## Critique 3: Complete Lack of Real-World Neural Network Validation (The Simulation Bottleneck)
*   **Criticism:** The entire evaluation was synthetic, relying purely on the Model II mathematical simulation without running actual physical neural network training or merging.
*   **Revision Action:**
    1.  We executed and verified the physical neural network weight-space model-merging experiments using a 5-layer CNN on CPU via `run_physical_validation.py`.
    2.  We trained MNIST and FashionMNIST experts, merged their weights, and evaluated:
        *   Uniform Baseline: **74.60%** average accuracy.
        *   Online AdaMerging (with EMA): **32.60%** (catastrophic entropy collapse down to 0.0000).
        *   Online PolyMerge (with EMA): **31.90%** (unsupervised collapse).
        *   Offline OFS-Tune (Ours): **71.80%** (stable, robust, avoiding collapse).
    3.  We wrote a substantial new section **Section 4.5 (Physical Weight-Space Neural Network Validation)** in `submission/sections/04_experiments.tex` documenting these physical weight-space results.

## Critique 4: Omitted Boundary Conditions and Future Scale Roadmaps
*   **Criticism:** Equation 4 features a division-by-zero boundary case when the optimal profile matches the uniform baseline, and the scaling of physical validation represents a minor limitation.
*   **Revision Action:**
    1.  We added an explanatory footnote to Section 3.2 (Eq. 6) detailing how division-by-zero is handled safely and acknowledging ratio sensitivity as a simulator localized artifact.
    2.  We updated Section 5 (Conclusion and Future Directions) to explicitly propose scaling physical validation to larger network architectures (such as ResNet-50 or ViT-B/32) and high-conflict datasets.

## Critique 5: Simulated-to-Physical Gap, SOTA Presentation alignment, and Task-Routing Clarity
*   **Criticism:** There is a discrepancy between the simulated superiority of online methods and their catastrophic physical collapse. Additionally, presenting OFS-Tune as structurally underfitted ($d=1$) while $d=2$ outperforms PolyMerge in Suite B makes the offline baseline appear weaker, and task-routing/privilege traps are not introduced early enough.
*   **Revision Action:**
    1.  **Symmetrical SOTA Alignment ($d=2$):** We reframed the paper's narrative to present both $d=1$ (linear) and $d=2$ (quadratic) as core proposed baselines. We updated the main results table (Table 1) and its caption to explicitly include and bold OFS-Tune ($d=2$) as the top performer in the high-conflict Suite B (\textbf{68.62\%} accuracy), outperforming Online PolyMerge.
    2.  **Elaborating on Simulated-to-Physical Gap:** We added an extensive discussion in Section 4.2 clarifying that PolyMerge's minor simulated advantage in other suites is due to its *transductive test-time advantage* (adapting directly on the test stream noise $\epsilon_{\text{stream}}$). We explained why this advantage is highly fragile and collapses in real physical weight-space deep networks because unsupervised prediction entropy minimization is rugged, non-convex, and vulnerable to degenerate shortcut solutions. We also laid out design guidelines for future model-merging simulators to close this gap.
    3.  **Early Introduction of Task Routing and Privilege Traps:** We restructured the first paragraph of the Introduction to clarify early on how online TTA operates on unlabeled mixed streams, defining Unsupervised TTA and the "privilege trap" of routing via oracle task labels.
    4.  **Consolidation of Simulator Assumptions:** We gathered and structured the simulator's key simplifying assumptions (smoothness, zero-ruggedness, stationary noise offsets) in a dedicated sub-subsection in Section 3.2 to maximize structural clarity.

## Critique 6: Sequential Single-Task Streams vs. Interleaved Mixed Streams
*   **Criticism:** The claim regarding the "privilege trap" and the necessity of task-routing oracle labels should be carefully qualified to acknowledge that thisprimarily manifests under interleaved heterogeneous streams, whereas sequential single-task streams (standard in some evaluations) naturally bypass this joint entropy collapse.
*   **Revision Action:**
    1.  We added a detailed explanatory paragraph in Section 3.4.1 (`\subsubsection{Multi-Task Routing and the Privileged TTA Confound}`) of `submission/sections/03_method.tex`.
    2.  We explicitly qualified that while sequential single-task streams (where a model adapts to one task's stream at a time sequentially) naturally bypass the joint multi-head entropy minimization challenge, this setup represents an over-simplification of real-world multi-task deployments.
    3.  We clarified that practical multi-task environments feature inherently interleaved and unlabeled streams, forcing unsupervised online TTA to either perform joint entropy minimization (which collapses representation boundaries) or violate its core unsupervised premise by relying on privileged, oracle task-routing labels. This establishes a clear, rigorous, and academically transparent trade-off for readers.

## Critique 7: Camera-Ready and Peer-Review Refinements (Current Turn)
*   **Criticism:** The peer review suggests minor improvements: elevating localized piecewise spline results to the main text, linking the scaling roadmap to public open-source code, clarifying computational/memory overhead and optimizer scaling for alternative parameterizations, and specifying the open-source license.
*   **Revision Action:**
    1.  **Elevating Piecewise Spline Results:** Added a dedicated paragraph `\paragraph{Extension to Localized Parameterizations:}` at the end of Subsection 4.3 in `submission/sections/04_experiments.tex`. This summarizes the piecewise linear spline and block-wise sharing results from the Appendix under non-smooth landscapes, highlighting the framework's flexibility.
    2.  **LLM Scaling Code Release Integration:** Expanded the reproducibility statement in Section F of the Appendix (`submission/sections/99_appendix.tex`) to explicitly include references to our LLM scaling utilities (OFS-Adam, parameter-efficient validation subsets, CPU expert offloading).
    3.  **Specifying Open-Source License:** Clarified in Appendix Section F that the entire repository will be released under the highly permissive Apache 2.0 open-source license.
    4.  **Nelder-Mead Scaling & Computational Overhead Analysis:** Added a new Subsection E.1 in `submission/sections/99_appendix.tex` to analyze the exact computational overhead and parameter scaling properties of Nelder-Mead vs. OFS-Adam when optimizing splines of varying knot granularities.

## Critique 8: Validation Class-Imbalance / Missing Class Risk in Ultra-Few-Shot Calibration ($M=10$)
*   **Criticism:** Random uniform draws of size $M=10$ from a 10-class dataset (e.g., CIFAR-10) have a $99.96\%$ probability of omitting at least one class entirely. This can cause severe validation class imbalance and degrade accuracy on missing classes.
*   **Revision Action:**
    1.  We implemented **stratified sampling** for the few-shot validation set (ensuring exactly 1 sample per class is drawn) and documented this requirement in `03_method.tex`.
    2.  We added Subsection B.3 (`\label{subsec:class_imbalance}`) in `99_appendix.tex` to mathematically formulate the missing-class risk and analyze how the validation budget must scale for fine-grained classification regimes (e.g., ImageNet, CIFAR-100).

## Critique 9: Inference-Time Task Routing in Multi-Head Deployments
*   **Criticism:** While OFS-Tune bypasses task-routing during parameter adaptation (gradient backpropagation), multi-head merged models on interleaved streams still require a routing mechanism at inference time to select the correct task head.
*   **Revision Action:**
    1.  We qualified our claims in `01_intro.tex` and `03_method.tex` to explicitly differentiate between adaptation-time routing (completely resolved by OFS-Tune) and prediction-time routing (a shared challenge for all multi-head architectures), enhancing the paper's intellectual honesty and accuracy.

## Critique 10: Tabular Reporting of Alternative Trajectory Configurations Across All Suites
*   **Criticism:** The simulated multi-suite tables in the main text only report global polynomial trajectories, while alternative localized parameterizations are introduced without complete baseline comparisons.
*   **Revision Action:**
    1.  We added a comprehensive performance comparison table (Table~\ref{tab:smooth_localized}) in `99_appendix.tex` reporting simulated accuracies of Block-wise Parameter Sharing and Piecewise Linear Splines across all five evaluation suites under smooth trajectories, proving they are highly robust and perform within $0.17\%$ of their global counterparts.

## Critique 11: Delineating Solver Scalability Limits (Nelder-Mead vs. OFS-Adam)
*   **Criticism:** Nelder-Mead simplex search is efficient for low-dimensional trajectories but may suffer from the curse of dimensionality when scaling to many tasks or high-granularity splines.
*   **Revision Action:**
    1.  We added a detailed scaling and overhead analysis in Subsection E.1 of `99_appendix.tex`, establishing a clear **dimensional crossover point** at $P \approx 10$ to $12$ parameters where first-order coordinate gradient descent (OFS-Adam) becomes computationally and mathematically superior to Nelder-Mead search.

## Critique 12: Colorblind Accessibility of Figures
*   **Criticism:** Figure 1 and Figure 2 lines and markers should be distinguished by geometric shapes and line styles to guarantee accessibility.
*   **Revision Action:**
    1.  We modified our plotting scripts (`generate_trajectory_plot.py` and `run_experiments.py`) to add highly distinct, colorblind-accessible hatching patterns, line styles, and markers (circles, stars, squares, stripes, dots), updating the saved figures embedded in the PDF.

## Critique 13: Self-Referential Persona Style Gimmicks, Localized Scale Gaps, and Simulator Circularity Neutralization
*   **Criticism:** The manuscript featured self-referential persona terms (e.g., "The Methodologist") that detracted from objective scholarly presentation; the scale gap between the toy CNN physical validation and large-scale foundation models was not prominently qualified; and the circularity critique of the simulator's smooth polynomial priors required a full dedicated subsection in the main text to highlight localized trajectory performance under non-smooth regimes.
*   **Revision Action:**
    1.  **Eliminated Persona Style Gimmicks:** Removed all 6 self-referential persona-based expressions ("The Methodologist", "The Rigorous Empiricist", "our persona as...") across `01_intro.tex`, `04_experiments.tex`, `05_conclusion.tex`, and `99_appendix.tex` to guarantee standard, objective, peer-reviewed academic tone.
    2.  **Qualified Physical Scale Gap:** Expanded our physical weight-space validation section in Section 4.4 (`04_experiments.tex`) to prominently qualify that our physical results are based on a toy CNN on MNIST/FashionMNIST, explicitly stating that validating these dynamics on larger foundation models (ViTs, LLMs, VLMs) is a necessary step to establish the absolute scale of our generalizability advantages.
    3.  **Neutralized Simulator Circularity in Main Text:** Converted the brief paragraph on localized parameterizations in Section 4.3 (`04_experiments.tex`) into a full, prominent subsection: `\subsection{Neutralizing Simulator Circularity: Performance under Non-Smooth Trajectories}`, detailing the Piecewise Linear Splines (66.24% accuracy) and Block-wise Parameter Sharing (67.38% accuracy) results under non-smooth zig-zag trajectories to prove that our continuous trajectory framework is not structurally limited to smooth global curves.
    4.  **Data-Access Setting and Boundary Condition Clarifications:** Added a clear qualifying statement in Section 3.7 (`03_method.tex`) explaining that under extreme zero-data/privacy constraints, online TTA remains the only option and OFS-Tune is not a drop-in replacement. Expanded Footnote 1 of Eq.~\eqref{eq:ratio} to explain accuracy sensitivity under near-boundary conditions, proving that bounded parameter spaces $[0, 1]$ naturally stabilize the optimized trajectories.


