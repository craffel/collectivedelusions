# Novelty Check Report (`2_novelty_check.md`)

## 1. Assessment of Key Novel Aspects
The paper proposes to evaluate the dimensionality of layer-wise dynamic model-merging routing trajectories. The primary claims of novelty lie in:
1. **Debunking the "Layer-Averaging Collapse" Theorem:** Showing that in realistic physical architectures with high task conflict, learned layer-wise coefficients do not collapse to a rank-1 subspace.
2. **The Bounded Sigmoid (BSigmoid) Router:** Introducing an independent, normalized sigmoidal gating activation to replace the competitive Softmax activation.
3. **SVD Spectral Audit:** Proposing the SVD Collinearity Ratio $\rho_{collinear}$ to measure the true dimensionality of layer-wise routing.
4. **The Batch-Averaged Multi-Task Inference Paradox:** Articulating a theoretical paradox where dynamic model merging under batch-averaging is either redundant or equivalent to static merging.

## 2. Critical Evaluation of the "Delta" from Prior Work
A closer, critical examination of these aspects reveals that the "delta" from existing literature is minor and highly incremental:
- **Divergence from "Layer-Averaging Collapse" (Prior Preprint `[anonymous]`):** The paper positions itself as a direct response to a single recent preprint. While it is useful to point out that over-simplified linear sandboxes do not capture full physical network dynamics, debunking a specific, possibly obscure, preprint's linear assumption is a very narrow contribution.
- **The BSigmoid Router is an Incremental Variant:** Gating mechanisms using sigmoids followed by sum-normalization are standard in machine learning (e.g., in multi-task learning, gating networks, and Mixture-of-Experts). Renaming a normalized sigmoid gating layer as "Bounded Sigmoid (BSigmoid)" does not represent a significant methodological breakthrough.
- **The SVD Diagnostic is a Direct Application of Standard Math:** Applying Singular Value Decomposition (SVD) to analyze the rank of a learned coefficient matrix is a standard linear algebra diagnostic. Its application here is a straightforward diagnostic step, not a novel theoretical development.
- **The "Batch-Averaged Paradox" is Conceptual, but Unresolved:** The authors deserve credit for identifying and thoroughly explaining the *Batch-Averaged Multi-Task Inference Paradox* in Section 3.5. This is perhaps the most intellectually honest and conceptually interesting section of the paper. However, the paper does not actually *solve* this paradox. It merely points it out and lists "future pathways" (such as LoRA-level dynamic merging or task-aware bucketing). In a physical evaluation paper, identifying a fundamental flaw that makes your proposed class of methods practically non-viable without actually offering a physical solution to it severely diminishes the paper's contribution.

## 3. Characterization of Novelty
The novelty of this submission must be characterized as **incremental** and **reconstructive/reactive**:
- **Incremental Methodology:** The proposed router and diagnostic tools are simple combinations of standard, pre-existing ML techniques (SVD, random projections, sigmoid gating, few-shot Adam calibration).
- **Reactive Scope:** The core narrative is framed around refuting a single theoretical claim from another paper. Instead of presenting a highly effective, ready-to-use new paradigm, the paper is primarily a critique of a prior preprint's assumptions, supported by an empirical framework that itself fails to show practical utility (it does not outperform static baselines on CNNs, achieves near-random accuracies on MLPs, and is plagued by a fundamental inference paradox).
- **Practical Impact is Negligible:** Because the authors' proposed dynamic layer-wise router consistently underperforms a simple static baseline (OFS-Tune) on TinyCNN-4, achieves unusable accuracies on DeepMLP-12 (barely above random guessing), and requires prior task labels to bypass the mixed-batch collapse, the practical utility of this work is virtually non-existent.
