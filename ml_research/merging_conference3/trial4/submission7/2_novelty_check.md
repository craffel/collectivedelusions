# 2. Novelty and Literature Context

## Conceptual Novelty
The primary conceptual novelty of this paper lies in the **systematic auditing and exposure of Task Suite Bias** within the model-merging literature. While "reality-check" papers are established in other machine learning subfields (such as metric learning and domain generalization), this is the first work to critically dissect the evaluation protocols of adaptive model-merging algorithms.
- Exposing that the standard four-dataset pool (MNIST, FashionMNIST, CIFAR-10, SVHN) is evaluated as a single monolithic block (Suite E) and that this specific mixture masks local representation collapse on high-conflict sub-components (such as Suite B: CIFAR-10 + SVHN) is a highly original and high-signal insight.
- The identification and formalization of the "privilege trap" (where online TTA methods rely on implicit oracle task-routing labels at inference time to route gradients through correct task heads on interleaved mixed streams) is another important contribution that clarifies realistic deployment assumptions.
- Differentiating between adaptation-time task routing (completely bypassed by OFS-Tune since parameters are static at deployment) and inference-time prediction routing (a shared challenge for all multi-head architectures) is a highly honest and nuanced conceptual contribution.

---

## Methodological and Algorithmic Novelty
The proposed alternative, **OFS-Tune** (Offline Few-Shot Validation Tuning), leverages two existing ideas in a novel, robust combination:
1. **Continuous Polynomial Trajectory Constraints:** Restricting merging coefficients to a low-degree polynomial of network depth was introduced by PolyMerge to reduce the online search space. However, this paper is the first to apply this constraint *offline* as an analytical low-pass filter to reject few-shot validation sampling noise.
2. **Offline Few-Shot Validation Tuning with Stratified Sampling:** While validation tuning is a standard approach, the paper's formulation combines it with stratified sampling to address the severe risk of class omission and imbalance in ultra-few-shot settings ($M=10$). The authors provide a detailed mathematical proof showing that random uniform draws have a 99.96% probability of omitting classes, which they resolve with stratified calibration draws.
3. **Alternative Localized Trajectory Formulations:** To capture localized sensitivity spikes (e.g., attention-MLP boundaries in Transformers) while maintaining a low-dimensional search space, the authors introduce **Piecewise Linear Splines** and **Block-wise Parameter Sharing**. This represents a highly novel, flexible extension proving that the continuous trajectory framework is not structurally limited to global polynomials.

The authors also introduce **OFS-Unconstrained** (optimizing unconstrained layer-wise coefficients offline) as a crucial ablation baseline. This allows them to mathematically isolate the regularizing value of the polynomial trajectory constraint from the effect of having labeled data, which is an elegant and rigorous methodological choice.

---

## Comparison with Existing Literature
- **vs. AdaMerging (TTA):** AdaMerging optimizes unconstrained layer-wise coefficients online via unsupervised entropy minimization on evaluation streams. This paper proves both theoretically and physically that AdaMerging's high parameter capacity (e.g., 48 parameters for a 12-layer ViT) overfits to local transductive stream noise and collapses under high representational conflict. OFS-Tune resolves this by shifting optimization offline, utilizing a supervised objective, and restricting parameter capacity to a low-degree polynomial.
- **vs. PolyMerge (TTA):** PolyMerge restricts coefficients to a polynomial trajectory but still optimizes them online over unlabeled streams. The paper demonstrates that PolyMerge's simulated advantage in some suites is a fragile artifact of a "transductive test-time advantage" (fitting specific test-stream noise). In real physical weight spaces, unsupervised prediction entropy minimization remains highly non-convex and rugged, causing PolyMerge to perform poorly, whereas offline OFS-Tune generalizes robustly.
- **vs. Static Merging (Task Arithmetic, Ties-Merge):** Static merging methods rely on uniform, hand-tuned coefficients across all layers. This paper shows that Uniform merging collapses under high-conflict regimes (Suite B), whereas OFS-Tune successfully optimizes layer-wise sensitivities to recover multi-task capabilities.
