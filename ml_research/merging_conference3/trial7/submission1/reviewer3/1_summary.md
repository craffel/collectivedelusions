# Summary: Layer-Averaging Collapse Deconstruction

## Main Topic
The paper focuses on weight-space dynamic model merging, where specialized, fine-tuned neural networks are merged on the fly during inference using sample-specific coefficients computed by an input-dependent routing network. It specifically aims to deconstruct a recent theoretical claim of "Layer-Averaging Collapse" (rank-1 collapse), which asserted that learned layer-wise routing trajectories must become perfectly collinear across layers, making layer-wise dynamic routing redundant compared to global (single-layer) routing.

## Proposed Approach
1. **Physical Dynamic Model-Merging Pipeline:** Evaluated directly on deep neural network backbones using Split-MNIST digit subsets.
2. **Bounded Sigmoid (BSigmoid) Router:** A novel gating mechanism that uses independent, element-wise Sigmoid activations followed by a sum-to-1 normalization. To project high-dimensional inputs with low parameter overhead, it projects inputs to a low-dimensional state ($d=8$) using a frozen random Gaussian projection.
3. **SVD Collinearity Audit:** Analyzes the dimensionality of the learned Batch-Averaged Layer-wise Coefficient Matrix $A \in \mathbb{R}^{L \times K}$ by computing the Collinearity Ratio $\rho_{collinear} = \sigma_1 / \sum \sigma_i$.
4. **Inter-Layer Cosine Similarity:** Computes pairwise cosine similarity between routing vectors at different layers to map spatial routing specialization.
5. **The Batch-Averaged Multi-Task Inference Paradox:** Articulates a fundamental conceptual limit where batch-averaging dynamic routing coefficients over heterogeneous batches collapses the model back to a static uniform compromise, while homogeneous batches make the merging redundant compared to direct Oracle expert routing.

## Key Findings
1. **Rebuttal of Rank-1 Collapse:** Under Cross-Domain task conflict, the Collinearity Ratio drops to $0.4987 \pm 0.08$ on DeepMLP-12 and $0.5673 \pm 0.03$ on TinyCNN-4, suggesting learned routing trajectories occupy a multi-dimensional subspace rather than collapsing to a rank-1 line.
2. **Emergence of Layer Specialization:** Pairwise inter-layer cosine similarity heatmaps show block-diagonal clustering under high-conflict task suites, indicating early layers and late layers learn distinct, specialized routing.
3. **Decoupled Gradients:** The independent BSigmoid router outperforms competitive Softmax routing, particularly on TinyCNN-4, which the authors attribute to decoupled gradient paths during calibration.
4. **Capacity-Variance Trade-off:** Under small calibration budgets (few-shot), the lower parameter footprint of static or global baselines acts as a regularizer, allowing them to outperform the high-capacity layer-wise router. Scaling the calibration budget to 1024 samples allows the dynamic router to cross over and surpass the static baselines.

## Explicitly Claimed Contributions and Stated Evidence
1. **Deconstruction of Rank-1 Collapse:** Evidenced by the SVD Collinearity Ratio dropping significantly below 1.0 (approaching 0.50) in Cross-Domain task environments.
2. **Emergence of Depth-Specialized Policies:** Evidenced by block-diagonal patterns in inter-layer cosine similarity maps for DeepMLP-12 under Cross-Domain conflict.
3. **The Capacity-Variance Trade-off:** Evidenced by ablations scaling the calibration split size $B$ from 64 to 1024 samples on TinyCNN-4, showing a performance crossover where the dynamic router eventually outperforms the static OFS-Tune.
4. **Bounded Sigmoid Gating Advantages:** Evidenced by direct comparison tables showing BSigmoid outperforming standard Softmax by up to ~25% on TinyCNN-4, supported by gradient norm tracking during calibration.
