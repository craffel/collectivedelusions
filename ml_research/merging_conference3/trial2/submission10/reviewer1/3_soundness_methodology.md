# 3. Soundness and Methodology Evaluation

## Clarity of the Description
The methodology of the paper is exceptionally clear, mathematically rigorous, and well-structured.
- Equations (1) through (10) are clean, self-contained, and use precise notation to define the task vectors, layer-wise/task-wise merging configurations, prediction entropy loss, shuffling operations, spatial averaging, and the calibrated prediction entropy objective.
- The terminology is consistent throughout the manuscript, and the authors explicitly lay out their mathematical assumptions.
- The boundaries of the evaluation setup—specifically the use of **Oracle Routing** (where inputs are routed to task-specific heads to evaluate the merged backbone representation)—are clearly defined, discussed in footnotes, and situated relative to standard practice in the model-merging literature.

## Appropriateness of Methods
The methods chosen to deconstruct AdaMerging are elegant, creative, and highly appropriate:
1. **Intra-Task Layer Shuffling**: This is a brilliant diagnostic tool to probe whether the optimized coefficients are actually capturing layer-specific, hierarchical representational properties or if they are just random scaling noise.
2. **Spatial Averaging**: This method is a highly appropriate control to test whether the high-dimensional parameter space ($L \times T$ parameters) is strictly necessary, or if a regularized task-wise representation ($T$ parameters) can be recovered post-hoc.
3. **Calibrated Prediction Entropy**: Formulating this normalized objective allows the authors to perform a controlled experiment. By balancing the loss contributions of all tasks to exactly $1.0$ at initialization, they can isolate and test whether the "Spatial Averaging Paradox" is purely a consequence of gradient magnitude imbalance at initialization, or if it is a more fundamental structural bottleneck issue.
4. **Linear CKA (Centered Kernel Alignment)**: Utilizing CKA layer-by-layer across all 12 blocks is a highly appropriate representation analysis technique to empirically trace how weight-space updates affect the internal feature representations of the visual backbone.

## Potential Technical Flaws & Limitations
The paper's technical reasoning is exceptionally solid, but there are a few subtle limitations and nuances to consider:
- **Entropy as a Surrogate Objective**: The paper rightly points out that minimizing prediction entropy can lead to overconfident misclassifications by artificially inflating weight magnitudes (since logits are scaled up, driving softmax sharpness without functional correctness feedback). However, the paper could further discuss how this pathology interacts with task-specific visual heads. Since visual classification heads are trained separately on frozen CLIP features, the scale of the inputs to these heads matters immensely. Minimizing entropy on a merged backbone by scaling up backbone weights might over-amplify features, driving heads into saturated softmax regions.
- **Homogeneity vs. Heterogeneity of Tasks**: The main experiments are conducted on highly heterogeneous datasets (MNIST, FashionMNIST, CIFAR-10, SVHN). The paper briefly notes in Section 5 that this heterogeneity triggers the uncalibrated entropy imbalance. On more homogeneous datasets, this gradient imbalance might be less severe. Testing on a homogeneous dataset split (e.g., DomainNet) would have clarified the exact boundary conditions of the paradox, though the authors rightly note this as a limitation and future direction.
- **The "Oracle Routing" Boundary Condition**: While standard in model merging research, Oracle Routing bypasses the problem of identifying which head to use at test-time. If a real-world system deployed this model, it would require an auxiliary task classifier. While this is a known limitation of the entire sub-field, it should be highlighted as a practical deployment barrier.

## Reproducibility
The reproducibility of this submission is **excellent**:
- **Detailed Hyperparameters**: Appendix A provides comprehensive hyperparameter configurations for both the head fine-tuning phase (optimizer, learning rate, weight decay, epochs, batch size) and the test-time adaptation phase (optimizer parameters, learning rates, step counts, mutation noise, etc.).
- **Infrastructure and Framework**: The authors explicitly specify their computing environment, Python/PyTorch versions, and the exact model architecture (CLIP ViT-B/32).
- **Statistical Rigor**: The paper reports mean and standard deviation across **three independent, seed-controlled runs**, and partitions datasets into disjoint head-training, calibration, and evaluation splits.
- **Massive Evaluation Scale**: Evaluating on the full, standard test splits of all four datasets (totaling **56,032 images**) provides highly stable, statistically significant, and reproducible performance metrics.
- **No Private Datasets**: All four datasets (MNIST, FashionMNIST, CIFAR-10, SVHN) and the base model (CLIP ViT-B/32) are publicly available standard benchmarks.
