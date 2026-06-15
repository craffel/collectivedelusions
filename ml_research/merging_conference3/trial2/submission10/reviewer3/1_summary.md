# Paper Summary

## Main Topic and Approach
The paper presents a deconstructive study of **AdaMerging**, a state-of-the-art unsupervised, test-time weight-space model merging scheme. Model merging integrates multiple task-specific expert models (fine-tuned from a shared pre-trained base) into a single multi-task model without retraining on original data or accessing it. AdaMerging optimizes merging scaling coefficients (either task-wise or layer-wise) using a small unlabeled test calibration batch by minimizing the average Shannon entropy of the merged model's predictions.

The authors deconstruct AdaMerging to identify and analyze two primary optimization phenomena:
1. **The Overfitting-Optimizer Paradox**: The authors argue that while layer-wise AdaMerging (optimizing $L \times T$ parameters) captures vital structural neural network hierarchy and representational routing, it is prone to transductive test-time overfitting due to the unsupervised entropy objective.
2. **The Spatial Averaging Paradox**: Direct test-time optimization of a low-dimensional bottleneck (exactly $T$ parameters, one global scalar per task) fails spectacularly (Task-wise AdaMerging drops to $81.19\%$ accuracy, below uniform initialization at $84.64\%$). Yet, taking the post-hoc spatial mean of layer-wise optimized coefficients (Spatial Averaging) acts as a regularizer, recovering a solid $84.96\%$ accuracy and beating the Task Arithmetic baseline.

The authors explain the second paradox via **multi-task gradient imbalance** caused by uncalibrated prediction entropy, where easy classification tasks (with naturally sharp logit distributions) dominate the optimization landscape, leading to destructive parameter interference and performance collapse on harder, more heterogeneous tasks (e.g., SVHN).

## Key Findings
1. **Intra-Task Layer Shuffling Collapses Performance**: Permuting layer-wise coefficients of each task across different layers of the visual encoder causes a performance collapse (from $88.05\%$ down to $78.61\%$). This confirms that learned coefficients are structurally specialized and tailored to the architectural hierarchy of the network.
2. **Spatial Averaging Regularizes Overfitting**: Averaging the optimized layer-wise coefficients post-hoc reduces degrees of freedom to exactly $T$ parameters, acting as a spatial low-pass filter. It achieves $84.96\%$, outperforming static Task Arithmetic ($84.64\%$), but trade-offs $3.09\%$ accuracy compared to the unconstrained layer-wise AdaMerging ($88.05\%$).
3. **Task-wise AdaMerging Pathologies**: Direct optimization of flat task-wise scales fails. The authors prove that this is due to multi-task gradient imbalance where uncalibrated entropy objectives on easy tasks dominate the joint objective under low-dimensional bottlenecks.
4. **Calibrated Prediction Entropy Fails**: Normalizing task losses at initialization (Calibrated Task-wise AdaMerging) does not resolve the bottleneck issue; accuracy remains low ($80.59\%$), proving that the low-dimensional constraint itself causes destructive weight-space interference.
5. **Architectural Routing Dynamics**: Early layers (Layers 1-4) maintain extremely high CKA similarity ($>0.995$) with task-specific experts across methods, while late layers (Layers 8-12) exhibit task-specific specialization in layer-wise AdaMerging.

## Explicitly Claimed Contributions (with Evidence)
- **Contribution 1**: Deconstructing the Overfitting-Optimizer Paradox in layer-wise AdaMerging. *Evidence*: Section 3.3 and 4.2.1 analyze Intra-Task Layer Shuffling and Spatial Averaging, showing the performance changes from $88.05\%$ to $78.61\%$ and $84.96\%$ respectively.
- **Contribution 2**: Discovering and explaining the Spatial Averaging Paradox. *Evidence*: Section 3.4 and 4.2.2 show that direct Task-wise AdaMerging fails ($81.19\%$), and provide mathematical and gradient analysis of the uncalibrated prediction entropy landscape.
- **Contribution 3**: Proposing Spatial Averaging as an elegant low-pass filter. *Evidence*: Section 3.3 and 4.2.1 show that post-hoc Spatial Averaging achieves $84.96\%$ without ground-truth labels.
- **Contribution 4**: Extensive seed-controlled evaluations of landscape flatness and representational similarity. *Evidence*: Section 4.3 and 4.4, including Table 2 and Figures 1(b), 1(c), and 2.
