# 1. Summary of the Paper

## Core Topic and Approach
The paper introduces **GranMerge**, an empirical framework designed to investigate the **Generalization-Granularity Trade-off** in adaptive multi-task model merging at test time. 
Model merging combines multiple task-specific expert neural networks (sharing a common pre-trained base) into a single model without full retraining. Recent methods (e.g., AdaMerging) perform test-time adaptation on a small, unlabeled calibration stream to optimize merging coefficients. 
The paper systematically studies how the physical granularity of these merging coefficients affects multi-task generalization accuracy. It evaluates five nested levels of parameter resolution (Global, Layer-wise, Block-wise, Component-wise, and Tensor-wise) under two optimizer families (first-order Adam and zero-order 1+1 Evolution Strategies) and introduces two spatial-depth regularizers (Elastic Spatial Regularization and Total Variation depth-wise smoothness) to control overfitting.

## Key Findings
1. **The Generalization-Granularity Trade-off**: Increasing structural resolution of merging coefficients leads to severe transductive overfitting on compact, unlabeled test-time calibration streams ($N=256$). Although intermediate granularities (L2 to L4) perform better than extremely coarse global merging (L1), fine-grained unregularized tensor-wise merging (L5) suffers from generalization collapse.
2. **Optimizer-Specific Dynamics**: First-order Adam gradient descent is highly susceptible to rapid, chaotic overfitting because it aggressively exploits local prediction entropy. In contrast, zero-order 1+1 Evolution Strategies (ES) demonstrate higher test-set generalization robustness, which is attributed to both isotropic walk constraints (implicit regularization) and optimization sluggishness (underfitting due to the curse of dimensionality) in high-dimensional parameter spaces.
3. **Regularization Limitations**: While soft spatial-depth regularizers (Elastic Spatial Regularization and Total Variation) successfully mitigate some overfitting (e.g., recovering Level 5 performance from 29.43% to 30.17% for ES, and 26.91% to 28.51% for Adam), they fail to match the performance of the simple, zero-overhead static Uniform Task Arithmetic baseline (30.41%).
4. **Surrogate Loss Misalignment**: Minimizing unsupervised prediction entropy on small calibration batches does not guarantee accurate decision boundaries; optimizers frequently converge to "confident but incorrect" weight configurations.

## Claimed Contributions and Evidence
* **Systematic Empirical Framework**: Evaluates 5 levels of structural granularity across 4 classification datasets (MNIST, FashionMNIST, CIFAR-10, SVHN) using a 12-layer Vision Transformer, supported by exhaustive sweeps across 3 independent seeds.
* **Discovery of Optimization Sluggishness in ES**: Demystifies why zero-order methods appear to generalize better at high granularities, showing it is largely a byproduct of high-dimensional underfitting remaining close to the robust initialization.
* **Spatial-Depth Regularizers**: Proposes Elastic Spatial Regularization (ESR) and Total Variation (TV) depth-wise smoothness, demonstrating statistical recovery of Level 5 merging performance.
* **Practical Guidelines**: Provides clear, actionable deployment guidelines regarding optimizer selection and structural constraints.
