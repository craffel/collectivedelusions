# Paper Summary: Task-Space Anchor Regularization (TSAR)

This paper addresses the critical and previously unaddressed problem of **low-data overfitting in dynamic model merging**. While dynamic model-merging techniques utilize lightweight routing layers to adaptively combine specialized expert models on the fly, they suffer from severe overfitting and representation-space collapse when calibrated on extremely data-scarce splits (e.g., $B_{cal} \le 64$ samples total across all tasks).

To resolve this, the authors propose **Task-Space Anchor Regularization (TSAR)**, which anchors layer-wise routing weights directly to pre-computed centroids (anchors) of pre-trained expert representations in a low-dimensional projected coordinate space.

Additionally, the paper exposes and resolves three other critical, systems-level issues in dynamic model merging:
1. **Multi-Task Gradient Cross-Talk**: Under low-data constraints, harder tasks (like SVHN or CIFAR-10) dominate the joint optimization gradient, driving simpler tasks (like MNIST) to drift from their anchors and collapse. The authors integrate **Projecting Conflicting Gradients (PCGrad)** to balance gradients and resolve this scaling anomaly.
2. **Heterogeneity Collapse**: Under mixed-task (heterogeneous) deployment streams, batch averaging of unconstrained linear routing coefficients causes positive and negative coefficients to cancel out, neutralizing routing. The authors propose a **Sigmoid-activated router** (scaled by 1.5) to keep coefficients non-negative, preventing cancellation with zero runtime overhead.
3. **Layer-Averaging Collapse**: The authors mathematically prove that layer-wise routers collapse to a single global router at inference time, yet they demonstrate that training the over-parameterized multi-layer router provides a beneficial ensembling effect (via gradient damping) that reduces seed variance during calibration.

### Key Contributions and Results:
- **Catastrophic Overfitting Collapse**: The authors systematically expose and document that standard unregularized dynamic routers collapse under low-data constraints ($B_{cal} \le 64$).
- **Substantial Accuracy Improvements**: On a 14-layer representation-space sandbox across 4 tasks (MNIST, FashionMNIST, CIFAR-10, SVHN), TSAR + PCGrad achieves a joint mean accuracy of **57.06%** (improving on standard $L_2$-only routing by **+12.34%**, Static Uniform Merging by **+5.20%**, and the "quantum-inspired" SOTA QWS-Merge by **+17.18%** absolute).
- **Physical Weight-Space Model Merging**: Validation on a real Vision Transformer (`vit_tiny_patch16_224`) shows a **+13.90%** absolute accuracy margin improvement over Static Uniform Merging.
- **Scientific Transparency**: The authors openly discuss and mathematically analyze several core properties, including:
  - The layer-averaging collapse of multi-layer routers.
  - The mathematical equivalence of classification head-level weight merging and output-level logit ensembling.
  - The superior performance of data-independent **Random Gaussian projection** over unsupervised PCA under extreme data scarcity ($B_{cal} \le 32$).
