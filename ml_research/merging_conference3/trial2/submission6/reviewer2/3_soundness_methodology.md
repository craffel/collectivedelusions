# Evaluation Step 3: Soundness and Methodology

## Clarity of the Description
The description of the proposed framework, Q-Merge, is exceptionally clear, logical, and structured. Every component of the pipeline is mathematically formalized:
- **Layer-Wise Blending:** Formalized in Equations 1 and 2, which clearly show how task vectors are scaled at each depth boundary of the network.
- **Quantization Operator:** Formulated using standard per-channel symmetric uniform quantization in Equations 3 and 4, which clearly defines how dynamic channel-wise scale factors $S^l_c$ are computed as a function of the merging coefficients $\Lambda$.
- **Entropy Loss:** Shannon prediction entropy over the unlabeled calibration batch is mathematically defined in Equations 5 and 6.
- **Optimization Strategy (Zero-Order):** The black-box mutation 1+1 ES strategy, complete with the Rechenberg 1/5th success rule, is clearly detailed in Equations 7 and 8.
- **Optimization Strategy (First-Order):** The Straight-Through Estimator (STE) approximation and the dual-path gradient propagation are detailed in Equations 9–15. The mathematical derivation showing how gradients flow simultaneously through the direct coordinates and the dynamic scale factors is particularly rigorous and elegant, clearing up any ambiguity regarding automatic differentiation behavior.

## Appropriateness of Methods
The methods employed are highly appropriate, standard, and cohesive:
- **Layer-Wise Parameterization:** Prior model merging literature shows that different layers represent different levels of semantic abstraction. Defining layer-wise coefficients provides sufficient degrees of freedom to align weight spaces while keeping the parameter space extremely small.
- **Entropy Minimization:** Minimizing joint entropy is a well-established test-time adaptation technique that acts as a strong, data-private proxy for accuracy when labeled target data is unavailable.
- **Per-Channel Quantization:** Per-channel (channel-wise) weight quantization is standard in network compression. The paper's demonstration that per-channel is a strict necessity for low-bit (4-bit) merging to prevent catastrophic collapse is a highly appropriate and impactful architectural design guideline.
- **Straight-Through Estimator (STE):** STE is the standard and most reliable method for backpropagating gradients through discrete operators (like rounding) in network quantization.

## Potential Technical Flaws and Mitigations
The authors proactively identify and mitigate several potential technical and systems-level flaws:
1. **Trivial Class Collapse:** Unsupervised entropy minimization can theoretically lead to a model predicting a single class with high confidence. The authors address this by explaining that because the parameter search space is restricted to a low-dimensional layer-wise blending manifold ($\Lambda$), the model's structural capacity is highly constrained, providing a powerful implicit regularization that prevents degenerate class collapse.
2. **Backpropagation Activation Memory:** Reverse-mode automatic differentiation requires caching activation maps, which can be memory-intensive at test-time. The authors address this by: (a) discussing how standard reverse-mode AD only caches activations of active layers, avoiding frozen parameters; (b) highlighting compatibility with Forward-Mode AD (Jacobian-Vector Products), which completely eliminates activation caching because the search space is tiny ($|\Lambda| \le 128$); and (c) offering the derivative-free 1+1 ES variant which requires absolute zero activation memory overhead.
3. **Imbalanced Calibration Streams:** Under non-stationary edge environments, an imbalanced stream can skew the optimized coefficients. The authors propose the "Confidence-Based FIFO Stratification" heuristic to accumulate balanced task batches on-device with negligible overhead, and empirically validate its effectiveness.
4. **Scale Factor Discretization:** Embedded hardware often requires fixed-point scale factor arithmetic. The authors analyze the sensitivity to scale bit-widths ($N_{\text{fraction}} \in \{8, 16, 32\}$) and show that the continuous blending coefficients naturally "absorb" the scale discretization noise.
5. **Activation Quantization:** The authors evaluate joint weight-activation quantization (W8A8 and W4A4) using a dual-STE formulation to ensure the method operates under integer-only execution.

## Reproducibility
The methodology is exceptionally reproducible. The paper specifies:
- The network backbone used (`vit_tiny_patch16_224` from `timm`).
- The training hyperparameters for the task-specific experts (5 epochs, Adam, specific learning rates for backbone vs. heads).
- The compact calibration split size (16 images per task, 64 total) and task composition (MNIST, FashionMNIST, CIFAR-10, SVHN).
- The exact layer-wise grouping ($L=14$ groups).
- Evaluation across three independent random trials and seeds (42, 100, 2026) with standard deviations reported.

The mathematical completeness of the derivations in the main paper and Appendix, combined with standard PyTorch-compatible formulations, ensures that any expert reader can easily reproduce the Q-Merge framework.
