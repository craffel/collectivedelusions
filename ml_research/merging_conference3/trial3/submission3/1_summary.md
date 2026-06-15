# Summary of the Paper

The paper introduces **FlatMerge**, a robust and memory-efficient Test-Time Adaptation (TTA) framework for model merging coefficients on edge devices. The authors identify a key vulnerability in existing TTA-based model merging techniques (such as AdaMerging) under real-world physical input noise (e.g., sensor noise, blur, compression artifacts), which they term **Noise-Entropy Collapse**. When the test-time input stream is corrupted, standard entropy minimization overfits the transductive noise, causing the layer-wise blending coefficients to drift catastrophically and degrade the model's out-of-distribution (OOD) generalization.

To resolve this "Overfitting-Optimizer Paradox" under noise, FlatMerge proposes a **dual-regularization mechanism**:
1. **Subspace-Constrained Blending (PolyMerge):** It restricts the layer-wise blending coefficients to a low-degree polynomial of normalized layer depth. This reduces the optimization parameter space by over 78% and filters out high-frequency spatial optimization noise.
2. **Zeroth-Order Flatness-Aware Randomized Smoothing:** It optimizes a smoothed entropy objective over randomized perturbations of the compact polynomial coefficient space. Because the optimization is formulated using Zeroth-Order (gradient-free) randomized smoothing, it requires zero weight-space backpropagation and zero activation memory caching during adaptation.

**Key Hardware and Practical Advantages:**
- **Zero Activation Memory Caching:** The peak adaptation memory is identical to standard forward-pass inference, preventing SRAM overflows on edge devices.
- **Asynchronous, Periodic Adaptation:** To mitigate the latency penalty of zeroth-order search ($3.73\times$ step latency), the authors propose updating coefficients periodically in the background (e.g., every 100 steps) on a small buffered batch. This amortizes the step latency overhead to a negligible $0.027\times$ (a mere 0.73% increase).

**Evaluation and Validation:**
- **Calibrated Numerical Simulation:** Mimics a CLIP ViT-B/32 backbone on a 4-task classification benchmark (MNIST, FashionMNIST, CIFAR-10, SVHN) across two loss landscapes (Convex Sandbox and Coupled Non-Convex Stress-Test) using 15 random seeds. Shows that FlatMerge achieves SOTA robustness and reduces seed variance by over 60% compared to PolyMerge and AdaMerging.
- **Physical MLP Validation:** Merges a 3-layer MLP fine-tuned on MNIST and FashionMNIST under pixel-level Gaussian noise, showing that standard first-order AdaMerging collapses while FlatMerge prevents transductive drift and improves performance.
- **Physical 5-layer CNN Validation:** Merges a 5-layer CNN fine-tuned on MNIST, FashionMNIST, and KMNIST, demonstrating that first-order AdaMerging collapses catastrophically (to ~16.67% joint accuracy), whereas ZO-FlatMerge successfully prevents collapse and achieves 48.57% joint accuracy.
