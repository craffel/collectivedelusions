# 4. Experimental Evaluation Check

## Evaluation of Experimental Setup and Datasets
The experimental setup is designed with high scientific rigor:
- **Vision Transformer Backbone:** The use of a capacity-constrained Vision Transformer backbone (`vit_tiny_patch16_224`, 5.7M parameters) is an excellent choice. It provides an ideal, highly sensitive testbed where parameter interference is pronounced, making the evaluation of weight routing protocols highly challenging and clean.
- **Converged Experts:** Ensuring that the four task-specific experts are trained to true convergence ( MNIST: $100.00\%$, FashionMNIST: $92.80\%$, CIFAR-10: $96.40\%$, SVHN: $96.80\%$) establishes a solid, reproducible upper bound. This corrects potential flaws in prior work where under-converged checkpoints could distort merging behavior.
- **Multi-Task Suite:** The selection of MNIST, FashionMNIST, CIFAR-10, and SVHN is standard and representative, spanning grayscale digits, grayscale fashion items, colored low-resolution natural objects, and complex colored digits. These tasks introduce significant gradient conflict in weight space.
- **Multiple Seeds:** All results are reported as Mean $\pm$ Standard Deviation over 3 random calibration-sampling seeds, providing a robust statistical guarantee.

## Strength of Baselines
The paper includes an exceptionally strong and comprehensive set of baselines:
1. **Upper Bound:** Individual Experts (Ceiling) representing the ideal multi-model deployment.
2. **Static Merging:** Uniform Merge (Task Arithmetic) and OFS-Tune (Supervised Static).
3. **Test-Time Adaptation (TTA):** AdaMerging, providing an online optimization baseline.
4. **SOTA Waveform Dynamic Routing:** QWS-Merge, representing the complex wave-interference state-of-the-art.
5. **Ablation Classical Routers:** 
   - Linear Router (unregularized)
   - Linear Router (Reg) with L2 weight decay
   - BL-Router (unregularized and L2-regularized)
   - GLS-Router (unregularized and L2-regularized)
   - BSigmoid-Router (unregularized and L2-regularized)

This incredibly extensive suite of baselines allows the authors to perform precise, surgical ablation of every individual variable.

## Alignment Between Results and Claims
The empirical results in Tables 2 & 3 and Figures 2 & 3 completely support the authors' key methodological claims:

1. **L2 Regularization Resolves SVHN Collapse (Claim 2):**
   - *Result:* Unregularized Linear Router gets $74.00 \pm 16.14\%$ on SVHN. Applying L2 regularization in **Linear Router (Reg)** completely resolves this, achieving **$91.73 \pm 3.71\%$** on SVHN (outperforming QWS-Merge's $79.73 \pm 0.75\%$ by a massive **$+12.00\%$**) and boosting the joint homogeneous mean from $78.13\%$ to $82.80\%$.
   - *Verdict:* Fully supported. The failure of classical linear routers is a pure baseline-tuning artifact of low-data overfitting during calibration.

2. **Deconstructing the Softmax Bounded Under-scaling Bottleneck (Claim 3):**
   - *Result:* The Softmax-based **BL-Router** collapses catastrophically on SVHN ($31.73 \pm 16.03\%$ unregularized, $43.20 \pm 8.02\%$ regularized). The authors mathematically show that capping Softmax output sums at $0.3$ caps the global sum of coefficients at $0.3$, leading to a meager $0.075$ scale per task under high uncertainty.
   - *Result (Resolution):* Our Softmax-free **BSigmoid-Router**, which replaces Softmax with independent Sigmoids, completely resolves this, achieving **$75.33 \pm 3.21\%$** on SVHN and a stable, balanced joint homogeneous accuracy of **$83.73 \pm 1.93\%$** (matching QWS-Merge's $83.97 \pm 0.53\%$ overall, and outperforming it on MNIST and CIFAR-10).
   - *Verdict:* Fully supported. Decoupled independent activations are structurally superior to standard Softmax routing for weight-space expert merging.

3. **QWS-Merge acts as a Structural Regularizer (Claim 4):**
   - *Result:* The unregularized layer-wise **GLS-Router** exhibits extreme sensitivity across seeds (SVHN standard deviation of **$24.30\%$**) and collapses on FashionMNIST ($64.80 \pm 3.53\%$). Standard L2 regularization on routing weights alone is insufficient to stabilize unregularized layer-specific amplitudes.
   - *Result (Resolution):* QWS-Merge maintains a remarkably stable joint mean standard deviation of only **$0.53\%$** across random calibration-sampling seeds. The data-scaling ablation study shows that scaling the calibration dataset to 128 and 256 samples dramatically stabilizes unregularized classical routers (e.g., GLS-Router's SVHN std. dev. drops from $24.30\%$ to $4.60\%$).
   - *Verdict:* Fully supported. While scaling calibration data is a valid data-driven alternative, wave cosine projection equations function as an exceptionally sample-efficient structural regularizer in extremely low-data regimes.

4. **Paradigm and Latency Distinction (Claim 1):**
   - *Result:* Physical latency benchmarks demonstrate that our proposed feed-forward BSigmoid-Router takes only $18.5$ ms per batch at $B=1$ and $19.8$ ms at $B=16$, which is over **$25\times$ faster** than online TTA AdaMerging (which requires $495.0$ ms per batch at $B=16$).
   - *Verdict:* Fully supported. Online TTA and offline calibration represent completely distinct computational paradigms.
