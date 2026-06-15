# Paper Summary

## Main Topic and Approach
The paper presents a rigorous, independent **Multi-Axial Robustness Audit and Methodological Deconstruction of Q-Merge**, a recent state-of-the-art framework for quantization-aware model merging. 

Model merging consolidates multiple task-specific expert networks (fine-tuned from a shared pre-trained backbone) into a single consolidated model. However, post-training quantization (PTQ) of merged models often causes catastrophic performance degradation. Q-Merge addresses this by using Straight-Through Estimators (STE) to optimize layer-wise merging coefficients directly under simulated target quantization constraints using prediction entropy minimization on small calibration streams.

The authors approach these claims with methodological skepticism, identifying critical unstudied assumptions (Quantization-Operator Monomorphism, Calibration Stream Purity, and STE Gradient Path Fidelity). They systematically evaluate quantization-aware model merging on a standardized `timm ViT-Tiny` backbone across four tasks (MNIST, FashionMNIST, CIFAR-10, SVHN) along four distinct evaluation axes:
1. **Calibration Stream Size Sweep:** Exploring transductive overfitting to discretization noise across calibration sizes $N \in \{1, 4, 16, 64\}$.
2. **Cross-Schema Generalization Matrix:** Introducing a cross-schema evaluation where merging coefficients optimized under a source quantization schema $Q_{\text{opt}}$ are evaluated under five hardware-relevant target schemas $Q_{\text{eval}}$ (e.g., symmetric/asymmetric, per-tensor/per-channel, and double quantization).
3. **Spatial Regularization & Derivative-Free Search:** Testing if spatial smoothing (Total Variation regularizer) or derivative-free black-box search (1+1 Evolution Strategy) can mitigate overfitting to Simulated rounding thresholds.
4. **Stream Distortion and Skew Robustness:** Stress-testing the adaptation under out-of-distribution (OOD) Gaussian corruptions and severe class imbalance (Gini skew).

In addition, the paper presents a **Supervised Calibration Baseline** to decouple data scarcity from unsupervised entropy collapse and executes **empirical extensions** covering CNN backbones (ResNet-18) and subspace-constrained SVD projections.

---

## Key Findings

- **Catastrophic Cross-Operator Overfitting (Quantization-Operator Overfitting):** Merging coefficients overfit intensely to the exact mathematical operator used during optimization. Moving from a channel-wise source to a tensor-wise target collapses accuracy back to random-guess levels (~10%), which poses a major deployment risk on heterogeneous hardware.
- **Superiority of Full-Precision Search (Quantized AdaMerging):** Unquantized optimization in FP16 followed by post-hoc quantization (Quantized AdaMerging) consistently and substantially outperforms Q-Merge's direct quantization-aware optimization via STE ($30.00\%$ vs $26.25\%$), indicating that direct low-bit optimization via STE introduces significant gradient noise that damages weight-space search.
- **Unconstrained STE Failure:** Unconstrained STE fails to consistently beat a naive M-then-Q baseline across varying sample sizes, and empirical sweeps show that this instability cannot be resolved by simple learning rate tuning.
- **Stochastic Search vs. Biased Gradients:** Derivative-free 1+1 ES outperforms gradient-based STE on the source schema ($20.75\%$ vs $17.88\%$) but overfits even more severely to the source operator's rounding boundaries, causing a massive cross-schema generalization gap ($-12.13\%$).
- **Label Skew Vulnerability:** Prediction entropy minimization is blind to class distributions. Under severe class skew, the optimized weights collapse, destroying decision boundaries for underrepresented classes.
- **Supervised Validation:** Optimizing merging coefficients using supervised cross-entropy on $N=16$ samples yields a massive boost ($35.00\%$ vs $26.25\%$ unsupervised), demonstrating that unsupervised entropy minimization is structurally fragile for quantized model merging.
- **Architectural & Subspace Mitigation:** Localized CNN kernels (ResNet-18) exhibit smaller cross-schema generalization gaps than ViT-Tiny. Restricting weight modifications to a low-rank subspace (SVD projection) closes the generalization gap, but results in a "Low-Capacity Generalization Illusion" due to severe representational capacity degradation.

---

## Explicitly Claimed Contributions and Evidence

1. **The Multi-Axial Robustness Audit Framework:** Evaluated systematically along 4 axes on `ViT-Tiny` with 4 experts. Supported by quantitative results in Tables 2, 3, 4, 5, and 6.
2. **Identification of Cross-Schema Generalization Gap:** Discovered that continuous coefficients overfit to the simulated quantization operator. Evidence is presented in Table 3 (Cross-Schema Generalization Matrix) where evaluating `sym_channel` on `sym_tensor` drops performance from $17.88\%$ to $10.13\%$.
3. **Deconstruction of the Necessity of Quantization-Aware Search:** Proved that Quantized AdaMerging (FP16 search + post-hoc quantization) outperforms Q-Merge's direct STE optimization ($30.00\%$ vs $26.25\%$). Evidence is in Table 2.
4. **Analysis of Unsupervised Objectives vs. Supervised Baseline:** Proved prediction entropy minimization is fragile and prone to class-skew collapse. Evidence is in Section 4.5 and Table 5, showing supervised cross-entropy achieves $35.00\%$ (standard) and $23.75\%$ (skewed) compared to unsupervised entropy's $26.25\%$ and $15.50\%$.
5. **Practical Methodological Recommendations:** Formulated four mandates for the community (mandatory cross-operator validation, calibration stream heterogeneity audits, re-evaluation of optimizer paradigms, and mitigating task interference prior to discretization) with concrete algorithmic proposals (e.g., confidence-thresholded pseudo-labeling, hybrid STE/ES optimization pipelines, TIES/DARE pre-smoothing).
