# Experimental Setup and Empirical Evaluation Check

This evaluation critically analyzes the paper's experimental setup, datasets, baseline selections, and whether the presented results support the core claims made by the authors.

---

## Experimental Setup and Datasets
The experimental setup is designed with high scientific rigor and complete control:
- **Architecture:** The authors utilize `timm vit_tiny_patch16_224` (5.7M parameters), partitioned into $L=14$ discrete layer groups. This provides a clean, manageable continuous search space of 56 parameters ($\Lambda \in [0, 1]^{4 \times 14}$).
- **Datasets:** The multi-task classification benchmark comprises four diverse, well-established datasets:
  - **MNIST** (handwritten digits)
  - **FashionMNIST** (clothing items)
  - **CIFAR-10** (natural objects)
  - **SVHN** (street view house numbers)
- **Upper Bound:** Individual unmerged experts are evaluated under full-precision FP16, establishing a strong average upper bound of **$90.88\%$** (Table 1).

---

## Evaluation of Baselines
The authors select highly informative baselines to contextualize their results:
1. **FP16 Task Arithmetic (Baseline):** Scores **$35.12\%$** on the average multi-task accuracy. This low baseline accuracy indicates a massive weight-space representation conflict among experts fine-tuned on the four datasets. This represents an **"extreme weight conflict"** scenario, which is highly appropriate for stress-testing merging algorithms under non-cooperative, fractured landscapes.
2. **Naive Merge-then-Quantize (M-then-Q):** Evaluated under INT4, scoring **$21.50\%$**. This establishes the baseline for post-hoc quantization without test-time adaptation.
3. **Quantized AdaMerging:** A crucial baseline that performs unquantized continuous gradient-based optimization in FP16, and then applies post-hoc target quantization. It scores **$30.00\%$**, which isolates and proves that direct quantization-aware optimization is not only unnecessary but actually inferior to unquantized search followed by post-hoc quantization.
4. **Supervised Calibration Baseline:** Optimizes merging coefficients directly via supervised cross-entropy on the $N=16$ sample calibration stream. It scores **$35.00\%$**, isolating the structural fragility of unsupervised prediction entropy minimization.

---

## Alignment of Empirical Results with Claims

The empirical results presented in the tables fully and robustly support the paper's central claims:

### Claim 1: Catastrophic Cross-Operator Overfitting (Axis 2)
- **Evidence:** Table 3 (Cross-Schema Generalization Matrix) show that coefficients optimized under symmetric per-channel (\texttt{sym\_channel}, 17.88%) collapse to **$10.13\%$** accuracy (random guess) when evaluated under symmetric per-tensor (\texttt{sym\_tensor}). Coefficients optimized under asymmetric per-channel (\texttt{asym\_channel}, 33.00%) drop to **$12.63\%$** under \texttt{sym\_tensor}—a catastrophic **$-20.37\%$ absolute drop**.
- **Support for Claim:** Excellent. This provides unequivocal proof of quantization-operator overfitting.

### Claim 2: Superiority of Full-Precision Search (Axis 1)
- **Evidence:** Table 2 shows that Quantized AdaMerging achieves **$30.00\%$** average accuracy at $N=16$, whereas direct STE optimization (Q-Merge) peaks at **$26.25\%$** at the same $N=16$, and plateaues at **$26.00\%$** even with $N=64$.
- **Support for Claim:** Excellent. This demonstrates that direct low-bit optimization via STE introduces significant gradient noise that damages weight-space search.

### Claim 3: Stochastic Search vs. Biased Gradients (Axis 3)
- **Evidence:** Table 4 shows that derivative-free 1+1 ES achieves **$20.75\%$** on the source schema (beating unregularized STE's $17.88\%$). However, on the mismatched target schema, it collapses to **$8.62\%$** (worse than STE's $10.12\%$), yielding a massive generalization gap of **$-12.13\%$**.
- **Support for Claim:** Excellent. It proves that while black-box searchers navigate the discontinuous quantized landscape better, they overfit to simulated rounding thresholds even more severely, exposing a trade-off between search capacity and boundary overfitting.

### Claim 4: Label Skew Vulnerability (Axis 4)
- **Evidence:** Table 5 shows average accuracy collapses from **$26.25\%$** (clean stream) to **$15.50\%$** under highly Gini-skewed streams (where a dominant class receives 80% of samples).
- **Support for Claim:** Excellent. It highlights the vulnerability of unsupervised entropy minimization under realistic test-time class distribution shifts.

---

## Critical Gaps, Limitations, and Strengths

### Strengths:
- **Ablation Studies:** The paper includes crucial ablations (varying learning rate sweeps, dynamic initialization using optimal FP16 coefficients, TV regularization scaling sweeps) that rule out simple hyperparameter-tuning issues.
- **Supervised Validation:** The supervised cross-entropy validation (Table 6) is a highly rigorous way to prove that prediction entropy minimization is structurally fragile rather than the dataset being data-starved.
- **PoC Generalizability Audits:** The CNN PoC (ResNet-18) and low-rank SVD subspace models (Table 7) provide empirical evidence of architectural and subspace behaviors, with the SVD model critically identified as a "Low-Capacity Generalization Illusion."

### Limitations (Fully Acknowledged by Authors):
- **Model Scale:** The evaluations are limited to `ViT-Tiny` (5.7M parameters). However, the authors defend this choice by explaining how the dimensionality curse in larger models is expected to *expand* rather than shrink the Cross-Schema Generalization Gap due to the exponential increase of independent discrete rounding thresholds.
- **Joint Weight-Activation Quantization (W4A8/W4A4):** Real-world edge NPUs often require dynamic activation quantization, which introduces outlier-aware noise. The authors acknowledge this and propose integrating activation smoothers (like SmoothQuant) into future merging pipelines.

---

## Conclusion of Experiment Check
The empirical results are exceptionally solid, rigorous, and directly support all core claims. The authors have constructed a highly controlled experimental framework that systematically isolates each variable, providing undeniable evidence of the unstudied vulnerabilities in quantization-aware model merging.
