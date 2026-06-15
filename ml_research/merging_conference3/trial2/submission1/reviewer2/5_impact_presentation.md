# Presentation and Impact Evaluation

## Major Strengths
1. **Critical Vulnerability Exposure:** Exposing the **Overfitting-Optimizer Paradox** and **Sacrificial Task Bias** in state-of-the-art adaptive merging (AdaMerging) is a highly valuable, high-signal contribution. These failure modes are critical for anyone attempting to deploy adaptive model merging in production settings.
2. **Highly Practical & Training-Free:** The proposed **CalMerge** (SNEW + CCN) and **ESR** require zero access to training data, are entirely unsupervised (utilizing prediction entropy on unlabeled streams), and are mathematically simple. They introduce almost zero computational overhead during optimization, making them highly attractive for resource-constrained or real-time deployment.
3. **Rigorous Empirical Standards:** The paper demonstrates exceptional empirical rigor:
  - Average over 3 independent seeds.
  - Evaluation of 7 diverse baseline and diagnostic configurations.
  - Dense 2D hyperparameter grid sweeps ($\beta \times \gamma$) demonstrating a highly smooth, stable, and predictable generalization landscape.
  - Dedicated simulation of heterogeneous label spaces with unequal class counts.
4. **Honest & Transparent Scientific Communication:** The authors explicitly address and deconstruct limitations, such as the complete path determinism of gradient descent across seeds, the representational conflicts of ESR, and the homogeneous class limits of standard visual benchmarks.

## Key Areas for Improvement (Practitioner's Critiques)
1. **Toy Dataset Evaluation Limitation:** The primary limitation is the exclusive use of academic "toy" datasets (MNIST, FashionMNIST, CIFAR-10, SVHN). To prove genuine real-world utility and encourage industrial adoption, the authors must scale evaluations to complex, high-resolution, diverse datasets such as **ImageNet-1K, VTAB, Flowers102, or Stanford Cars**.
2. **Missing Runtime and Latency Benchmarks:** Test-time adaptation introduces inference latency because optimization (gradient descent or evolutionary mutation) occurs on the fly. The paper completely lacks runtime execution measurements (e.g., seconds per adaptation block, FLOPs, or memory overhead). This is a vital metric for real-world production systems.
3. **Complexity-Performance Trade-off Analysis:** The calibrated single-scalar baseline **Cal-Mean** (Method 9) is vastly simpler, optimizing only 4 variables, and achieves 61.13% Joint Mean. Our proposed flagship **CalMerge** (Method 8) optimizes 52 variables and achieves 61.82%. The paper does not analyze whether a marginal **0.69%** gain in accuracy is worth the significantly increased architectural and optimization complexity of layer-wise coefficients in practical settings.
4. **Unexplained Optimization Anomaly:** Table 1 shows that spatially shuffling the optimized coefficients for the 1+1 ES optimizer (*Shuffled 1+1 ES*) actually *improves* joint accuracy (from 59.77% to 60.45%). The authors do not address or explain this anomaly, which suggests optimization instability under derivative-free settings.

## Overall Presentation Quality
- **Rating: Excellent.**
- **Justification:** The paper is beautifully written, highly structured, and easy to follow. The mathematical equations are precise and elegant. The tables are clean and informative. Crucially, the "Methodological Discussion and Empirical Limitations" section (Section 3.5) shows an outstanding level of intellectual honesty and depth, addressing determinism, class-capacity scaling, and hierarchical representational conflicts.

## Potential Impact and Significance
- **Significance: High.** If the proposed calibration-aware framework is validated on larger-scale models (LLMs, large vision-transformers) and complex datasets, it has the potential to become a standard tool in model merging pipelines (e.g., being integrated directly into libraries like `mergekit`). 
- **Broader Impact:** The work serves as a foundational warning to the research and applied communities against unconstrained test-time layer optimization on small calibration batches, shifting the focus of test-time adaptation toward structurally constrained and scale-calibrated weight-space dynamics.
