# 3. Soundness and Methodology

## Clarity of the Description
- **High Mathematical Clarity:** The mathematical formulation of the dynamic merging process, the Bounded Sigmoidal Router (BSigmoid-Router), and the Task-Correlation Prior Regularization (TCPR) variants are presented clearly and rigorously in Section 3. Equations 1 through 13 provide complete mathematical transparency.
- **Narrative Inconsistency / Contradiction:** There is a severe narrative disconnect in the paper. The title, abstract, introduction, and contributions section frame TCPR as a successful, highly effective regularization method that "consistently prevents high-conflict task collapse." However, the actual experimental results in Section 4.4, Section 4.5, and the conclusion explicitly state that TCPR *fails* to improve upon the unregularized BSigmoid-Router and that it actively degrades performance at active scales ($\beta \ge 1.0$). This internal contradiction severely harms the clarity of the paper's thesis and requires complete reframing.

## Appropriateness of Methods
- **BSigmoid-Router:** Extremely appropriate. Replacing the standard competitive Softmax with decoupled, independent sigmoid functions is a highly elegant, simple, and direct way to resolve the zero-sum competitive constraint of multi-task routing.
- **Heterogeneous Benchmark:** Combining MNIST, FashionMNIST, CIFAR-10, and SVHN on a Vision Transformer (`vit_tiny_patch16_224`) is an excellent, challenging setup for testing dynamic model merging.
- **Low-Data Calibration Regime:** Setting calibration data to 16 samples per task (64 total) is highly appropriate and realistically simulates edge-AI and low-resource constraints.
- **Expert Training Sub-Optimality:** The authors train experts to a sub-optimal regime (e.g., MNIST 73.20%, SVHN 23.20%). While they argue this simulates representational noise and computationally constrained edge-AI environments, evaluating *only* sub-optimal experts is a major limitation. It is highly possible that the "Alignment-Interference Paradox" and the failure of TCPR are artifacts of this extreme parameter noise in under-trained models, rather than an intrinsic flaw of prior regularization in well-trained regimes.

## Potential Technical Flaws and Limitations
- **Single-Seed Evaluation:** The main results in Table 1 are reported for a single seed (`seed=42`). Given the extreme volatility of low-data calibration (64 samples) and noisy under-trained experts, single-seed results are highly prone to statistical noise. Without evaluating multiple seeds and providing standard deviations, a difference of 0.30% (e.g., 25.50% for BSigmoid-Router vs 25.20% for TCPR) or even larger differences could easily be within the margin of error.
- **Unexplored Hyperparameters:** 
  - The scale ceiling $\lambda_{\text{max}} = 0.3$ is fixed. No ablation study is provided to show how the BSigmoid-Router behaves under other ceiling values.
  - The L2 weight decay $\gamma = 10^{-4}$ is fixed.
- **Generality of the Deconstruction:** Since the study is limited to a single model architecture (`vit_tiny_patch16_224`) and a single set of sub-optimal experts, it remains unproven whether the deconstruction of static prior regularization holds for larger-scale architectures (e.g., LLMs, ViT-Base) or fully converged, high-performance experts.

## Reproducibility
- The paper is highly reproducible from a mathematical perspective. The equations are self-contained, and the exact hyperparameter configurations (learning rates, epochs, seed, sample counts) are clearly specified in Section 4.1.
