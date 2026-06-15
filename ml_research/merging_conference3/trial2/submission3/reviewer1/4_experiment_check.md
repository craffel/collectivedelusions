# 4_experiment_check.md

## Critical Evaluation of the Experimental Setup
The paper's experimental setup is exceptionally thorough, systematic, and well-designed:
1. **Calibrated Continuous Simulator:** Designing a simulator calibrated directly on Vision Transformer (ViT-B/32) empirical statistics from prior literature is a brilliant and cost-effective way to isolate and study the optimization dynamics under transductive noise.
2. **Exceptional Statistical Rigor:** Running the simulated benchmarks over **30 independent random seeds** and reporting means and standard deviations is a major strength. Most papers in this field only report single-run results or averages over 3 seeds. This statistical robustness is highly commendable.
3. **Statistical Significance Testing:** The authors perform formal paired t-tests over all 120 evaluations, providing strong t-statistics and microscopic p-values ($9.53 \times 10^{-13}$ vs. unconstrained Adam, $1.04 \times 10^{-13}$ vs. early-stopped Adam), which mathematically confirms the superiority of PolyMerge.
4. **Latency Measurement:** Measuring the wall-clock step latency in PyTorch (Table 4) is an excellent way to prove that continuous subspace projections introduce absolutely zero computational overhead or graph-backpropagation latency in PyTorch.

## Datasets
The benchmarks used are standard and diverse:
- **Simulated benchmarks:** MNIST, FashionMNIST, CIFAR-10, SVHN. These represent a challenging multi-task scenario with varying optimal layer-importance profiles.
- **Physical Residual MLP:** Synthetic task datasets with 2-dimensional task indicators to mimic CLIP's text-conditioning mechanism.
- **Physical CLIP:** Real image datasets (CIFAR-10 and GTSRB), with official tokenized class prompts.

## Baselines
The paper compares PolyMerge against a highly comprehensive set of baselines:
- **Task Arithmetic (Uniform Baseline):** Fixed uniform merging coefficient.
- **Unconstrained AdaMerging (Layer-wise):** Independent layer-wise coefficients optimized to full convergence (500 steps) or early-stopped (10 steps).
- **Spatial Mean (Mean Treatment):** Post-hoc spatial averaging.
- **TV & L2 Regularization:** Penalty-term regularizers. TV is swept to find the optimal $\beta$ (e.g., $\beta=20.0$ in simulation, $\beta=50.0$ under non-convex landscape, and $\beta=5.0$ in CLIP).
- **PolyMerge (d=0):** Optimizing a single uniform scalar per task via TTA.

*Critique:* While the baseline selection in the simulation is flawless, the **physical validations lack competitive baselines**:
- In the PyTorch Residual MLP validation (Table 5), the authors do not compare PolyMerge against the Spatial Mean or L2 regularization baselines.
- In the CLIP validation (Table 6), the only baselines are Task Arithmetic, Unconstrained TTA, and TV-Regularized Adam. Since they implemented a physical setup, they could have compared PolyMerge against other concurrent adaptive merging or TTA methods like SyMerge, AdaMerging++, or L2 regularization. Omitting them weakens the comparative strength of the physical results.

## Whether the Results Support the Claims
The results mostly support the authors' central claims, but with several critical caveats that an empiricist must highlight:
1. **The Core Paradox is Validated:** The results in Table 1 (SVHN accuracy collapsing to $63.16\%$ under unconstrained Adam, accompanied by highly jagged coefficient profiles in Figure 1) and in the physical Residual MLP (roughness exploding to $0.0883$ in Table 5) provide definitive empirical proof of the Overfitting-Optimizer Paradox.
2. **Subspace Constraints Prevent Overfitting:** PolyMerge ($d=2$, Adam) completely stabilizes simulated TTA, raising SVHN accuracy to $75.19\% \pm 2.58\%$ and achieving a state-of-the-art multi-task average accuracy of $86.57\% \pm 7.48\%$ in Table 1. Under zero-order 1+1 ES, PolyMerge ($d=2$, ES) achieves $84.91\%$, significantly outperforming TV-regularized ES ($84.45\%$), proving its unique advantage for black-box adaptation.
3. **The MLP Validation Discrepancy (Table 5):** 
   While the authors claim that PolyMerge "successfully regularizes test-time weight-adaptation," Table 5 shows that **static Task Arithmetic (85.90% $\pm$ 3.28%) actually outperfroms all adapted models (including PolyMerge d=2 at 85.43% $\pm$ 2.18% and TV at 85.67% $\pm$ 2.25%)**. This suggests that test-time adaptation via entropy minimization actually hurts accuracy on this task, or that the expert models were already merged optimally. The authors do not address this discrepancy, which weakens their claim that adaptation is useful on this architecture.
4. **The Underfitting Bottleneck of Global PolyMerge (Table 6):**
   In the physical CLIP validation, Global PolyMerge ($d=2$) drops average accuracy from 94.00% to 89.00%, and PolyMerge ($d=4$) only recovers to 90.00%. This is significantly worse than the static baseline! This result directly refutes the general applicability of global polynomials to real functional weights, which exhibit highly heterogeneous layer-wise sensitivities. While the authors propose SplineMerge (Piecewise Constant) to resolve this and show it achieves 96.00% accuracy, this indicates that the core "PolyMerge" global polynomial framework is insufficient for physical foundation models, and that piecewise-continuous splines are necessary.
5. **No Confidence Intervals / Multiple Seeds for CLIP Validation:**
   Unlike the simulated sweeps (30 seeds) and MLP validation (10 seeds), Table 6 (CLIP validation) reports only single-run accuracies without any statistical error bars or multiple seeds. Since it uses a very small stream of 50 images per dataset, the CLIP results might have high statistical variance. Multiple seeds should be run to ensure the CLIP findings are statistically sound.
