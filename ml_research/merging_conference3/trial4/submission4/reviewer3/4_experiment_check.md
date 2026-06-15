# 4. Experimental Evaluation and Empirical Check

As an **empiricist**, this document performs a rigorous, data-first evaluation of the experimental setup, baselines, datasets, statistical soundness, and whether the empirical findings actually support the paper's central claims.

## Evaluation of the Experimental Setup

### 1. Strengths
- **Exhaustive Simulation Rigor**: The continuous multi-task weight-merging simulation landscape (Model II, Vision Transformer ViT-B/32) is exceptionally well-studied. Evaluating across **30 independent random seeds** (seeds 42 to 71 inclusive) and reporting standard deviations for all tasks and averages (Tables 1 and 2, and Table 3) represents an outstanding level of statistical rigor. This is the gold standard for empirical validation in ML.
- **Multiverse Stress-Testing**: The paper does not restrict itself to standard settings. It subjects the model-merging algorithms to multiple highly adversarial and non-stationary stream environments:
  - **Extreme Label Shift**
  - **Bursty Task Streams**
  - **Small Batch Size Noise (Batch size = 4)**
  - **Isotropic and Structured Validation Target Selection Bias (swept from 0.0 to 0.3)**
  This multi-axial stress-testing provides strong empirical evidence of the generalization and robustness of the spectral representation.
- **Honesty in Reporting Failures**: The paper is exceptionally honest and transparent about the failure of SpectralMerge-LP ($F=3$) and LP-Adaptive on the physical ResNet-18 checkpoints (both collapsing to the majority-class baseline of 29.00%). Rather than hiding this result, the authors use digital signal processing principles to explain the **PEFT-Induced Step-Function Discontinuity**, which mathematically necessitates the soft regularization of SpectralMerge-Reg. This level of empirical honesty and deep diagnostic reasoning is highly commendable.

### 2. Weaknesses and Areas for Improvement

#### A. Lack of Statistical Rigor on Physical Neural Networks
- **The Critique**: While the simulated benchmark uses 30 random seeds with standard deviations, the actual PyTorch physical network experiments (Heterogeneous MLP and Pre-trained ResNet-18) are reported as **single-run point estimates** without any standard deviations, confidence intervals, or mention of multiple seeds.
- **Why this matters**: At extremely low sample complexities (e.g., $M=10$ or $M=15$ samples on CIFAR-10), the validation set is highly susceptible to sampling variance. The performance of the optimized coefficients can vary wildly depending on *which* specific few-shot validation samples are selected. To make a convincing empirical claim on physical models, the authors **must** average their physical experiments over multiple seeds (e.g., 5 or 10 runs) and report standard deviations.

#### B. Simplistic and Small-Scale Datasets
- **The Critique**: The physical experiments are conducted on very simple, small-scale datasets:
  - The PyTorch MLP uses a synthetic classification dataset with 3 conflicting tasks.
  - The ResNet-18 experiment uses a highly stripped-down version of CIFAR-10, formulated as two binary classification tasks (Vehicles and Animals) evaluated within a shared 10-class output space. Furthermore, the task-specific training sets are tiny (120 samples per task), and the task-specific experts are weak (Expert 1 achieves only 65.00% accuracy).
- **Why this matters**: While CIFAR-10 binary tasks are useful for proof-of-concept, they do not represent modern model-merging applications. Parameterized merging is typically deployed on large-scale Vision Transformers (e.g., ViT-L, ViT-H on VTAB benchmarks) or Large Language Models (e.g., Llama-3-8B/70B, Mistral on GLUE/MMLU benchmarks). Evaluating on these larger models and datasets would provide a far more convincing demonstration of SpectralMerge's practical utility.

#### C. Modest Absolute Performance on CIFAR-10
- **The Critique**: The individual experts achieve 86.00% and 65.00% accuracy on their respective tasks. When merged, the best-performing model (SpectralMerge-Reg) achieves **54.00%** accuracy.
- **Why this matters**: Although 54.00% is a major improvement over the spatial unconstrained/polynomial models (29.00% - majority class collapse) and the Uniform baseline (41.00%), it still represents a massive drop from the individual expert performance (especially Task 0's 86.00%). This indicates that while spectral regularization successfully prevents validation overfitting, it does not fully resolve the underlying representation clashes or inter-task interference. The paper should discuss this limitation more explicitly.

---

## Analysis of Baselines
The paper includes a highly comprehensive and competitive set of baselines:
1. **Uniform (Task Arithmetic)**: The standard baseline for weight interpolation.
2. **Online Adaptive Baselines (AdaMerging, RegCalMerge)**: Represents the state-of-the-art in online test-time optimization.
3. **PolyMerge (d=2) / Poly-Val ($d=2$)**: Represents the state-of-the-art in continuous spatial trajectory smoothing.
4. **Global Task-Wise (DC-Only)**: An exceptionally strong and well-thought-out baseline. By optimizing a single global scaling factor per task across all layers (equivalent to $c_{k,0}$ only), it acts as a highly constrained control group.

### Critiques on Baselines:
- **Baseline Tuning**: The paper states that PolyMerge and spatial models are optimized under the same conditions. However, were the learning rates and optimization steps for these baselines fully tuned? For instance, because the Vandermonde matrix in PolyMerge is highly ill-conditioned, standard learning rates used for Adam might be suboptimal. Did the authors sweep the learning rates for PolyMerge to ensure a fair comparison?
- **Omission of Sparsification Baselines**: Heuristic sparsification baselines like TIES-Merging or DARE are discussed but not included in the quantitative comparisons. While SpectralMerge is conceptually orthogonal to them, including a baseline that combines TIES/DARE with SpectralMerge would have empirically verified the claimed "synergy" between these methods.

---

## Do the Results Support the Claims?

1. **Claim**: *SpectralMerge-LP and SpectralMerge-Reg achieve state-of-the-art simulated accuracies.*
   - **Support**: **Fully Supported**. Table 1 shows that SpectralMerge-LP ($F=3$, 86.46%) and SpectralMerge-Reg (86.44%) outperform Poly-Val (85.67%) and unconstrained spatial search (83.81%).
2. **Claim**: *Frequency-domain parameterization completely resolves the Overfitting-Optimizer Paradox.*
   - **Support**: **Supported, but qualified**. At $M=5$, unconstrained spatial search collapses to 82.77% (worse than the Uniform baseline), while SpectralMerge-Reg maintains 86.20%. This supports the claim. However, on the physical ResNet-18 checkpoints, SpectralMerge-LP ($F=3$) and LP-Adaptive collapsed to 29.00% due to the PEFT-induced step-function discontinuity. This indicates that *only* the soft-regularized variant (SpectralMerge-Reg) successfully resolves the paradox under localized adaptation.
3. **Claim**: *SpectralMerge is remarkably resilient to validation selection bias and adversarial stream noise.*
   - **Support**: **Fully Supported**. Table 2 and Figure 6 clearly demonstrate that SpectralMerge-LP and Reg maintain high accuracies ($>85\%$) and degrade gracefully compared to unconstrained spatial search.
4. **Claim**: *DCT-II offers a massive optimization advantage over PolyMerge due to perfect conditioning ($\kappa = 1.0$).*
   - **Support**: **Fully Supported**. Figure 4 demonstrates the perfect condition number of the DCT basis compared to the exponential growth of the Vandermonde matrix. Figure 8 demonstrates that SpectralMerge exhibits exceptionally rapid and stable convergence compared to PolyMerge, which stalls or fluctuates.
