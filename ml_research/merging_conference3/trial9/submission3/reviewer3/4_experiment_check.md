# Experimental Evaluation: Contraction-Regularized Router (CR-Router)

## 1. Quality and Completeness of Experimental Setup
The experimental evaluation is **exceptionally thorough, highly realistic, and methodologically sound**. Rather than relying solely on a simplified synthetic coordinate sandbox, the authors evaluate across three increasingly challenging settings:
1. **Experiment 1 (Orthogonal Task Subspaces)**: A controlled baseline representing perfectly decoupled task manifolds.
2. **Experiment 2 (Overlapping Task Subspaces)**: Introduces 48 dimensions of coordinate overlap between adjacent tasks, inducing severe representational cross-talk and task interference.
3. **Experiment 3 (Real-World Vision Embedding Manifolds)**: Uses ResNet18-extracted 512-dimensional embeddings of actual datasets (MNIST, Fashion-MNIST, KMNIST, USPS) projected to 192 dimensions via PCA and normalized to have a mean norm of 1.0 ($R_h = 1.0$).

All experiments are executed and averaged over **10 independent random seeds (42 to 51)**, reporting standard deviations, which ensures statistical significance.

## 2. Baselines and Competitor Selection
The paper compares CR-Router against a very comprehensive suite of competitive serving configurations:
- **Ceiling / Non-Parametric Ceilings**: *Expert Oracle Ceiling*, *SABLE* (state-of-the-art non-parametric nearest-centroid), and *ChemMerge* (kinetic routing with continuous smoothing).
- **Static Baseline**: *Uniform Merging* (fixed uniform weights $\alpha_k = 1/K$).
- **Parametric / Learned Baselines**:
  - *Linear Router (Unregularized)*: Calibrated via gradient descent on the calibration split without regularization.
  - *Shared Router*: Employs a single routing head shared across all 14 layers to evaluate Hierarchical Multi-Task networks where early layers mix and later layers specialize.
  - *L2-Fixed Router*: Standard parameter regularization (L2) with fixed routing temperature ($\tau_l = 0.05$).

The inclusion of **Shared Router** and **L2-Fixed Router** is a powerful baseline choice. It directly stress-tests the necessity of the proposed joint spectral-temperature contraction regularizer under depth-heterogeneous settings.

## 3. Methodological Rigor: Active Gating Evaluation Metrics
A common flaw in ensembling literature is evaluating "routing accuracy" using the final activation's cosine similarity to prototypes. In orthogonal spaces, static Uniform Merging can achieve near-oracle routing accuracy because soft coordinate alignment filters noise, creating an "oracle" illusion where no routing is actually performed.
- **The Solution**: The authors introduce two secondary online gating evaluation metrics:
  1. **Direct Gating Accuracy (%)**: Measures how often the router assigns its maximum coefficient to the true task expert.
  2. **Gating Cross-Entropy**: Measures the average cross-entropy loss of routing probability on the true active task.
- **Impact**: These metrics successfully expose that Uniform Merging has a random-guess 25.00% Direct Gating Accuracy, resolving the "oracle" illusion and demonstrating high scientific integrity.

## 4. Main Experimental Results and Empirical Validation
The empirical results strongly support the paper's claims and highlight the practical value of the joint regularizer:

### A. Synthetic Sandbox
- **Orthogonal Subspaces (Exp 1)**: The unregularized Linear Router overfits severely (34.73% classification accuracy). CR-Router achieves **53.35% $\pm$ 3.84%** accuracy, outperforming unregularized by **+18.62%** absolute.
- **Overlapping Subspaces (Exp 2)**: Uniform Merging collapses to **27.48% $\pm$ 2.88%** due to cross-talk, proving that static ensembling fails under overlap. CR-Router recovers a strong **43.48% $\pm$ 4.70%** classification accuracy, representing a massive **+16.00%** absolute improvement over Uniform Merging and **+12.86%** over the unregularized router.

### B. Real-World Embedding Manifolds (Exp 3)
- Under realistic manifold overlap, Uniform Merging collapses to **7.70% $\pm$ 0.87%**.
- CR-Router achieves an outstanding **53.70% $\pm$ 2.37%** classification accuracy and **84.22% $\pm$ 3.09%** representation routing accuracy.
- Under this challenging benchmark, CR-Router significantly outperforms the simpler **L2-Fixed Router** baseline by **+6.37% absolute classification accuracy** (**53.70% vs. 47.33%**). This victory proves that static, sharp temperature heuristics are highly unstable under realistic representation overlaps, whereas CR-Router’s joint contraction regularization successfully guides optimization to stable, convergent trajectories.

## 5. Practical Extensions and Efficiency Profiling

### A. Empirical Validation of Label-Free Tuning Heuristics
The authors provide a high-resolution grid sweep (Table 8) showing a smooth transition across regularization scale $\lambda$. It validates their three proposed online, label-free heuristics (Gating Depth-Variance, Shannon Gating Entropy, and Gating Lipschitz Bound), showing that peak joint performance sits in a balanced, stable entropy valley where depth-variance is minimized.

### B. Adaptive Test-Time Temperature Annealing (A Major Breakthrough)
Reducing the scale factor $\gamma_{\text{scale}}$ from 1.0 down to 0.10 boosts classification accuracy from 53.55% up to **62.45% $\pm$ 2.98%** (a massive **+8.90% absolute gain**). This demonstrates how sharpening gating decisions at test-time successfully filters out representation noise, successfully resolving the "expert dilution" trade-off while retaining the optimization stability of training-time contractive bounds.

### C. Profiling Serving Efficiency (Practical Relevance)
The CPU profiling benchmark (Table 10) shows that CR-Router is highly lightweight. At batch size 400, CR-Router processes **15,785.1 samples/sec** with a latency of **25.34 ms**, compared to **10,464.1 samples/sec** and **38.23 ms** for SABLE. 
The authors provide a strong GPU scaling and hardware acceleration analysis:
- **FLOP Reduction**: CR-Router is $C=10$ times computationally lighter in gating FLOPs than SABLE/ChemMerge because it performs simple matrix multiplication $\mathcal{O}(B \cdot K \cdot D)$ rather than pairwise Euclidean distance calculations $\mathcal{O}(B \cdot K \cdot C \cdot D)$ and reductions.
- **Tensor Core Synergy**: Dense matrix multiplications map directly onto specialized Tensor Cores (GEMM), while SABLE's non-linear distance computations and reductions are bound by memory latency and GPU CUDA core bandwidth, suffering from memory-bottlenecks as batch sizes scale up.
This is of extreme practical relevance to machine learning engineers deploying multi-task pipelines.
