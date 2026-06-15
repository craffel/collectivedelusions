# Detailed Paper Outline

## Title: The "No-Data" Strawman: Demystifying Test-Time Adaptation vs. Offline Few-Shot Validation Tuning

## Authors: Marcus Thorne (University of Oxford) - marcus.thorne@cs.ox.ac.uk

### Abstract
- Challenge the online test-time adaptation (TTA) paradigm in model merging.
- Point out that TTA claims SOTA by comparing against a naive uniform baseline, ignoring the practical availability of small validation sets (the "no-data" strawman).
- Propose Offline Few-Shot Validation Tuning (OFS-Tune) using tiny validation sets ($M \in [5, 50]$).
- Highlight results: OFS-Tune outperforms TTA under standard streams and is completely robust to extreme distribution shift, temporal task burstiness, and gradient noise, with zero test-time compute.

### 1. Introduction
- **Context:** Model merging (e.g., Task Arithmetic) combines specialized models without retraining.
- **The TTA Trend:** Recent works (AdaMerging, RegCalMerge, PolyMerge) perform online test-time adaptation on unlabeled test streams via entropy minimization to optimize merging coefficients.
- **The Critique (The Methodologist):**
  - **The "No-Data" Strawman:** Comparing highly complex online adaptation against an unoptimized uniform baseline is a false dichotomy. In practice, a tiny validation set ($M \le 10$) is almost always accessible.
  - **The Fragility of Online TTA:** Online adaptation assumes stable, i.i.d. streams. Realistic conditions (label shift, task clustering, small batch sizes) introduce catastrophic noise and transductive overfitting.
- **The Solution (OFS-Tune):** Optimize low-dimensional coefficient profiles (GT-Merge, Poly-Val) offline on a few-shot validation set.
- **Contributions:**
  - Expose the "no-data" strawman and the empirical fragility of online TTA.
  - Show that OFS-Tune matches or exceeds TTA performance under clean conditions.
  - Show that OFS-Tune is completely robust to adversarial streams.
  - Validate that low-dimensional search spaces prevent validation overfitting.

### 2. Related Work
- **Model Merging:** Overview of weight-space merging, task vectors, and uniform/heuristics-based merging.
- **Test-Time Adaptation (TTA):** Entropy minimization (Tent) and its application to model merging (AdaMerging, RegCalMerge).
- **Polynomial Parameterization:** Layer-wise coefficient modeling (PolyMerge).
- **Methodological Criticisms:** Rigorous comparison of our static offline baseline against the trend of overly complex online methods.

### 3. Methodology
- **Problem Formulation:**
  - Task vectors: $V_k = W_k - W_{base}$.
  - Merged weights: $W_{merged}^{(l)} = W_{base}^{(l)} + \sum_k \alpha_k(l) V_k^{(l)}$.
- **Search Space Parameterizations:**
  - **GT-Merge:** $\alpha_k(l) = \alpha_k$ (constant across layers, $K$ parameters).
  - **Poly-Val-Merge:** $\alpha_k(l) = \sum_{j=0}^d c_{kj} (l/L)^j$ (polynomial profile, $K(d+1)$ parameters).
  - **Layer-wise Search:** Unconstrained scalars per task per layer ($K \times L$ parameters).
- **Offline Black-Box Optimization:**
  - Nelder-Mead optimization on a validation set $D_{val}$ of $M$ samples per task.
  - Minimizing joint cross-entropy loss $\mathcal{L}_{val}(\theta)$.
- **Deployment:** Deployment of a static, zero-test-time-compute model, avoiding backpropagation and forward adaptation passes at runtime.

### 4. Experimental Setup & Robustness Stress-Testing
- **Backbone & Tasks:** 12-layer Vision Transformer (ViT-B/32) merged across 4 tasks (MNIST, FashionMNIST, CIFAR-10, SVHN).
- **Simulation Environment:** Fast, stable, and deterministic continuous weight-merging simulation calibrated on empirical ViT-B/32 statistics.
- **Online Baselines:** Online AdaMerging, Online RegCalMerge, Online PolyMerge ($d=2$).
- **Adversarial Stream Conditions:**
  - *Extreme Label Shift:* Dirichlet imbalance ($\alpha=0.1$) on class distribution.
  - *Bursty Task Streams:* Sequential arrival of task blocks (temporal non-i.i.d. shift).
  - *Small Batch Sizes:* Batch size of 1 or 2, modeled with high-variance gradient noise ($\sigma = 0.5$).
- **Multi-Seed Sweep:** Execution across 30 independent seeds (42 to 71) for rigorous confidence intervals.

### 5. Results & Empirical Analysis
- **Standard Stream Performance:** OFS-Tune ($d=1, M=10$) achieves $85.89\%$, beating Uniform ($84.44\%$) and online TTA AdaMerging ($79.72\%$) and RegCalMerge ($80.70\%$).
- **Robustness under Adversarial Streams:**
  - Online TTA methods collapse under label/temporal shift and batch noise (e.g., AdaMerging drops to $77.99\%$ under label shift).
  - OFS-Tune remains completely robust ($85.89\%$ across all shifts) due to its static, offline-optimized nature.
- **Sample Complexity & Overfitting:**
  - Analyze validation sample size $M \in \{5, 10, 20, 50\}$.
  - Demonstrate the **Overfitting-Optimizer Paradox**: layer-wise search space (48 parameters) overfits to validation noise when $M=5$, while low-dimensional GT-Merge or Poly-Val ($d=1$) act as low-pass filters to prevent overfitting, achieving superior performance.
- **Compute and Generalization Trade-offs:** Show that OFS-Tune is not only more robust but requires zero test-time compute, whereas online TTA requires substantial backpropagation and forward passes.

### 6. Conclusion
- Summary of findings: static offline tuning is a superior, safer, and more realistic paradigm than online adaptation.
- Call for more rigorous baseline evaluation in the model merging literature.
