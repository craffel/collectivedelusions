# Evaluation Component 4: Experimental Check

An empirical evaluation of the paper's experimental design, datasets, baselines, and results reveals several critical strengths and major limitations.

## 1. Experimental Setup and Statistical Soundness
* **Strengths:** 
  * The paper is statistically rigorous within its chosen simulation framework, reporting the mean and standard deviation across **5 independent random seeds** for all main results (Table 1).
  * The empirical sweeps are highly systematic, covering three key dimensions: gating threshold sensitivity ($\gamma_{\text{conf}} \in [0.0, 1.0]$ in Figure 1), calibration data scarcity ($N \in \{16, 32, 64, 128, 256, 512\}$ in Figure 2), and batch heterogeneity scale ($B \in \{1, 8, 32, 128, 512\}$ in Figure 3).
  * The authors conduct detailed systems-level latency benchmarking (Table 3), evaluate Fusion Weight Caching across discretization steps (Table 4), and simulate parallel GPU-vectorized latency and warp batch padding (Tables 4 & 5).
* **Weaknesses:**
  * **Purely Synthetic Simulation:** The entire quantitative evaluation (including all figures and main results tables) is conducted within a 1-layer synthetic coordinate model (the *Isolating Coordinate Sandbox*). There is no validation on actual real-world neural network backbones (such as ResNets, Vision Transformers, or BERT/LLaMA architectures).
  * **Pre-compiled Latency Artifacts:** The systems benchmarks in Table 3 are run on a sequential CPU-bound Python-loop simulator. While the authors transparently acknowledge that the near-zero overhead of MBH at large batch sizes is an artifact of this CPU-bound simulator and provide a simulated GPU projection (Table 4), this still lacks physical verification on actual parallel GPU hardware.

---

## 2. Evaluation of Datasets
* **Critical Critique on "Datasets":** 
  * The paper claims to evaluate performance across four distinct datasets: MNIST, Fashion-MNIST, CIFAR-10, and SVHN. However, **the authors do not actually run experiments on these datasets.**
  * Instead, they simulate the "expert performance ceilings" of these datasets by corrupting disjoint coordinates of a 192-dimensional feature space with varying levels of Gaussian noise ($\sigma_0, \sigma_1, \sigma_2, \sigma_3$).
  * Therefore, the "datasets" evaluated are entirely synthetic, and the input representations are generated via i.i.d. normal distributions. This completely bypasses the complex, high-dimensional, and non-linear manifold structures present in actual image or text datasets.

---

## 3. Appropriateness and Strength of Baselines
* **Baselines Evaluated:** The paper compares against static Uniform Merging, unregularized/regularized Linear Routers, Task-Variance Regularization (VR-Router), Task-Space Anchor Regularization (TSAR), and Parameter-Free Subspace Routing (PFSR).
* **Weaknesses in Baseline Comparison:**
  * **Simplistic Setting for Advanced Merging:** In Section 4.1, the authors state that state-of-the-art static model-merging techniques like Task Arithmetic, TIES-Merging, and DARE mathematically reduce to Uniform Merging because the experts reside in disjoint block coordinates (meaning there are no parameter conflicts across experts).
  * By evaluating only in this coordinate-isolated sandbox, the authors bypass the very problem that advanced merging methods (like TIES and DARE) are designed to solve (i.e., resolving interference and sign conflicts in overlapping, highly coupled weight spaces). Thus, the comparison with static model-merging is oversimplified and does not represent real-world ensembling challenges.
  * **Lack of SOTA Serving Engine Benchmarks:** While the authors provide a qualitative comparison with S-LoRA and Punica (Table 6), they do not run actual empirical comparisons against these frameworks, leaving the exact performance and latency delta unquantified on real hardware.

---

## 4. Whether Results Support the Claims
* **Within the Sandbox (Supported):** Within the highly controlled mathematical environment of the sandbox, the results strongly support the claims. CGHR successfully leverages the PFSR fallback to prevent transductive overfitting under extreme data scarcity ($N=16$), and MBH successfully groups mixed-task samples to completely shield the system from heterogeneity collapse across all batch sizes.
* **Generalization to Real-World Settings (Unproven):** Because the sandbox is highly idealized (disjoint, orthogonal block-coordinate representation space), it is highly uncertain whether the results would generalize to real-world models. In real models, representations overlap significantly, which violates the core assumption of the UNC-PFSR Equivalence. While the SVD Subspace Projection (Table 7) shows promising results in a proof-of-concept simulation with overlapping subspaces, this is also a synthetic simulation with random orthonormal bases, leaving the real-world scalability and performance of the proposed methods unproven.
