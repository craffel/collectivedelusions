# 1. Summary of the Submission

## Main Topic
The paper presents **SuiteMerge**, a critical methodological audit of modern adaptive model-merging paradigms. Specifically, it contrasts two main approaches to layer-wise merging coefficient tuning:
1. **Online Test-Time Adaptation (TTA):** Unsupervised adaptation at test-time on unlabeled target streams by minimizing Shannon prediction entropy (e.g., AdaMerging, PolyMerge).
2. **Offline Few-Shot Validation Tuning (OFS-Tune):** A highly regularized supervised offline approach that leverages a tiny labeled calibration set (e.g., $M=10$ samples per task) to optimize stable, low-degree polynomial layer-wise coefficient trajectories.

The authors challenge the prevailing evaluation protocol in the adaptive model-merging literature, exposing a severe **"Task Suite Bias"** where algorithms are typically validated on only a single, arbitrary combination of datasets (MNIST + FashionMNIST + CIFAR-10 + SVHN), which masks critical failure modes in online adaptation.

## Proposed Approach
- **SuiteMerge Evaluation Framework:** The standard pool of four visual classification tasks is systematically partitioned into five distinct evaluation suites designed along axes of domain distance and representational conflict:
  - **Suite A (Highly Homogeneous - Low Conflict):** MNIST + FashionMNIST
  - **Suite B (Highly Heterogeneous - High Conflict):** CIFAR-10 + SVHN
  - **Suite C (Cross-Domain Digits):** MNIST + SVHN
  - **Suite D (Cross-Domain Objects):** FashionMNIST + CIFAR-10
  - **Suite E (Full 4-Task Suite - Control):** MNIST + FashionMNIST + CIFAR-10 + SVHN
- **Model II Landscape Simulator:** Rather than synthetic mathematical functions, the authors design a coupled non-convex sensitivity landscape calibrated against empirical Vision Transformer (ViT-B/32) classification statistics. This includes modeling correlated stream noise ($\epsilon_{\text{stream}}$) and validation sampling noise ($\epsilon_{\text{val}}$).
- **Offline Few-Shot Validation Tuning (OFS-Tune):** OFS-Tune constrains layer-wise merging coefficients to a low-degree continuous polynomial function of depth (linear $d=1$ or quadratic $d=2$) and optimizes them offline using Nelder-Mead derivative-free local search to minimize validation cross-entropy on a stratified, tiny validation set ($M=10$ samples per task).
- **Physical Weight-Space Neural Network Validation:** To validate simulated findings, the authors train a 5-layer Convolutional Neural Network on CPU and evaluate merging performance under scratch-trained (independent initialization) and pre-trained (shared initialization) эксперт regimes.

## Key Findings
1. **Task Suite Bias Confirmed:** The relative ranking of model-merging methods is highly sensitive to the chosen task suite. On highly homogeneous tasks (Suite A), simple Uniform averaging is highly competitive.
2. **Online TTA Overfitting and Collapse:** Under unconstrained online TTA (AdaMerging), optimizing high-dimensional layer-wise coefficients (e.g., 48 parameters for a 12-layer ViT) overfits to local, transductive stream noise. In physical weight spaces under high-conflict regimes, this unsupervised entropy-minimization objective causes the parameters to collapse into degenerate states.
3. **The Privilege Trap of Online TTA:** At inference time on interleaved target streams, online TTA requires oracle task labels to route gradients through the active task head alone ("Privileged TTA") to avoid immediate collapse. If forced to optimize prediction entropy jointly across heads without labels ("Unsupervised TTA"), performance suffers significantly.
4. **OFS-Tune as an Analytical Noise Filter:** Restricting the search space to continuous polynomial trajectories offline filters out high-frequency noise and validation sampling bias. In physical networks, OFS-Tune outperforms online PolyMerge by up to $3.70\%$ and online AdaMerging by up to $4.20\%$, all while requiring **zero test-time compute, zero test-time backpropagation, and zero privileged routing labels**.

## Explicitly Claimed Contributions and Evidence
- **Contribution 1: Exposing Task Suite Bias.**
  - *Evidence:* Table 1 and Figure 2 show that while online TTA (AdaMerging/PolyMerge) appears superior in the full control Suite E, it exhibits severe stability and performance degradation on high-conflict suites (Suite B & Suite D).
- **Contribution 2: Mathematical Formulation and Empirical Demonstration of Transductive Overfitting.**
  - *Evidence:* By modeling stream noise as a correlated offset sampled once per session, the authors show that unconstrained AdaMerging lags behind polynomial-constrained PolyMerge by $5.93\%$ in Suite B and exhibits more than double the standard deviation. Figure 3 illustrates unconstrained parameters oscillating wildly and drifting far from ground truth.
- **Contribution 3: Establishing OFS-Tune as a Robust, Zero-Overhead Regularizer.**
  - *Evidence:* OFS-Tune ($d=2$) consistently matches or exceeds PolyMerge across all simulated suites (reaching $68.62\%$ in Suite B). In the physical weight-space validation (Table in Section 4.5), OFS-Tune avoids the degradation suffered by online TTA under the pre-trained cooperative regime, achieving $83.00\%$ accuracy (outperforming AdaMerging by $4.20\%$).
- **Contribution 4: Multi-seed, Multi-suite Comparative Benchmark.**
  - *Evidence:* Main simulation results are verified over 30 independent random seeds across five multi-task suites, demonstrating high statistical confidence.
