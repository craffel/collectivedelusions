# Evaluation Component 1: Paper Summary

## Main Topic and Objective
This paper critically deconstructs the recently popularized paradigm of online test-time adaptation (TTA) in weight-space model merging. It exposes two core methodological vulnerabilities in the current literature: the "no-data" strawman (where complex, active online methods are compared only against a naive, unoptimized uniform baseline, ignoring the practical availability of small validation sets) and the severe fragility of unsupervised TTA objectives (such as entropy minimization) under realistic, non-i.i.d. target deployment streams. The authors propose **Offline Few-Shot Validation Tuning (OFS-Tune)** as a simple, static, and robust alternative that leverages a tiny labeled validation set to find optimal merging parameters offline, requiring zero test-time compute.

## Proposed Approach
OFS-Tune optimizes weight-merging coefficients offline using a few-shot validation set ($M \in [5, 50]$ samples per task). To prevent overfitting under such data scarcity, the authors parameterize the search space using low-dimensional trajectories:
1. **Global Task-Wise Coefficients (GT-Merge):** A constant coefficient per task across all layers ($K$ parameters).
2. **Polynomial Coefficient Profiles (Poly-Val-Merge):** Modeling layer-wise task weights as low-degree polynomial functions of depth, yielding $K(d+1)$ parameters.

Optimization is performed using either **Nelder-Mead simplex search** (for low $K$) or differentiable optimization with **PyTorch Adam** (scaling up to $K=64$). This results in a static, ready-to-deploy multi-task model with zero runtime adaptation overhead.

## Key Findings
1. **Deconstruction of Online TTA SOTA:** Under standard i.i.d. streams, OFS-Tune ($d=1, M=10$) achieves an average accuracy of **85.89%** in simulation, outperforming naive uniform merging (84.44%) and dominating online AdaMerging (79.72%) and RegCalMerge (80.70%) with zero test-time compute.
2. **Extreme Fragility of Online TTA:** Under realistic stream corruptions (extreme label shift, sequential task clustering, and small batch sizes), online methods catastrophically collapse (e.g., AdaMerging drops to 77.99% under label shift and 79.56% under bursty streams) due to transductive noise and representational drift. OFS-Tune remains completely robust and stable at 85.89%.
3. **The Overfitting-Optimizer Paradox:** When data is extremely scarce ($M=5$), unconstrained high-capacity optimization (such as 48-parameter layer-wise tuning or full-joint/head tuning) overfits catastrophically to validation noise when minimized with highly capable optimizers like Adam. In contrast, low-dimensional parameterizations (GT-Merge or Poly-Val) act as analytical low-pass filters that reject noise and achieve superior generalization.
4. **Physical Neural Network Validation:** Physical weight-merging experiments on a 5-layer PyTorch CNN (MNIST and FashionMNIST) confirm the Overfitting-Optimizer Paradox on real weights. High-capacity baselines (Few-Shot Head Tuning and Joint FT) catastrophically underperform the Uniform baseline (achieving 47.97% and 43.77% vs. Uniform's 55.27%) and collapse further under validation label noise, while OFS-Tune Poly-Val consistently generalizes (56.31% on clean val, 56.35% on noisy val).

## Explicitly Claimed Contributions (with Evidence)
1. **Exposing the "No-Data" Strawman and Fragility of TTA:** Demonstrated via a calibrated simulation landscape with 30 seeds, showing TTA methods collapse under Extreme Label Shift, Bursty Streams, and Small Batch Sizes (Section 4.4.2, Table 2).
2. **Introducing OFS-Tune:** Formulating low-dimensional search spaces (Section 3.2, Equations 3-5) and proposing the static offline baseline.
3. **The Overfitting-Optimizer Paradox:** Formulating the trade-off where unconstrained spaces overfit while low-dimensional spaces act as filters, supported by empirical controls using PyTorch Adam vs. Nelder-Mead (Section 4.4.3, Table 4).
4. **Task Scalability and Validation Domain Shift Analysis:** Sweeping $K$ tasks from 4 to 64 and validation bias up to 30%, showing how Poly-Val filters systematic bias and PyTorch Adam scales smoothly (Section 4.5 and Appendix).
5. **Real-world CNN Verification:** Running physical experiments on a 5-layer CNN, verifying that OFS-Tune Poly-Val outperforms unconstrained joint/head tuning and is robust to 30% label noise (Section 4.5, Table 5).
