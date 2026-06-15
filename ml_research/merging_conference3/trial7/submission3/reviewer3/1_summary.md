# 1. Summary of the Paper

## Main Topic and Motivation
The paper addresses the challenge of post-hoc dynamic model merging (or parameter blending) under extreme resource constraints—specifically, when only a tiny validation/calibration dataset is available (e.g., $N=64$ samples) and when inputs arrive in heterogeneous streaming batches at test time. The authors identify two critical failure modes in existing dynamic routing approaches:
1. **The Overfitting-Optimizer Paradox:** Parametric routers (e.g., linear or MLP routing networks trained via gradient descent) easily memorize representation-space noise when trained on tiny calibration splits, leading to near-perfect training scores but catastrophic test-time performance collapse.
2. **Heterogeneity Stream Collapse (Vectorization Collapse):** Under heterogeneous streaming batches, standard parallel batch forwarding forces representations to average out across tasks, flattening dynamic routing weights to static uniform averages and erasing localized expert advantages.

## Proposed Approach
To solve these challenges, the authors propose a training-free, non-parametric framework containing two main components:
* **Gaussian Process Dynamic Routing (GP-DR):** A non-parametric Bayesian framework that replaces parametric neural-network routing gates. Inputs are projected onto a task-specific coordinate subspace using maximum cosine similarity to pre-computed task prototypes. A Gaussian Process prior with a Radial Basis Function (RBF) or other positive-definite kernel is placed on the routing mapping. This allows the optimal parameter blending weights to be solved analytically as a closed-form posterior mean in a single forward pass, completely bypassing gradient-based backpropagation.
* **Micro-Batch Homogenization (MBH):** A stream-level batch buffer partition mechanism. It groups incoming heterogeneous inputs into task-homogeneous micro-batches based on their predicted routing preferences, forwarding them separately to prevent representation-averaging collapse in the network backbone.

## Key Findings
* On a synthetic $14$-layer, $192$-dimensional Isolating Coordinate Sandbox ($K=4$ tasks, $N=64$ calibration samples), GP-DR achieves a **$72.40\%$** Joint Mean accuracy with zero training loops, outperforming unregularized global parametric linear routers by **$+42.40\%$** absolute.
* Under mixed-task streams, standard dynamic routers collapse to uniform merging performance ($\sim 25\% - 31\%$). Pairing GP-DR with MBH recovers performance to **$70.20\%$** accuracy (a recovery margin of **$+42.80\%$**).
* GP-DR provides an uncalibrated but exact measure of epistemic uncertainty via its closed-form posterior predictive variance $\sigma^2(\psi_*)$. This variance successfully flags out-of-distribution (OOD) orthogonal inputs (achieving $100.00\%$ AUROC and $0.00\%$ FRR).
* However, GPR posterior variance is mathematically and empirically shown to suffer from a critical "unit-sphere variance collapse" under realistic noise (such as random unit-sphere noise), and simpler distance-based heuristics (like 5-Nearest Neighbor distance) outperform it by a massive margin under representational overlap.
* Real-world evaluations on text classification (GLUE benchmark using BERT-Tiny) and generative settings (using GPT2) validate the applicability and robustness of both GP-DR and MBH in practical pre-trained spaces.

## Explicitly Claimed Contributions and Evidence
1. **Characterization of Overfitting and Stream Collapse:** Exposing the Overfitting-Optimizer Paradox and Heterogeneity Stream Collapse. *Evidence:* Table 1 (comparison matrix), Section 1 (Introduction), and Section 4.2/4.4.
2. **Gaussian Process Dynamic Routing (GP-DR):** Bypassing parametric training loops via analytical, closed-form GPR posterior mean formulas. *Evidence:* Section 3.1-3.3, Table 4 ( scoreboard accuracy of $72.40\%$).
3. **Uncertainty-Guided OOD Rejection & Analysis of Limitations:** Exposing the geometric origin paradox, numerical safeguards (Cholesky solver), and the unit-sphere variance collapse. *Evidence:* Section 3.4, Section 4.5, Table 8.
4. **Micro-Batch Homogenization (MBH):** Eliminating vectorization collapse under mixed-task streams. *Evidence:* Section 3.6, Section 4.4, Table 6.
5. **Rigorous Empirical sweeps and Real-world Validation:** Comparative evaluations on coordinate sandboxes, GLUE benchmark (BERT-Tiny), and GPT-2 settings. *Evidence:* Section 4 (Empirical Evaluation), Tables 4, 5, 6, 7, 8, 9, 10.
