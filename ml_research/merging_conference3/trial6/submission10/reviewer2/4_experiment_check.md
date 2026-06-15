# 4. Experiment Check

## Evaluation of Experimental Setup
- **Model and Backbones:** The choice of a compact `vit_tiny_patch16_224` backbone is highly practical for running exhaustive hyperparameter sweeps and calibrating in low-resource settings.
- **Data Splitting:** Calibrating on 16 samples per task (64 total) and testing on 250 samples per task (1000 total) is a rigorous split that appropriately evaluates low-data generalization.
- **Expert Training Regime:** The experts are intentionally under-trained (achieving 73.20% on MNIST and 23.20% on SVHN, with a joint mean of 62.40%). The authors state this is to simulate edge-AI scenarios and representational noise. However, this creates a highly sub-optimal baseline environment. It is unclear whether the experimental findings (especially the failure of TCPR) generalize to standard regimes with fully-trained, high-performance experts.

## Quality of Baselines
The paper compares against a highly comprehensive and rigorous set of **seven baselines**, which is a major strength:
1. **Specialist Expert:** The empirical upper bound.
2. **Uniform Merge (Task Arithmetic):** The baseline static merging method.
3. **Linear Router (Classical, Unregularized):** Basic unconstrained linear routing.
4. **BL-Router (Softmax, Unregularized):** Standard softmax-based linear router.
5. **BL-Router (Reg):** Softmax-based linear router with standard L2 regularization.
6. **BSigmoid-Router (Unregularized):** The proposed softmax-free sigmoidal router.
7. **BSigmoid-Router (Reg):** Sigmoid-based router with standard L2 regularization.
8. **QWS-Merge (SOTA):** The state-of-the-art wave-interference method.

This represents an exceptionally thorough and strong baseline comparison, covering static, dynamic, regularized, and complex state-of-the-art physics-inspired paradigms.

## Do the Results Support the Claims?
- **Claim 1: "TCPR consistently prevents high-conflict task collapse, bridges the performance gap to specialist experts, and provides a robust, scale-invariant pathway." (Abstract/Intro)**
  - **Verdict: NOT Supported.** The experimental results in Table 1 and the hyperparameter sweep in Figure 1 directly refute this claim. Table 1 shows that TCPR-Param and TCPR-Rep (at the optimal $\beta = 10^{-6}$) achieve **25.20%** joint mean accuracy. This is identical to BSigmoid-Router (Reg) (**25.20%**) and slightly worse than the completely unregularized BSigmoid-Router (**25.50%**).
  - Furthermore, Figure 1 shows that any active regularization scale ($\beta \ge 1.0$) leads to a severe performance collapse. Therefore, TCPR is either mathematically inactive (when $\beta \le 10^{-6}$) or actively harmful (when $\beta \ge 1.0$). It does *not* prevent collapse or bridge the performance gap; instead, the unregularized sigmoidal router is what prevents collapse and achieves the best performance.
  - The authors' own analysis in Section 4.4 explicitly deconstructs and admits this failure.

- **Claim 2: "Decoupling routing pathways via independent sigmoidal projections (BSigmoid-Router) completely eliminates the competitive zero-sum bottleneck of Softmax routing, achieving the top-performing multi-task profile (25.50% Joint Mean)."**
  - **Verdict: Strong Support.** The results in Table 1 show a massive performance gap between the Softmax-based BL-Router (**19.10%**) and the Sigmoid-based BSigmoid-Router (**25.50%**). This simple change yields a +6.40% improvement and also outperforms the highly complex, wave-interference-based QWS-Merge (**21.80%**). This is a compelling, high-signal result.

- **Claim 3: "Standard unregularized routing heads suffer from severe representational collapse on high-conflict datasets."**
  - **Verdict: Supported.** The BL-Router (Softmax, Unreg) collapses to **9.60%** accuracy on MNIST (near random guessing, given that it's a 10-class task) and **15.20%** on SVHN.
