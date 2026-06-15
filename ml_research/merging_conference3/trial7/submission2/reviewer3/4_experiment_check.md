# Evaluation Task 4: Experimental Evaluation Check

## Baseline Coverage and Statistical Rigor
The paper's experimental evaluation is exceptionally thorough in terms of baseline comparisons and statistical rigor:
- **Baseline Selection**: Compares against five major ensembling and routing models (Static Uniform, Unregularized Linear, QWS-Merge SOTA, L3-Softmax, and unweighted PFSR + MBH). This ensures that both parametric (optimization-based) and parameter-free models are represented.
- **Statistical Significance**: Quantitative results are reported as means and standard deviations evaluated across **10 independent random seeds** (seeds 42 to 51), which is a high standard of empirical rigor.
- **Stability and Stream Robustness**: Tests ensembling performance under heterogeneous, mixed-task streams across varying batch sizes ($B=1$ to $512$).

## Empirical Verification of Failure Modes
The results successfully and convincingly verify the key failure modes of existing dynamic routers:
- **The Dynamic Routing Paradox**: All parametric routers (Linear, QWS-Merge, L3-Softmax) catastrophically collapse to near-uniform accuracy ($\approx 36-39\%$) on a 64-sample calibration split, proving they cannot generalize under extreme data scarcity.
- **Vectorization Collapse**: Parametric routers fluctuate wildly or collapse when batch-averaging is removed ($B=1$).
- FIOSR, combined with Micro-Batch Homogenization (MBH), demonstrates absolute flat-line stability across all batching regimes.

## The "Reality Gap" on Physical Models (Practitioner's Analysis)
While the empirical results are statistically rigorous, there is a substantial gap between the performance in the synthetic "Analytical Coordinate Sandbox" and the performance on actual, physical models. This is a critical concern for any real-world deployment:

1. **Severe Performance Diminution on Physical Models**:
   - In the synthetic homogeneous sandbox, FIOSR outperforms the flat Cosine PFSR baseline by a massive **8.56%** absolute accuracy ($76.86\%$ vs $68.30\%$).
   - In the high-fidelity simulated LoRA activation space, the joint classification accuracy improvement of FIOSR over PFSR is **6.67%** absolute ($77.00\%$ vs $70.33\%$).
   - However, in the **physical end-to-end ResNet-18 deployment** (evaluated on real MNIST, FashionMNIST, and SVHN features), the performance gains are **highly modest**:
     - Routing accuracy: 56.33% (PFSR) vs. 59.00% (FIOSR) (an improvement of only **+2.67%**).
     - Joint ensembling accuracy: 50.67% (PFSR) vs. 52.00% (FIOSR) (an improvement of only **+1.33%**).
   
   This dramatic reduction in empirical gains suggests that in realistic, high-dimensional activation spaces, the assumption of coordinate-aligned noise does not hold, and real representations have dense, non-axis-aligned coordinate correlations. Under these conditions, the simple diagonal Fisher Information Matrix provides only marginal utility over standard unweighted cosine similarity. A practitioner must question whether a mere 1.33% joint accuracy improvement in an actual deployment justifies the additional complexity of dFIM estimation, pre-calibration mean-centering, and extreme-value normalization.

2. **The Calibration Size ($N_c$) Bottleneck**:
   The sensitivity analysis reveals a sharp statistical phase transition. At extremely scarce calibration regimes ($N_c \le 4$ samples per task), the variance estimator is mathematically underdetermined and unstable. At $N_c = 2$, FIOSR achieves **55.97%** accuracy, which represents a massive **-9.48%** absolute loss compared to the flat baseline. This means that if calibration data is extremely limited, the proposed method actually hurts performance significantly compared to a simple, unweighted cosine similarity, presenting a serious operational risk.

3. **Gating Limit and Ensembling Compromise**:
   The Top-$M$ expert gating ablation shows that setting $M=1$ (hard routing) completely eliminates sequential batch-partitioning latency while achieving competitive performance ($76.87\%$). However, as the authors acknowledge, setting $M=1$ collapses the framework back to a hard task-routing selection mechanism, which raises a fundamental conceptual question: if we are simply doing hard routing to a single expert, why perform complex parameter-space weight merging in the first place?
