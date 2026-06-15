# Evaluation Component 1: Summary of the Paper

## 1. Main Topic and Approach
This paper investigates the vulnerabilities of **dynamic model merging** (test-time expert blending) under realistic deployment conditions and proposes a robust framework to mitigate them. Specifically, the paper focuses on two key issues:
* **Calibration Data Scarcity (Small-$N$ Overfitting):** Standard parametric routers overfit and suffer from transductive collapse when trained on very small calibration datasets.
* **Deployment Stream Batch Heterogeneity (Heterogeneity Collapse):** Standard dynamic routers experience representational smoothing and flatline performance when processing mixed-task batches at inference time.

To address these vulnerabilities, the authors propose a dual-pathway system consisting of:
1. **Confidence-Gated Hybrid Routing (CGHR):** A dynamic gating mechanism that routes inputs using a trained parametric linear router when its confidence is high, and falls back to a zero-shot, parameter-free subspace router (PFSR) when confidence falls below a threshold ($\gamma_{\text{conf}}$).
2. **Micro-Batch Homogenization (MBH):** An on-the-fly partitioning technique that groups incoming heterogeneous batches into homogeneous micro-batches based on predicted task labels, performing localized model merging and execution to prevent representation averaging.

---

## 2. Key Findings
* **Overfitting of Parametric Routers:** Standard parametric routing architectures (Linear, VR-Router) degrade significantly on complex tasks (e.g., CIFAR-10 joint accuracy drops to $54.80\%$) when calibration data is scarce.
* **PFSR Stability:** The parameter-free subspace router (PFSR) performs robustly in a zero-shot manner ($76.60\%$ joint accuracy), though it lacks the adaptive scaling capability of parametric models when data is plentiful.
* **Optimal Hybrid Envelope:** CGHR discovers an optimal "peak performance envelope" at intermediate thresholds ($\gamma_{\text{conf}} \approx 0.85$ under Max Probability), combining the precision of parametric routing with the zero-shot robustness of PFSR.
* **Prevention of Heterogeneity Collapse:** Standard routers collapse to the baseline performance of static Uniform Merging ($63.10\%$) as the batch size increases to $B=512$ on mixed-task streams. Integrating MBH completely prevents this collapse, maintaining flat, robust performance across all batch sizes.
* **Cascaded Routing Failure:** On very noisy, weak experts (such as the simulated SVHN task with $\sigma_3 = 1.25$), both PFSR and CGHR experience cascaded routing failures, frequently misrouting samples to cleaner experts.

---

## 3. Explicitly Claimed Contributions (with Evidence)
1. **Empirical Characterization of Failure Modes:** The paper identifies and systematically evaluates the transductive collapse of parametric routers under data scarcity and the heterogeneity collapse of dynamic merging under mixed-task streams (evidenced by extensive sweeps in Section 4).
2. **Confidence-Gated Hybrid Routing (CGHR):** The authors formulate a hybrid routing mechanism that bridges parametric and parameter-free models. They show that CGHR maintains high performance under extreme scarcity ($N=16$) and scales gracefully with more calibration data (evidenced by Figures 1 and 2).
3. **Micro-Batch Homogenization (MBH):** The paper proposes MBH to protect dynamic model merging in production environments. They show that MBH maintains robust ensembling accuracy across batch sizes up to $B=512$ (evidenced by Figure 3).
4. **Systems Latency and Caching Optimization:** The authors provide actual and simulated systems latency profiles for MBH and empirically validate a "Fusion Weight Caching" optimization that rounding coefficients to steps of $0.10$ achieves a $98.2\%$ cache hit rate and a $2.87\times$ speedup in weight fusion latency (evidenced by Tables 2, 3, 4, and 5).
5. **SVD Subspace Projections for Overlapping Environments:** The authors provide a theoretical extension and empirical proof-of-concept simulation using SVD projection operators to handle overlapping (non-orthogonal) representation spaces, recovering routing accuracy to $75.00\%$ (evidenced by Table 7).
