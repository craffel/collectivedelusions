# 1_summary.md - Summary of the Submission

## Main Topic and Motivation
The paper addresses test-time dynamic model merging (expert blending) at inference time, where lightweight task-specific adapters (like LoRA) are dynamically blended using an input-dependent router. The motivation is to achieve multi-task capability without expanding the physical deployment footprint. 

Specifically, the authors identify two critical vulnerabilities in real-world deployment:
1. **Calibration Data Scarcity (The Small-$N$ Overfitting Regime):** Standard parametric routers require labeled calibration data to learn task-routing coefficients. When this dataset is very small ($N \le 32$), parametric routers overfit, leading to high seed variance and poor test generalization ("transductive collapse").
2. **Deployment Stream Batch Heterogeneity (Heterogeneity Collapse):** Standard routers assume homogeneous batches at inference time. In reality, deployment streams have mixed tasks within a batch. Processing heterogeneous batches causes task representation dynamics to average out across tasks, leading the router to output uniform weights ("heterogeneity collapse").

---

## Proposed Approach
To resolve these issues, the authors propose a dual-pathway routing system augmented by a stream-partitioning technique:
1. **Confidence-Gated Hybrid Routing (CGHR):** A hybrid gating mechanism that routes high-confidence samples using a lightweight, trained parametric linear router (Pathway A), while falling back to a training-free Parameter-Free Subspace Router (PFSR) (Pathway B) when prediction confidence falls below a threshold $\gamma_{\text{conf}}$.
2. **Micro-Batch Homogenization (MBH):** On-the-fly partitioning of mixed-task batch streams into homogeneous micro-batches based on predicted task labels. Routing coefficients are calculated and weights are fused independently per micro-batch, bypassing batch-averaging representation smoothing.
3. **Advanced Extensions in Appendices:**
   - **IT-UNC (Inference-Time block-wise Unit-Norm Calibration)** to recover global PFSR accuracy under coordinate noise.
   - **Mitigations for Cascaded Error Propagation:** Soft-Confidence Fallback Homogenization and Hierarchical MBH (H-MBH).
   - **SVD Subspace Projections** to handle overlapping representation spaces in deep architectures.

---

## Key Findings and Claims
- **Peak Performance Envelope:** Max Probability confidence gating under CGHR achieves peak joint mean performance at intermediate thresholds ($\gamma_{\text{conf}} \approx 0.85$).
- **Overfitting Mitigation:** Under extreme data scarcity ($N = 16$), CGHR maintains stable performance with near-zero variance across seeds ($\pm 0.09$), leveraging the robust zero-shot fallback of PFSR.
- **Robustness to Mixed Streams:** Integrating MBH maintains flat, robust performance up to batch size $B=512$, whereas standard routers collapse to uniform ensembling accuracy ($63.10\%$).
- **SVD Projection Recovery:** Under overlapping subspaces, SVD-projected PFSR is claimed to "filter out out-of-subspace noise," "bridging the gap to the clean Local PFSR baseline."
- **Mitigation Efficacy:** Soft-Confidence Fallback Homogenization is claimed to resolve medium-error collapses, and Hierarchical MBH is claimed to preserve high low-error performance.

---

## Explicitly Claimed Contributions (with Evidence provided in the paper)
1. **Systematic Auditing of Failures:** Extensive sweeps over confidence thresholds, calibration data size $N$, and batch size $B$ across 5 random seeds to identify "transductive collapse" and "heterogeneity collapse."
2. **CGHR + MBH Architecture:** Implementation of a dual-pathway, on-the-fly stream homogenization pipeline.
3. **Mathematical & Systems Frameworks in Appendix:** The authors provide detailed systems-level latency models, CUDA/Triton implementation blueprints, and theoretical proofs of equivalence (UNC-PFSR) and SVD projection manifolds.

*Evidence Baseline:* The paper provides evaluation tables and plots (Figure 1, 2, 3) and appendix tables (Tables 3, 4, 5) generated entirely using a synthetic 1-layer coordinate-isolated simulation called the **Isolating Coordinate Sandbox**. No real-world deep neural network architectures (e.g., Transformers, real LoRA experts) or standard benchmark datasets (e.g., GLUE, DomainNet) were empirically tested in the main experiments.
