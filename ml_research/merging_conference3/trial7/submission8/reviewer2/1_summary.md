# 1. Summary of the Paper

## Main Topic and Motivation
The paper addresses test-time dynamic model merging (test-time expert blending), which dynamically routes input-dependent sample-wise representations to a blend of expert models (e.g., lightweight LoRA adapters fine-tuned on individual tasks) without expanding the physical footprint of the deployed model. The authors identify and investigate two primary real-world failure modes of standard dynamic routing models under deployment conditions:
1. **Calibration Data Scarcity (The Small-$N$ Overfitting Regime):** Standard parametric routers require labeled calibration data to learn task-routing coefficients. When this calibration set is limited ($N \le 32$), parametric routers experience transductive overfitting and representation collapse, leading to high seed variance and poor generalization.
2. **Deployment Stream Batch Heterogeneity (Heterogeneity Collapse):** Dynamic routers typically assume homogeneous batches at inference. In mixed-task heterogeneous streams, processing mixed batches causes representations of different tasks to smooth/average out, leading the router to output near-uniform weights across all experts (referred to as *heterogeneity collapse*).

To mitigate these vulnerabilities, the authors propose a dual-pathway routing system called **Confidence-Gated Hybrid Routing (CGHR)**, augmented by a stream-partitioning technique named **Micro-Batch Homogenization (MBH)**.

---

## Technical Approach and Formulation

### System Model: The Isolating Coordinate Sandbox
The main experiments are conducted using a synthetic 1-layer coordinate-disjoint multi-task simulation framework (the *Isolating Coordinate Sandbox*), where:
- The global hidden feature dimension is $D=192$ and there are $K=4$ specialized experts.
- The representation space is divided into $K$ disjoint block coordinates, each of dimension $d = D/K = 48$.
- Each expert is a classification head $W_k \in \mathbb{R}^{C_k \times d}$ representing MNIST, Fashion-MNIST, CIFAR-10, and SVHN, with pre-normalized classifier weights (Unit-Norm Calibration).
- Noise is added to replicate realistic ceilings (MNIST and Fashion-MNIST have $\sigma = 0.05$ with $100\%$ ceiling; CIFAR-10 has $\sigma = 0.35$ with $88.6\%$ ceiling; SVHN is a deliberately weak expert with $\sigma = 1.25$ and $26.4\%$ ceiling).
- Inactive coordinate blocks are filled with standard random Gaussian noise.

### Confidence-Gated Hybrid Routing (CGHR)
CGHR balances learned parametric routing (Pathway A) against zero-shot parameter-free projection routing (Pathway B):
- **Pathway A (Parametric Gating):** A lightweight linear router optimized using Cross-Entropy loss over task labels on a small calibration set of size $N$.
- **Pathway B (Parameter-Free Subspace Routing, PFSR):** Projects representation blocks $z_{k, b}$ onto expert classification manifolds using cosine similarity, normalized by a calibration factor $\sqrt{2\log C_k / d}$ to enable unbiased comparison.
- **Confidence Gating:** Sample-wise selection using a threshold $\gamma_{\text{conf}} \in [0, 1]$. If the parametric confidence is high ($\ge \gamma_{\text{conf}}$), use Pathway A routing coefficients; otherwise, fall back to Pathway B. The confidence metric can be Max Probability, Negative Entropy, or Margin.

### Micro-Batch Homogenization (MBH)
To resolve heterogeneity collapse, MBH partitions a heterogeneous batch $X$ of size $B$ on the fly into $G \le K$ homogeneous micro-batches based on the predicted task argmax. It averages routing coefficients locally within each micro-batch, fuses the expert parameters, performs specialized inference on each micro-batch, and collates the final results back into the original order.

---

## Key Findings and Quantitative Results
Based on sweeps over 5 independent random seeds in the sandbox environment:
1. **Overfitting Resilience:** On a scarce calibration set ($N = 16$), standard unregularized linear routers suffer from severe overfitting and variance. CGHR maintains near-perfect, flatline-like stability, leveraging the non-parametric PFSR fallback to prevent transductive overfitting, while scaling up performance as $N$ increases.
2. **Gating Threshold Sensitivity:** CGHR displays a robust "peak performance envelope" at intermediate thresholds ($\gamma_{\text{conf}} \approx 0.85$ using Max Probability), outperforming both pure parametric and pure non-parametric routers.
3. **Resilience to Heterogeneity Collapse:** Under mixed-task streams, standard routers suffer severe degradation as batch size $B$ scales to $512$ (collapsing to Uniform Merging at $63.10\%$). The integration of MBH completely prevents this collapse, maintaining flat, robust performance across all batch sizes.
4. **Systems Latency Trade-offs:** The authors analyze systems latency and find that while MBH introduces sequential sub-pass overhead for small batches ($1.67\times$ at $B=1$ on CPU), their proposed *Homogeneity Bypass* and *Fusion Weight Caching* successfully mitigate serving latency without accuracy loss.

---

## Explicitly Claimed Contributions and Evidence
The paper claims the following main contributions:
- **Characterization of Overfitting and Heterogeneity Collapse:** Formally conceptualizes "transductive collapse" under data scarcity and "heterogeneity collapse" under batch heterogeneity in dynamic model merging. *Evidence:* Demonstrated via empirical sweeps in the Isolating Coordinate Sandbox (Figures 2 and 3).
- **Confidence-Gated Hybrid Routing (CGHR):** Proposes a dual-pathway routing pattern that gates predictions sample-by-sample and falls back to a robust, zero-shot projection router (PFSR). *Evidence:* CGHR outperforms baselines under small-$N$ regimes and achieves a stable joint mean of $76.44\% \pm 0.09$ under standard settings (Table 1).
- **Micro-Batch Homogenization (MBH):** Proposes a dynamic partitioning strategy to isolate mixed-task streams. *Evidence:* Empirical verification showing perfectly flat performance curves for CGHR+MBH up to $B=512$ (Figure 3).
- **Theoretical and Practical Extensions:** 
  - *UNC-PFSR Equivalence Theorem:* Mathematically proves the equivalence of global unpartitioned and local block-sliced projections under Unit-Norm Calibration. *Evidence:* Proof in Appendix F and empirical validation with Inference-Time block-wise Unit-Norm Calibration (IT-UNC) showing a recovery of accuracy from $30\%$ to $74.2\%$ (Table 4).
  - *Mitigations against Routing Errors:* Details Soft-Confidence Fallback Homogenization and Hierarchical MBH (H-MBH). *Evidence:* Quantitative sweeps under simulated routing errors showing robustness gains (Table 5).
  - *Systems-Level Optimization:* Outlines Fusion Weight Caching, Triton Segmented-BGEMM kernels, and Warp Padding. *Evidence:* Empirically evaluates weight caching showing $2.87\times$ speedup (Table 3) and simulates warp padding under stream skew (Table 2).
