# 4. Experimental Check and Empirical Validation

This section critically evaluates the experimental setup, datasets, baselines, and whether the quantitative results support the paper's claims.

---

## 1. Experimental Setup and Datasets
The primary empirical vehicle is the **Isolating Coordinate Sandbox**, which models:
- $K=4$ tasks and experts representing MNIST, Fashion-MNIST, CIFAR-10, and SVHN.
- Standalone expert ceilings pre-calibrated via Unit-Norm Calibration (UNC) and coordinate-specific noise levels: MNIST ($100.0\%$), Fashion-MNIST ($100.0\%$), CIFAR-10 ($88.6\%$), and SVHN ($26.4\%$).

### Critical Evaluation of the Setup:
- **Pros:** The sandbox acts as a highly controlled mathematical instrument. It allows the authors to perform precise parameter sweeps over the gating threshold ($\gamma_{\text{conf}}$), calibration size ($N$), and stream batch size ($B$) while isolating individual routing behaviors without the confounding factors of deep architectures or optimizer bugs.
- **Cons:** It is a synthetic, 1-layer coordinate-disjoint simulation. There are no real-world images or text inputs, and no deep feature extraction backbones. The "expert models" are simply single linear classification heads. This represents a significant gap from actual production deployments where feature extraction, representation drift, and multi-adapter inference are far more complex.

---

## 2. Baselines and Benchmark Evaluation
The benchmark evaluation in Section 4.1 is extensive:
- **Static Baselines:** Uniform Merging. (As discussed, advanced static methods like Task Arithmetic, TIES-Merging, and DARE reduce to Uniform Merging in this disjoint coordinate setup).
- **Parametric Baselines:** Linear Router (Unregularized), Linear Router (Regularized with $L_2$ decay), VR-Router (Task-Variance Regularization), and TSAR (Task-Space Anchor Regularization).
- **Non-Parametric Baselines:** Parameter-Free Subspace Routing (PFSR).

### Critique of Baseline Comparisons:
- **Artificially Crippled Static Baselines:** Due to the disjoint coordinate setup, advanced static model-merging baselines are unable to showcase their ability to resolve parameter conflicts, reducing their representative baseline to standard uniform averaging. This presents an over-simplified comparison where dynamic ensembling has a massive structural advantage.
- **Unverified Baseline Implementations:** Because VR-Router, TSAR, and PFSR are **completely uncited** in the text, it is difficult to verify whether their implementations conform to their original formulations. For example, TSAR relies on task-space anchors; since there are no real task embeddings in a 1-layer coordinate sandbox, how the anchors are calculated and mapped is structurally non-obvious.

---

## 3. Results and Support for Claims

The quantitative results are exceptionally thorough, with parallel sweeps run over 5 independent seeds. Below is an analysis of how well the results support each primary claim:

### Claim A: CGHR discovers a robust peak performance envelope at intermediate thresholds.
- **Support:** Figure 1 plots joint mean accuracy against the gating threshold $\gamma_{\text{conf}} \in [0, 1]$ across Max Probability, Negative Entropy, and Margin confidence metrics. A clear, robust peak is visible at $\gamma_{\text{conf}} \approx 0.85$ (for Max Probability) and similar intermediate envelopes for the other metrics. This strongly supports the claim that gating balances the high precision of parametric updates against the robust zero-shot fallback.

### Claim B: Under severe calibration data scarcity, CGHR maintains flatline stability.
- **Support:** Figure 2 sweeps calibration size $N \in \{16, 32, 64, 128, 256, 512\}$. At $N=16$, unregularized linear routers collapse and display massive variance across seeds. Standard regularization stabilizes them but still suffers. Our proposed CGHR maintains a flat line at $76.44\%$ across all regimes, seamlessly matching the zero-shot PFSR fallback at small $N$ and scaling up as $N$ increases. This strongly supports the claim.

### Claim C: The integration of MBH completely prevents heterogeneity collapse.
- **Support:** Figure 3 sweeps stream batch size $B \in \{1, 8, 32, 128, 512\}$ under mixed-task heterogeneous streams. Without MBH, all standard routers degrade rapidly to Uniform Merging ($63.1\%$) by $B=512$ due to batch-averaging logit smoothing. With MBH, both PFSR+MBH and CGHR+MBH show completely flat, collapse-free performance curves. This strongly supports the claim.

### Claim D: Systems acceleration via Fusion Weight Caching.
- **Support:** Appendix D evaluates weight caching. Table 3 shows that discretizing ensembling coefficients to a step of $0.10$ achieves an outstanding cache hit rate of $98.2\%$, cutting weight fusion latency by $2.87\times$ with absolutely zero accuracy degradation on our test splits. This provides solid empirical support for the caching optimization.

### Claim E: Theoretical extensions and routing error mitigations.
- **Support:** 
  - **IT-UNC:** Table 4 in Appendix F shows that standard Global PFSR collapses to $30.00\%$ under coordinate noise, but applying Inference-Time block-wise Unit-Norm Calibration (IT-UNC) completely recovers the clean localized upper bound of $74.20\%$.
  - **Routing Error Mitigations:** Table 5 in Appendix G sweeps routing error rate $P_{\text{error}} \in [0.0, 0.75]$. Standard MBH collapses to $62.34\%$ under a $30\%$ routing error rate (performing worse than Uniform Merging). Soft-Confidence Fallback Homogenization with $\beta=0.5$ completely eliminates this dip, maintaining a highly robust $64.14\%$ accuracy. Hierarchical MBH is also shown to preserve high performance in low-error regimes.
  - **SVD Subspace Projections:** Table 6 in Appendix H shows SVD-Projected Global PFSR successfully filters out cross-task noise under overlapping subspaces, recovering classification accuracy to $70.20\%$ (bridging the gap to the clean Local PFSR baseline of $71.50\%$).
- **Overall:** These extensive, quantitative ablation sweeps are highly rigorous and strongly support all claimed theoretical and architectural extensions.

---

## Summary of Experimental Check
The empirical validation of this paper is outstandingly thorough, featuring exhaustive sweeps across multiple dimensions and seeds, complete systems profiling (latency and cache rates), and comprehensive stress-testing (coordinate noise, routing error injection, overlapping subspaces). However, **the major caveat remains that all experiments are simulated within a synthetic, 1-layer coordinate sandbox.** The absence of evaluations on standard, real-world deep multi-task benchmarks (such as DomainNet or GLUE) with pre-trained Transformer backbones limits the direct practical validation of the framework, leaving it as a highly rigorous, mathematically verified sandbox prototype.
