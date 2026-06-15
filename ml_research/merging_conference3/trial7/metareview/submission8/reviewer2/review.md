# Peer Review

## Summary of the Paper
This paper investigates test-time dynamic model merging (test-time expert ensembling), focusing on two critical vulnerabilities in deployment streams:
1. **Calibration Data Scarcity (Small-$N$ Regime):** Parametric routers require labeled calibration data and overfit severely when $N$ is small ($N \le 32$), leading to transductive overfitting and representation collapse.
2. **Deployment Stream Batch Heterogeneity:** Mixed-task batches cause dynamic routers to average representation dynamics across tasks, flattening routing coefficients—a phenomenon the authors term *heterogeneity collapse*.

To restore robustness, the authors propose **Confidence-Gated Hybrid Routing (CGHR)**, a dual-pathway system that gates predictions sample-by-sample, falling back to a training-free projection-based **Parameter-Free Subspace Router (PFSR)** when parametric confidence is low. They combine this with **Micro-Batch Homogenization (MBH)**, which dynamically partitions incoming mixed-task batches into homogeneous micro-batches on the fly, preventing representational smoothing.

The proposed framework is evaluated using a 1-layer synthetic **Isolating Coordinate Sandbox** that models expert ceilings across MNIST, Fashion-MNIST, CIFAR-10, and SVHN. The authors run extensive quantitative sweeps over 5 independent random seeds and show that CGHR+MBH maintains high stability under severe data scarcity ($N=16$) and prevents heterogeneity collapse up to batch sizes of $B=512$.

---

## Strengths and Weaknesses

### Strengths
1. **Important and Well-Motivated Focus:** The paper addresses highly realistic vulnerabilities of dynamic model merging under deployment conditions. Highlighting transductive overfitting and heterogeneity collapse as major serving-level bottlenecks is a valuable contribution.
2. **Deployment-Aware Systems Focus:** The authors do not restrict their focus to abstract algorithms. They provide an exceptionally thorough systems-level profiling, including CPU vs. GPU latency regimes, *Homogeneity Bypass*, *Fusion Weight Caching* ($2.87\times$ speedup with zero accuracy loss), warp divergence profiling under stream skew, custom *Triton Segmented-BGEMM kernels*, and LRU caching policies.
3. **Rigorous Empirical Analysis:** The paper features comprehensive quantitative evaluations with parallel sweeps run over 5 independent seeds. The authors thoroughly stress-test their system under unnormalized coordinate noise and simulated routing error propagation ($P_{\text{error}} \in [0, 0.75]$).
4. **Strong Theoretical Extensions:** The theoretical derivations are mathematically rigorous. The appendices contain valuable proofs, including extreme value normalization calibration (Appendix A), the *UNC-PFSR Equivalence Theorem* (Appendix F), and an SVD subspace projection protocol to handle overlapping representation manifolds (Appendix H).

### Weaknesses
1. **Severe Failures in Literature Integration and Contextualization:**
   - **Baseline Citations Missing:** The primary baselines evaluated in Table 1—VR-Router (Task-Variance Regularization), TSAR (Task-Space Anchor Regularization), and PFSR (Parameter-Free Subspace Routing)—are discussed as "recent efforts" in Section 2.3 but **completely lack citations**. This is a severe gap that makes baseline verification impossible.
   - **Ignoring Specialized Prior Art:** A review of the submission's bibliography database (`references.bib`) reveals 15 specialized predecessor papers from the *Transactions on Model Merging* covering transductive overfitting, dynamic routing, on-device model merging, and layer-averaging collapse. **Not a single one of these 15 highly relevant predecessor papers is cited in the main text.**
   - **False Novelty Claims:** The authors claim to "zoom in on" and conceptualize transductive collapse and heterogeneity collapse (or representation smoothing) for the first time. However, `PredecessorT2S1` is explicitly titled *"Transductive Overfitting in Multi-Task Weight Fusion"* and `PredecessorT5S5` is titled *"Layer-Averaging Collapse in Dynamic Weight-Space Routing"*. Presenting these failure modes as newly discovered while leaving the papers that originally studied them completely uncited (despite having them in their `.bib` database) is a severe misrepresentation of the scientific landscape.
   - **Citation Misattribution:** In Section 1 and Section 2.1, the authors cite `Wortsman2022` (Model Soups) for Task Arithmetic. Task Arithmetic was proposed by Gabriel Ilharco et al. in *"Editing models with task arithmetic"* (2023). Although `Ilharco2022` is present in their `.bib` file, they cited `Wortsman2022` (which averages models on the *same* task) instead, which is a major factual error.
2. **Idealized and Artificial Experimental Setup:**
   - **Structural Asymmetry:** The main sandbox experiments are heavily biased in favor of PFSR (Pathway B), which receives block-sliced inputs ($z_{k, b} \in \mathbb{R}^{48}$), whereas the parametric router (Pathway A) must handle the noisy global vector ($z_b \in \mathbb{R}^{192}$). This pre-provides PFSR with privileged coordinate boundary knowledge.
   - **Sandbox Reliance:** The main results (Table 1) and sweeps are evaluated exclusively within a synthetic, 1-layer disjoint coordinate sandbox. In realistic deep architectures, representation spaces are highly overlapping and non-orthogonal. While the authors outline an elegant SVD projection roadmap (Section 9.2) and a toy overlapping subspace simulation (Table 6), the actual performance of CGHR+MBH on standard multi-task benchmarks (such as GLUE or DomainNet) with pre-trained Transformers remains unverified.
   - **Artificially Crippled Baselines:** In the disjoint coordinate sandbox, advanced static merging methods (Task Arithmetic, TIES-Merging, DARE) mathematically reduce to Uniform Merging due to the lack of overlapping parameter delta conflicts. This trivializes parameter interference and over-simplifies the baseline comparison, giving dynamic routing an artificial advantage.

---

## Soundness
**Rating:** Good  
**Justification:**  
The methodology, mathematical formulations, and algorithmic flows are technically sound, and the proofs in the appendices are rigorous. The quantitative evaluations are exceptionally thorough within the chosen environment. However, the soundness of the comparative analysis is limited because:
1. The main experimental comparison relies on a structural asymmetry that biases results in favor of the parameter-free fallback.
2. The primary baseline implementations (VR-Router, TSAR, PFSR) are unverified due to a total lack of citations.
3. The evaluation is conducted entirely within a toy, 1-layer synthetic sandbox where advanced static baselines are artificially reduced to uniform averaging.

---

## Presentation
**Rating:** Fair  
**Justification:**  
While the paper is clearly written, well-structured, and features high-quality figures, the literature integration and citation practices fall far short of the standards expected for a top-tier conference. Proposing a method while failing to cite any papers for three primary baselines, misattributing Task Arithmetic to Model Soups, and completely ignoring 15 highly specialized predecessor papers in its own bibliography database represents a severe failure to accurately describe the landscape and properly attribute ideas.

---

## Significance
**Rating:** Fair  
**Justification:**  
The work is highly significant for low-resource and edge ML practitioners, where the combination of CGHR and MBH provides a practical, hardware-agnostic solution to prevent heterogeneity collapse without requiring custom CUDA serving engines (such as S-LoRA). However, its significance is tempered by the exclusive reliance on a synthetic 1-layer sandbox, leaving its practical utility on realistic, overlapping representation manifolds in deep networks unproven.

---

## Originality
**Rating:** Fair  
**Justification:**  
The individual components of CGHR (confidence-based gating fallback) and MBH (label-based batch partitioning and local weight fusion) are straightforward combinations of well-established ensembling and batching principles. Furthermore, the conceptual originality of defining transductive collapse and heterogeneity collapse as new vulnerabilities is heavily undermined by the presence of uncited predecessor papers in their `.bib` database that explicitly study these exact phenomena.

---

## Overall Recommendation
**Score:** 3: Weak Reject  
**Justification:**  
The paper has clear merits: it addresses a highly realistic and practical problem in dynamic model merging, offers a sophisticated systems-level servability analysis, and provides exceptionally thorough empirical sweeps and rigorous mathematical derivations. 

However, these merits are currently outweighed by severe scholarly and literature integration failures. The complete omission of citations for its three primary baselines, the misattribution of Task Arithmetic, and the failure to reference or position its contributions relative to 15 highly specialized predecessor papers in its own `.bib` database (which explicitly study transductive overfitting and layer-averaging collapse) are critical weaknesses. Presenting these failure modes as newly discovered while leaving the original papers uncited is a significant misrepresentation of the field's state. 

To become ready for publication, the authors must perform a major revision that resolves these literature gaps, corrects the misattributions, and clearly articulates their actual technical "delta" relative to these specialized prior works.
