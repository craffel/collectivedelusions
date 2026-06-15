# Evaluation Part 4: Experimental Evaluation and Claims Check

## Experimental Design and Methodology
The experimental evaluation is exceptionally thorough, structured, and rigorous. It combines a highly controlled, high-fidelity synthetic simulation environment with a proof-of-concept empirical validation on physical neural representations.

### 1. Synthetic Sandbox (ICS)
The 14-layer Isolating Coordinate Sandbox (ICS) is analytically designed to model the activation-subspace and power-law performance characteristics of a Vision Transformer. 
* **Strengths:** The authors provide a brilliant, high-dimensional geometric analysis in Section 4.1, justifying why extreme noise coefficients ($\sigma_3 = 1.20$) are solvable in a $D=192$ dimensional representation space due to projection geometry, while explaining how **Cosine Similarity Dilution** shrinks raw similarities toward zero. This mathematical clarity justifies why the non-linear Lotka-Volterra dynamics act as a powerful attractor network to amplify diluted coordinate differences.
* **Limitations:** The accuracy results are governed by an analytical performance model (Equation 13). This abstracts away complex physical adapter-blending dynamics, which could present more non-linear behaviors than a simple power-law function with a bilinear interference penalty.

### 2. Physical Model Verification (Section 4.7)
Evaluating the non-parametric routers on CLS token activations extracted from Layer 12 of a pre-trained `vit\_tiny\_patch16\_224` model across MNIST, Fashion-MNIST, CIFAR-10, and SVHN is a **major strength**. It directly addresses the "simulation-to-reality" gap.
* **Non-Strawman Baselines:** The inclusion of both a **Few-Shot Linear Router** (16 samples per task) and a **Fully-Optimized Linear Router** (64 samples per task) provides a highly rigorous, competitive baseline.
* **Joint Evaluation Metric:** The construction of a mathematically rigorous **joint 40-class probability distribution** (Section 4.7) to evaluate downstream classification is a highly appropriate, sound metric. It naturally penalizes incorrect routing decisions, representing a true systems-level evaluation.

---

## Baseline Selection
The baseline selection is comprehensive and extremely fair:
1. **Expert Ceiling:** Defines the empirical upper bound of perfect routing.
2. **Uniform Merging:** Represents a baseline where no routing or selection occurs.
3. **Linear Router (Weight-Space):** A trained parametric head that averages coefficients to merge parameters at the batch level.
4. **Linear Router (Act):** The same trained head performing sample-wise activation blending, isolating the weight-space constraint from parametric routing itself.
5. **SABLE:** A non-parametric activation ensembling baseline using raw similarity projections.
6. **SPS-ZCA:** The prior state-of-the-art framework using sharp, winner-take-all temperature-scaled routing.

---

## Strength of Quantitative Evidence
The empirical findings strongly support the paper's central claims:

1. **State-of-the-Art Performance:** Table 1 and Table 7 confirm that ESM-LVC matches or exceeds existing non-parametric routing baselines (SABLE, SPS-ZCA) across both synthetic and physical environments.
2. **Noise Resilience:** Table 2 demonstrates ESM-LVC's self-regulating noise-filtering properties. Under extreme noise (Scale 2.5), ESM-LVC outperforms SOTA SPS-ZCA by **+2.63%** absolute.
3. **Batch Heterogeneity Collapse:** Figure 1(b) and Table 1 provide clear, empirical proof of heterogeneity collapse. The Linear Router (Weight-Space) suffers a **-19.86%** performance drop at $B=512$, while ESM-LVC maintains flatline performance (0.00% collapse) because activation blending occurs sample-wise.
4. **GMC-BSC Breakthrough:** Table 7 and Table 8 show that the multi-centroid GMC extension successfully breaks the single-centroid attractor bottleneck, boosting physical routing accuracy to **93.50%** (clean) and **89.75%** (extreme noise $\sigma=2.0$), outperforming the Fully-Optimized Linear Router by **+4.75%** absolute under noise.

---

## Crucial Observations and Scientific Transparency
The experimental section stands out for its high level of scientific honesty and intellectual transparency:

* **Attractor Equivalence Disclosed:** The authors openly disclose that SABLE, SPS-ZCA, and single-centroid ESM-LVC share identical routing accuracies in Table 7 because they are driven by the same Zero-Shot Centroid Alignment (ZCA) coordinates. This proves that under pure hard-gated routing, the continuous solver does not alter the argmax, and its true benefits are realized under soft, multi-expert co-activation.
* **Low Downstream Classification Accuracies Explained:** The joint 40-class classification accuracies in Table 8 are relatively low (approx 20%-28%). Rather than hiding this, the authors provide a rigorous diagnostic explanation: individual classifiers trained on 64-sample calibration splits on a tiny pre-trained ViT feature space have a low average clean accuracy of $29.75\%$, which mathematically bounds the joint multi-task performance. This transparent explanation is highly refreshing.
* **The Regularization Anomaly Resolved:** Under moderate noise ($\sigma = 1.5$), SABLE achieves higher accuracy than the rigid winner-take-all SPS-ZCA because soft ensembling acts as a regularizer. The authors show that the proposed **E-ITAS** and **DM-BSC** operators successfully resolve this trade-off by dynamically scaling back the sharpening exponent based on routing entropy, matching winner-take-all performance under clean settings while retaining soft regularization under noise.
