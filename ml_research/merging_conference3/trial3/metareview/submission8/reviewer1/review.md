# Peer Review

## 1. Summary of the Paper
This paper presents **GP-BayesMerge**, a Gaussian Process (GP) PAC-Bayes framework designed to regularize test-time model merging (TTA). Multi-task model merging combines task-specific expert neural networks (fine-tuned from a shared base model) into a single multi-task model without expensive retraining. 

To exploit layer-wise heterogeneity, recent test-time model merging methods (such as AdaMerging) optimize layer-wise merging coefficients on small unlabeled test batches by minimizing prediction entropy. The authors identify a vulnerability in this paradigm, termed the **Overfitting-Optimizer Paradox**: unconstrained layer-wise optimization aggressively fits the transductive noise of small calibration batches, leading to volatile, high-frequency spatial oscillations across adjacent layers and causing generalization collapse on unseen target test data.

To resolve this, the authors apply PAC-Bayes generalization theory directly to the control space of layer-wise merging coefficients ($\Lambda \in \mathbb{R}^{L \times K}$). By modeling the prior over coefficients as a continuous GP over normalized network depth (using Squared Exponential or Ornstein-Uhlenbeck kernels), they derive a quadratic precision-matrix ($\Sigma_\ell^{-1}$) regularizer. This formulation mathematically unifies distance-from-initialization and spatial-smoothness constraints. They further extend this to **MT-GP-BayesMerge**, which places a joint prior over tasks using a Kronecker product ($B \otimes \Sigma_\ell$), where the task correlation matrix $B$ is estimated online and data-free using activation Centered Kernel Alignment (CKA) on target calibration samples.

The method is evaluated on both a high-fidelity synthetic non-convex simulation and actual physical weight-merging of pre-trained CLIP ViT-B/32 and CLIP ViT-L/14 backbones across 8 real-world vision datasets.

---

## 2. Strengths

### Rigorous and Original Theoretical Grounding
The primary strength of this work is its first-principles formulation. While existing works (such as RegCalMerge) rely on disjoint, heuristic spatial-smoothing penalties that require manual hyperparameter tuning, this paper derives a unified quadratic precision-matrix regularizer directly from PAC-Bayes generalization theory. Placing the PAC-Bayes prior on the low-dimensional control space of merging coefficients rather than the high-dimensional weight space is a creative and mathematically elegant approach.

### Elegant Architectural and Scalability Design
The continuous Gaussian Process prior over normalized network depth is highly elegant. The multi-task joint prior formulation using the Kronecker product ($B^{-1} \otimes \Sigma_\ell^{-1}$) successfully models cross-task correlations while bypassing the cubic scaling cost of joint covariance inversion. Additionally, recommending the Ornstein-Uhlenbeck (OU) process for ultra-deep models—showing that its tridiagonal precision matrix can be assembled analytically in $O(L)$ linear time—demonstrates deep awareness of physical scalability for modern large foundation models.

### Outstanding Empirical Stability on Physical Weights
GP-BayesMerge and MT-GP-BayesMerge demonstrate exceptional capacity to reduce optimization volatility across random seeds. In the physical experiments on SVHN, unconstrained Layer-Wise AdaMerging exhibits a standard deviation of $\pm 1.84\%$, whereas GP-BayesMerge stabilizes this to $\pm 0.35\%$, proving that the continuous spatial prior is highly effective at anchoring optimization in robust basins.

### Excellent Writing and Transparency
The presentation is outstanding. The paper is exceptionally clear, beautifully structured, and professional. The Appendix is incredibly thorough, offering rigorous proofs and addressing several deep theoretical and practical edge cases, such as the Truncated Gaussian Paradox (resolving the KL explosion), the Surrogate-to-Target Risk Gap, and Boundary Truncation Bias.

---

## 3. Weaknesses

### 1. Selective Omission of Key Baselines in Physical Weight Merging
The most critical empirical concern is the omission of the strongest spatial-regularization and subspace-constraint baselines in the physical weight-merging experiments. 
In Table 1 (simulated results), the paper compares the proposed method against **RegCalMerge (ESR)** and **PolyMerge (Subspace)**. However, in Table 2 (actual physical weights of CLIP ViT-B/32), these two baselines are completely absent. Since RegCalMerge and PolyMerge are the most direct competitors that also try to address spatial volatility and transductive noise, omitting them from the physical weight experiments represents a significant gap. To establish the true empirical superiority of GP-BayesMerge, these baselines must be evaluated on the physical weights.

### 2. Inherent Design Bias and Exaggerated Baseline Collapse in Simulation
The "high-fidelity non-convex simulation" models the ground-truth optimal parameters using a spatial covariance matrix $\Sigma_{\text{true}}$ with a decaying correlation structure. This is structurally identical to the GP prior used by GP-BayesMerge. While the authors transparently acknowledge this design bias, it remains a methodological limitation because the simulation is inherently structured to favor spatially-smooth regularizers. 
Furthermore, the simulated SVHN collapse under Standard AdaMerging ($46.64\%$) is artificially severe due to excessive transductive noise injection ($\sigma^2 = 0.12$). In the actual physical weight experiments (Table 2), unconstrained Layer-Wise AdaMerging achieves a strong $87.02\%$, outperforming the Task Arithmetic baseline ($82.05\%$). This means that the "catastrophic generalization collapse" motivated as the core problem does not actually materialize on physical weights under realistic adaptation conditions, indicating a substantial gap between the simulated "Overfitting-Optimizer Paradox" and physical reality.

### 3. Compute and Latency Overhead of Online CKA
For MT-GP-BayesMerge, the task-correlation matrix $B_{\text{online}}$ is estimated dynamically on-the-fly by computing the pairwise activation CKA similarities on target calibration samples. Extracting these activations requires feeding the calibration batch through *each of the $K$ distinct task expert models*. If there are 8 tasks (as in Table 2), this increases the test-time forward pass overhead by $8\times$. While the paper claims CKA adds "negligible computational overhead," it glosses over this substantial activation-generation cost. Running 8 separate models in parallel or sequence is a major latency and memory bottleneck for real-time edge deployments.

### 4. Overlapping Performance Intervals and Small Sample Size
All physical experiments are evaluated across only 3 random seeds, which is a small sample size for establishing strong statistical confidence. The accuracy gains of GP-BayesMerge over Layer-Wise AdaMerging++ in Table 2 are modest (average accuracy of $82.35\%$ vs $81.15\%$, a delta of $1.2\%$). Several task-specific performance intervals overlap significantly (e.g., SVHN's $90.15 \pm 0.35\%$ vs $89.62 \pm 0.98\%$). The paper lacks statistical significance tests (such as paired t-tests) to confirm that these modest gains are statistically sound rather than random fluctuations.

---

## 4. Questions and Requested Clarifications

1. **Missing Baselines on Physical Weights**: Can the authors report the physical weight-merging performance of **RegCalMerge** and **PolyMerge** on the 8 tasks in Table 2? This is critical to verify whether the theoretically derived GP-prior actually outperforms prior spatial-regularization or subspace-constraint heuristics on real weight deployments.
2. **Overhead of Online CKA**: What is the wall-clock latency and memory cost of running the calibration batch through all $K$ expert models to extract the activations for the online CKA estimation in MT-GP-BayesMerge? If $K$ is large, how do you mitigate this $K\times$ inference-time forward-pass overhead?
3. **Discrepancy in SVHN Performance**: Why does Standard AdaMerging collapse to $46.64\%$ on SVHN in the simulation, while on actual physical weights Layer-Wise AdaMerging achieves a strong $87.02\%$ (outperforming uniform Task Arithmetic)? Can you clarify if the transductive noise level ($\sigma^2 = 0.12$) injected in the simulation was artificially high?
4. **Statistical Rigor**: Could you provide paired t-tests or p-values comparing GP-BayesMerge against Layer-Wise AdaMerging++ across the 3 seeds to demonstrate statistical significance?

---

## 5. Detailed Evaluation of Criteria

* **Soundness**: **Good**. The mathematical derivations, PAC-Bayes formulations, and properties of the precision matrices are highly sound. The empirical soundness is slightly limited due to the inherent design bias of the simulation and the selective omission of key baselines (RegCalMerge and PolyMerge) in the physical weight experiments.
* **Presentation**: **Excellent**. The paper is beautifully written, exceptionally clear, and highly structured. The Appendix is incredibly thorough and professional.
* **Significance**: **Good**. The paper addresses a highly relevant and active problem (unsupervised test-time model merging). By providing a rigorous theoretical foundation based on PAC-Bayes theory, it could heavily influence future research, encouraging a transition from empirical heuristics to mathematically justified prior structures.
* **Originality**: **Excellent**. Applying PAC-Bayes directly to the low-dimensional control space of layer-wise merging coefficients, formulating it as a continuous GP prior over normalized depth, and deriving the Kronecker multi-task joint prior with online CKA activation similarities are highly original and creative.

---

## 6. Overall Recommendation
**Score: 4 (Weak Accept)**

**Justification:**
This is a technically solid, highly elegant, and beautifully written paper that makes a strong theoretical and practical contribution to the field of parameter-space model merging. Connecting empirical layer-wise test-time adaptation to first-principles PAC-Bayes theory is a substantial conceptual advance, and the continuous GP prior successfully and dramatically stabilizes optimization across random seeds. 

However, there are notable empirical limitations that prevent a higher rating at this stage: (1) the omission of the strongest direct baselines (RegCalMerge and PolyMerge) from the main physical weight experiments, (2) the inherent design bias of the simulation, which exaggerates the failure of the baselines, and (3) the unquantified computational overhead of running $K$ distinct expert networks at test-time for online CKA. If the authors can incorporate the missing baselines on physical weights and address the CKA forwarding latency, this paper would be a strong candidate for a full accept.
