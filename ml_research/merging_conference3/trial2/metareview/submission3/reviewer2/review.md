# Peer Review of Conference Submission

## Summary of the Paper
The paper addresses the issue of **transductive overfitting** during **test-time adaptation (TTA)** for multi-task model merging, a phenomenon the author terms the **"Overfitting-Optimizer Paradox"**. Adaptive merging methods, such as AdaMerging, optimize individual layer-wise coefficients at test-time on unlabeled target streams by minimizing an unsupervised surrogate loss ( Shannon entropy). The authors argue that this unconstrained optimization exploits spatial degrees of freedom to fit transductive noise, leading to highly jagged, non-physical coefficient profiles and catastrophic generalization collapse.

To resolve this, the authors propose **PolyMerge**, which projects the merging coefficient search space onto a continuous, low-degree polynomial of normalized layer depth. By reducing the learnable parameters per task from $L$ layers to $d+1$, PolyMerge acts as a spatial low-pass filter to reject transductive noise and prevent degenerate representations. The authors also propose **SplineMerge** to handle layer-wise structural heterogeneity using piecewise-continuous local splines.

The paper evaluates these methods across extensive multi-axis sweeps in a custom 12-layer Vision Transformer weight-merging simulator across 30 seeds. Additionally, the authors present two small physical validations: one on a 12-layer PyTorch Residual MLP, and another on a pre-trained CLIP ViT-B/32 foundation model using a subset of real test images.

---

## Overall Assessment & Recommendation

### Recommendation: 3: Weak Reject
*The paper has clear merits in terms of presentation, statistical rigor, and codebase transparency, but also possesses several critical methodological weaknesses and empirical limitations that outweigh its merits in its current form. Revisions and more substantial physical experiments are required before this work can be meaningfully built upon by others.*

### Soundness Rating: Fair
*While the authors are transparent about their experimental configurations, the primary results are restricted to a custom synthetic emulator. The physical validations are executed on exceptionally small datasets (24 samples and 50 images) which are statistically suspect, and the main proposed method (PolyMerge) suffers from severe underfitting (a 12% performance regression) on physical weights.*

### Presentation Rating: Excellent
*The writing style is highly polished, professional, and mathematically precise. The figures and tables are detailed and accompanied by excellent, self-contained captions.*

### Significance Rating: Fair
*The significance is limited by the fact that the continuous polynomial projection (PolyMerge) degrades performance on real weights, forcing a fallback to SplineMerge, which is conceptually identical to simple block-wise layer grouping. Furthermore, the proposed automated dynamic programming partition method performs worse than a simple manual heuristic.*

### Originality Rating: Fair
*The "Overfitting-Optimizer Paradox" is a grandiose re-branding of classical overparameterized overfitting. The proposed solution—using polynomials and splines to smooth layer weights—is a straightforward application of standard dimensionality reduction and spatial smoothing tools to a known problem.*

---

## Detailed Strengths and Weaknesses

### Major Strengths
1. **Excellent Academic Integrity:** The authors are highly transparent and honest, explicitly stating that their main results in Table 1 represent simulated performance metrics and do not involve physical weight merging on GPUs.
2. **Impressive Statistical Rigor:** Sweeping all experimental configurations across 30 independent random seeds and running paired t-tests over 120 evaluations represents a very high standard of statistical verification.
3. **Thorough Baseline Comparisons:** The paper includes a wide range of baselines, including Total Variation (TV) and $L_2$ regularization, early stopping, post-hoc spatial averaging (Mean Treatment), and multiple polynomial degrees $d \in \{0, 1, 2, 3\}$.
4. **Clear Actionability:** Providing a complete PyTorch implementation snippet for the `PolyMergeGenerator` in the Appendix makes the proposed method highly accessible and easy to integrate into existing pipelines.

---

### Major Weaknesses and Critical Flaws

#### 1. Methodological Limitation of the "Simulation-First" Evaluation
The bulk of the paper's empirical evidence (Table 1, Figure 2, Figure 3, Table 3) is generated entirely within a custom, hand-crafted weight-merging simulator. While this allows for rapid prototyping, it incorporates highly stylized assumptions (convex quadratic landscapes, linear additive merging approximations, and stylized spatial noise) that do not represent the actual non-linear representation dynamics of physical deep learning. A weight-merging paper must establish its primary results on physical weights rather than a synthetic sandbox.

#### 2. Exceptionally Small Sample Sizes in Physical Validations
To complement the simulator, the authors present physical experiments in Sections 4.6 and 4.7. However, the scale of these validations is so small that they fail to provide scientifically robust proof of generalization:
* **Residual MLP (Section 4.6):** The TTA stream consists of only **24 unlabeled samples** (12 per task). 
* **CLIP Foundation Model (Section 4.7):** The test stream consists of only **50 images** from CIFAR-10 and **50 images** from GTSRB, adapted for 15 steps.
On a stream of 50 images, a single classification change represents a massive 2% shift in accuracy. These tiny, toy setups are highly prone to high-variance artifacts and do not reflect realistic test-time adaptation deployments, which continuously process hundreds or thousands of streaming samples.

#### 3. Catastrophic Underfitting of the Primary "PolyMerge" Framework
In the CLIP physical validation (Table 4), the primary proposed method, **PolyMerge ($d=2$, Quadratic)**, suffers from a catastrophic **12% absolute accuracy collapse** on CIFAR-10 (dropping from $92.00\%$ to $80.00\%$ accuracy compared to the unoptimized Task Arithmetic baseline).
* This regression indicates that a global polynomial constraint severely restricts the functional capacity of physical weights across layers (underfitting the non-monotonic layer sensitivities).
* While SplineMerge (Piecewise Constant) recovers the performance ($92.00\%$), this piecewise constant variant is conceptually identical to **block-wise model merging** (grouping layers and optimizing a single weight per group). Block-wise merging is a highly standard baseline in merging literature, meaning the paper's actual novel contribution (continuous polynomials) degrades physical performance, while its successful variant (SplineMerge) is just a re-packaged baseline.

#### 4. Flawed Noise Assumption in the Mathematical Proof
The mathematical proof of Proposition 3.1 (Appendix B.1) assumes that transductive noise $\boldsymbol{\eta}$ is added **directly** to the coefficient vector ($\boldsymbol{\lambda}_k = \boldsymbol{\lambda}^*_k + \boldsymbol{\eta}_k$). 
* This is a highly unrealistic representation of test-time adaptation. In actual TTA, noise arises from the statistical perturbations of the unlabeled target data batch, and enters the parameters through the **gradients of the non-linear Shannon entropy loss**.
* Because the relationship between data streams, logits, entropy, and merging weights is highly non-linear, the linear projection argument ($\mathbf{P}\boldsymbol{\eta} \approx \mathbf{0}$) does not hold in practice. The proof is mathematically neat but physically meaningless.

#### 5. Failure of the Automated Boundary Partitioning Method
In Table 2, the authors compare manual uniform partitions against their proposed automated "Dynamic Programming (DP) Discovered Partitioning" for SplineMerge:
* Manual Uniform Partitioning: **$86.80\% \pm 6.79\%$**
* DP-Discovered Partitioning: **$86.12\% \pm 7.53\%$**
The automated optimization method degrades performance by 0.68%. The authors attempt to spin this as an "insightful and counter-intuitive result" showing "transductive boundary overfitting." However, from an algorithmic standpoint, if an automated boundary finder performs worse than a simple manual split, it represents a failure of the proposed optimization algorithm.

#### 6. Flawed Latency Profiling Environment
In Table 5, the authors report wall-clock optimization step latencies measured on an **Intel CPU** to claim that PolyMerge has "absolutely zero computational overhead."
* In practical deep learning, test-time adaptation is executed on GPUs, where latency is dominated by kernel launch overheads, GPU memory transfers, and PyTorch autograd graph construction, rather than CPU FLOPs. Latency profiling on a CPU is irrelevant for actual deployment.

#### 7. Rhetorical Overselling of Basic Concepts
The paper heavily sells standard phenomena using grandiose terminology. The "Overfitting-Optimizer Paradox" is simply **overfitting** resulting from overparameterization on small sample sizes. The "degenerate entropy minimization trap" is the classical **constant-predictor collapse** well-documented in test-time adaptation literature. Grounding the narrative in established machine learning terms would improve scientific precision.

---

## Detailed Questions & Feedback for the Authors

1. **Physical Scale:** Can the authors execute their physical CLIP validations on full-scale, standard TTA benchmarks (such as the complete CIFAR-10-C, ImageNet-C, or standard multi-task retrieval datasets) rather than restricting the evaluation to tiny subsets of 50 images? 
2. **Advanced Merging Baselines:** Why are there no comparison baselines to standard, advanced non-adaptive merging techniques such as **TIES-Merging** or **RegMean** in the physical validation tables (Table 3 and Table 4)? To prove that TTA is worth the online computational cost, you must compare it against static advanced merging.
3. **Addressing PolyMerge Collapse:** Why does PolyMerge ($d=2$) cause a 12% accuracy drop on physical CLIP weights, and how can the authors justify "PolyMerge" as the primary title and thesis of the paper when it underperforms simple Task Arithmetic?
4. **Noise Model Realism:** How do the authors reconcile their clean additive white noise model on the coefficients with the highly non-linear gradient noise that actual test-time adaptation optimizers encounter?
5. **GPU Latency Profiling:** Can the authors provide GPU wall-clock latency measurements to replace the CPU profiling in Table 5?
