# Mock Review

## 1. Summary of the Paper
This paper presents a rigorous, transparent, and deconstructive scientific audit of test-time adaptive model merging in deep foundation models (specifically CLIP ViT-B/32). Rather than proposing a highly complex, parameter-heavy architecture to claim marginal state-of-the-art (SOTA) improvements, the authors focus on the limits of extreme parameter frugality. They introduce **Barycentric Proximity-Anchored Merging (BPAM)**, a minimalist weight-space merging framework restricted to exactly $K$ global task-wise scalars (where $K=8$ on the standard 8-task image classification benchmark). 

BPAM is designed as a boundary probe to investigate where weight-space optimization fails and where extra degrees of freedom (such as layer-wise coefficients) or auxiliary adaptations (such as classification head tuning) become indispensable. The paper decouples weight-space merging from classifier head adaptation, showing that head adaptation drives the bulk of performance gains in parameter-constrained regimes. It also quantitatively resolves the "0-weight performance mystery" using Centered Kernel Alignment (CKA) metrics to demonstrate representation sharing in the compact fine-tuning loss basin.

---

## 2. Strengths
1. **Refreshing Scientific Honesty and Deconstructive Framing:** The paper takes an exemplary, self-reflective scientific approach. Instead of trying to obscure BPAM's performance gap compared to high-capacity methods (e.g., FoldMerge and SyMerge), it highlights this gap to map the physical boundary where layer-wise scaling and coordinate-warping become essential.
2. **Symmetric, Transparent Evaluation (Parts A & B):** Splitting the evaluation into strictly frozen classification heads (Part A) and active classification head tuning (Part B) is a masterclass in transparent experimental reporting. This decoupling reveals that classification head tuning is the primary driver of performance (+6.01% gain) in parameter-constrained regimes, exposing a major confounding factor in the test-time model merging literature.
3. **Exceptional Analysis of the "0-Weight Performance Mystery":** The paper provides a brilliant, quantitative explanation of why a merged model can achieve high classification accuracy on SVHN (78.15%) and MNIST (88.09%) under frozen heads, even when their specialized expert parameters are completely pruned (assigned a coefficient of $0.0000$). Using Centered Kernel Alignment (CKA) metrics, the authors show that other fine-tuned experts (like GTSRB) retain digit-like features that the frozen MNIST/SVHN classifiers can exploit, providing concrete evidence of representation sharing.
4. **Rigorous Validation of the Proximity Safeguard:** The authors are honest that the Mean-Field Proximity Penalty ($\beta=0.01$) is empirically redundant in the default 8-parameter regime due to the low search space acting as an intrinsic regularizer. They then design a highly rigorous extreme low-data calibration experiment (5 samples per class) to show that under extreme data scarcity, the proximity penalty is essential to stabilize optimization and prevent parameter drift, backed by a comprehensive hyperparameter sensitivity analysis of $\beta$.
5. **Outstanding Presentation and Writing Clarity:** The paper is exceptionally well-written, clear, and mathematically precise, with highly structured sections, self-contained formulations, and beautifully detailed tables.

---

## 3. Weaknesses & Areas for Improvement (Constructive Feedback)
While the paper is outstanding, addressing the following mathematical, empirical, and practical aspects would make the manuscript even more robust and complete:

### A. Ray-Scaling Projection vs. Exact Euclidean Projection
* **Critique:** To enforce the convex simplex constraints, the authors use a simple ray-scaling ($L_1$-normalization) projection:
  $$\lambda_k^{(t+1)} \leftarrow \frac{\lambda_k^{(t+1)}}{\sum \lambda_j^{(t+1)}} \quad \text{if} \quad \sum \lambda_j^{(t+1)} > 1.0$$
  While selected for extreme computational simplicity and preserving the relative directional ratios of updated coefficients, ray-scaling is *not* an exact orthogonal Euclidean projection onto the convex simplex (with respect to the $L_2$ norm). Non-orthogonal projections do not mathematically guarantee the same convergence properties as standard Projected Gradient Descent (PGD).
* **Suggestion:** Discuss this mathematical trade-off in Section 3.4. Compare or discuss its convergence behavior relative to standard sorting-based exact Euclidean projection algorithms (e.g., Duchi et al., 2008).

### B. Practical Overhead of Teacher-Guided Adaptation (Expert Leaks)
* **Critique:** The test-time adaptation objective requires computing predictions from all $K$ expert teachers on the target streams. If the primary goal of model merging is to avoid hosting multiple large-scale networks simultaneously due to memory and computational bottlenecks, requiring all $K$ expert networks to be active and run in parallel during the test-time adaptation/calibration phase introduces a massive peak in computational and memory footprints (requiring $K+1$ parallel forward passes).
* **Suggestion:** While the "post-hoc calibration" framing mitigates this during actual deployment, this adaptation-phase memory and compute overhead (which we refer to as the $K+1$ "expert leak" issue) is a severe practical constraint. The authors should explicitly discuss this overhead as a limitation in Section 4.5.

### C. Learning Rate and Optimizer Shared Configuration in Joint Adaptation
* **Critique:** In Section 3.4, the authors state that during joint optimization (BPAM-Full), the 8 merging coefficients and the 388,096 parameters of the task-specific classification heads are updated concurrently using the exact same learning rate ($\eta = 10^{-3}$) and optimizer (Adam) without any specialized schedule. Because the classification heads contain several orders of magnitude more parameters than the 8 scalar coefficients, their gradient magnitudes and landscape curvature are likely vastly different. Employing a single uniform learning rate without a specialized schedule or scaling factor could cause the classification heads to dominate the optimization process or lead to sub-optimal convergence of the merging coefficients.
* **Suggestion:** Explore or discuss the impact of asymmetric learning rates or specialized scheduling for the weight-space vs. head-space parameters. For example, evaluating whether a smaller learning rate on the classification heads allows the weight-space coefficients to adapt more meaningfully would strengthen the empirical analysis.

---

## 4. Questions for the Authors
1. **Exact Orthogonal Projection:** Did you explore exact orthogonal Euclidean projection onto the simplex during your early development phase? If so, did it make a notable difference in convergence speed, optimization stability, or final accuracy compared to the simple ray-scaling projection?
2. **Joint Optimization Learning Rates:** Since BPAM-Full barely outperforms Task Arithmetic + Head Tuning (+0.42% absolute), did you explore using asymmetric learning rates (e.g., a smaller learning rate for the classifier heads to prevent them from dominating the 8 global task-scalars)? Could a different optimization schedule potentially unlock better joint-space convergence?
3. **Teacher-Free Objectives:** For the computational/memory overhead (Expert Leaks) during adaptation, did you experiment with teacher-free adaptive objectives (such as self-training, entropy minimization on the merged model, or pseudo-labeling)? If so, how did their performance compare to the teacher-guided KL objective?
4. **Asymmetric Proximity Anchors:** In Section 3.4, you mention that future work could explore non-uniform prior anchors, such as assigning a higher baseline weight to the pre-trained base model $w_{base}$ due to its general-purpose representation capabilities. Did you run any preliminary experiments with such an asymmetric prior, and how did it affect performance compared to the uniform prior centroid?

---

## 5. Quantitative Ratings

* **Soundness:** **Excellent**
  The mathematical formulations are rigorous, the spatial configurations are clearly defined, the split-test calibration/OOD generalization design is robust, and the extreme low-data evaluation and CKA-based representation sharing validation are exceptionally thorough and scientifically sound.
* **Presentation:** **Excellent**
  The writing is clear, direct, and structured. Decoupling the results into frozen heads (Part A) and active heads (Part B) is highly informative and exemplary. The "0-weight performance mystery" is beautifully deconstructed and resolved.
* **Significance:** **Excellent**
  Exposing classification head adaptation as a major confounding factor in previous test-time adaptive model merging is of very high significance to the community. Establishing a minimalist lower-bound baseline (BPAM) and quantitatively proving representation-sharing via CKA will likely guide and influence future theoretical and empirical works exploring weight-space geometry.
* **Originality:** **Good**
  While the individual algorithmic components of BPAM (linear vector combinations, test-time adaptation, simplex projection, proximal regularization) are incremental, the conceptual and meta-analytical originality of this paper—focusing on a critical, deconstructive scientific audit and mapping the boundaries of parameter constraints—is exceptionally high.

---

## 6. Overall Recommendation

**Rating:** **5: Accept**
**Justification:** This is an exceptionally high-quality, rigorous, and scientifically honest paper. Instead of proposing a highly complex, parameter-heavy method to chase marginal SOTA improvements, the authors take a step back and perform a vital, deconstructive audit of test-time adaptive model merging. The symmetric evaluation (decoupling head and weight tuning) is highly informative, and the quantitative resolution of the "0-weight performance mystery" using CKA-based representation similarities is an outstanding piece of science. The minor mathematical and baseline gaps are constructive areas of improvement that do not undermine the core scientific contributions. I highly recommend this paper for publication.
