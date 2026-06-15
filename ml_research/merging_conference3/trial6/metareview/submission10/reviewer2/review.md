# Peer Review

## Summary of the Paper
The paper addresses the challenge of calibrating dynamic routing heads in low-data regimes for multi-task model merging. It identifies that standard Softmax-based routing heads suffer from overfitting and representational collapse, especially on high-conflict datasets. To resolve this, the authors introduce two main components:
1. **Bounded Sigmoidal Router (BSigmoid-Router):** A decoupled, Softmax-free routing head that uses independent sigmoid functions for each task, scaling by a ceiling ($\lambda_{\text{max}} = 0.3$). This eliminates the zero-sum competitive constraint of Softmax.
2. **Task-Correlation Prior Regularization (TCPR):** A regularization method that uses pre-computed cross-task similarity priors (parameter-space or representation-space) to guide the calibration of the routing signatures, employing centering and unit sphere normalization.

Through evaluations on a four-task Vision Transformer benchmark (MNIST, FashionMNIST, CIFAR-10, SVHN) calibrated on only 16 samples per task, the paper demonstrates that the simple, unregularized **BSigmoid-Router** achieves state-of-the-art performance (**25.50%** joint mean accuracy), outperforming complex wave-interference methods like QWS-Merge (**21.80%**). Crucially, the authors' own empirical sweeps reveal that the proposed **TCPR** regularization fails to improve upon the unregularized sigmoidal baseline, acting as a "dead" regularization at small scales and actively degrading performance at active scales due to scale mismatch and representational interference (the alignment-interference paradox).

---

## Strengths and Weaknesses

### Strengths
1. **Elegant and Simple Routing Solution:** The proposal of the **BSigmoid-Router** is a beautifully simple and highly effective solution. By replacing the standard competitive Softmax with decoupled, independent sigmoid functions, the paper completely resolves the zero-sum competitive bottleneck of multi-task routing. This simple change yields a massive performance boost (from 19.10% to 25.50%) and handily outperforms highly complex, physics-inspired wave-interference models like QWS-Merge (21.80%). It elegantly demonstrates that complex problems can be solved with highly simple and direct mechanisms.
2. **Exhaustive Baseline Evaluation:** The paper evaluates against a comprehensive set of seven baselines, including static weight interpolation (Task Arithmetic), classical linear routers, Softmax-based routers (with and without L2 regularization), and a complex wave-interference state-of-the-art model (QWS-Merge).
3. **Rigorous and Honest Empirical Inquest:** Sections 4.4 and 4.5 exhibit an extraordinary level of scientific integrity and depth. Instead of hiding the failure of their proposed TCPR regularizer, the authors provide a thorough, mathematically grounded deconstruction of why it fails (analyzing scale mismatch, the alignment-interference paradox, and static-dynamic conflicts). This honest analysis is extremely valuable to the model-merging community, providing a clear warning against ineffective, static prior regularizations.

### Weaknesses
1. **Critical Narrative Contradiction:** There is a severe, unacceptable contradiction between the framing of the paper (Title, Abstract, Introduction, and Contributions) and the actual empirical findings in Section 4.4 and the conclusion. 
   - The Abstract claims: *"we demonstrate that TCPR consistently prevents high-conflict task collapse and bridges the performance gap to specialist experts... confirm that incorporating task-relatedness priors provides a robust, scale-invariant pathway..."*
   - Section 1 claims: *"TCPR penalizes diverging routing signatures... TCPR consistently outperforms standard L2-regularized baselines and SOTA wave-interference methods..."*
   - However, Section 4.4 explicitly states: *"the proposed static prior regularization fails to deliver empirical improvements over unregularized sigmoidal routing, and actively degrades performance at larger scales."*
   - Indeed, Table 1 shows TCPR-Param/Rep achieving **25.20%** joint mean accuracy, which is identical to BSigmoid-Router (Reg) (**25.20%**) and slightly *worse* than the unregularized BSigmoid-Router (**25.50%**).
   - The paper cannot be accepted in its current form with such a misleading framing. It must be completely reframed to focus on the elegant and simple BSigmoid-Router as the primary contribution, with TCPR repositioned as an instructive case study deconstructing the pitfalls of static prior regularizations.
2. **Single-Seed Evaluation in Volatile Regimes:** The main results in Table 1 are reported for a single seed (`seed=42`). In low-data calibration regimes (16 samples per task, 64 total) and under-trained experts, optimization trajectories are highly volatile. Without multi-seed statistics (reporting mean and standard deviation across at least 5-10 runs), the reported marginal performance differences (e.g., 25.50% for BSigmoid-Router vs 25.20% for TCPR) may be statistically insignificant and driven by random noise.
3. **Limitation to Under-trained Experts:** The experts used are extremely sub-optimal (MNIST 73.20%, SVHN 23.20%, Joint Mean 62.40%). While the authors justify this as simulating edge-AI constraints and representational noise, evaluating *only* sub-optimal experts is a major limitation. It is highly possible that the "Alignment-Interference Paradox" and the failure of TCPR are artifacts of the extreme parameter noise in under-trained models, rather than an intrinsic flaw of prior regularization in well-trained, converged regimes. The authors should evaluate their methods on fully-converged experts as well.
4. **Lack of Ceiling Sensitivity Analysis:** The scale ceiling $\lambda_{\text{max}} = 0.3$ is fixed. The paper lacks any ablation study or sensitivity analysis to show how the BSigmoid-Router behaves under other ceiling values.

---

## Detailed Evaluation Ratings

### Soundness: Fair
The paper introduces a highly sound and appropriate method in the BSigmoid-Router, and its deconstruction of TCPR is mathematically and empirically rigorous. However, the evaluation is limited by reporting only a single seed in a highly volatile low-data regime and using exclusively under-trained, noisy experts. These limitations make it unclear whether the empirical conclusions hold under statistical replication or on standard converged models.

### Presentation: Fair
While the paper is written with high clarity and mathematical rigor, its overall presentation is severely flawed due to the critical narrative contradiction. Framing the entire paper and title around a regularizer (TCPR) that is empirically disproved in the experiments and shown to be unnecessary/harmful is highly confusing and scientifically misleading. The paper requires a complete structural reframing to resolve this contradiction.

### Significance: Good
The paper's significance is high. By demonstrating that a very simple, decoupled sigmoidal routing head completely eliminates the Softmax competitive bottleneck and outperforms highly complex physics-inspired wave-interference models, it champions simplicity and elegance. This demystifies the field and provides a clear warning against unnecessary complexity in dynamic model merging.

### Originality: Good
The application of independent sigmoidal routing to resolve the multi-task zero-sum competitive bottleneck of model merging is a neat, simple, and original concept. Furthermore, the rigorous "negative results" deconstruction of static prior regularization is highly original and scientifically valuable.

---

## Overall Recommendation

**Rating: 3: Weak Reject**

### Justification
The paper has clear merits, particularly the proposal of the elegant and simple **BSigmoid-Router**, which achieves superior joint performance over complex state-of-the-art wave-inspired methods. However, the severe narrative contradiction—where the front matter (Title, Abstract, Intro) frames TCPR as a highly successful regularizer while the experimental results and back matter prove it fails and actively degrades performance—precludes acceptance in its current form. 

To be considered for publication, the authors must perform a major revision addressing the following:
1. **Complete Reframing:** Re-title and reframe the paper to focus on the simplicity, elegance, and success of the unregularized **BSigmoid-Router**. TCPR must be repositioned not as a successful regularizer, but as an instructive deconstruction showing that static, pre-computed prior regularizations introduce representational noise and fail to improve upon simple, unconstrained routing.
2. **Multi-Seed Evaluation:** Provide mean and standard deviation across multiple seeds (at least 5-10 runs) for the results in Table 1 to establish statistical significance.
3. **Converged Experts Baseline:** Evaluate the router on fully converged, standard experts to prove that the findings (including the failure of prior regularization) generalize beyond sub-optimal, noisy regimes.
4. **Ceiling Sensitivity:** Provide an ablation study for the scale ceiling parameter $\lambda_{\text{max}}$.
