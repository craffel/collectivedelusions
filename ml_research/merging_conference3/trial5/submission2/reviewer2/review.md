# Peer Review: Rademacher-Bounded Polynomial Merging

## Summary of the Paper
The paper addresses the overparameterization and transductive overfitting challenges of adaptive weight-space model merging under extreme data scarcity (e.g., $M = 10$ labeled samples per task). The authors propose **Rademacher-Bounded Polynomial Merging (RBPM)**, which projects the high-dimensional layer-wise ensembling coefficients onto a low-degree continuous polynomial trajectory across normalized network depth. 

The paper derives empirical Rademacher complexity bounds for both the 1D trajectory space and the merged deep network classifier (the latter via first-order functional linearization), proving that restricting coefficients to follow a degree-$d$ trajectory reduces functional capacity by a factor of $\mathcal{O}(\sqrt{L / \log(d)})$. The authors introduce a specialized **Consensus-Pulling Rademacher Penalty** to pull ensembling coefficients back toward the stable uniform ensembling consensus baseline, preventing representation and scale distortion. 

The proposed framework is evaluated on a 12-layer deep CNN across 4 visual classification tasks (MNIST, FashionMNIST, CIFAR-10, SVHN) and physically validated on CLIP ViT-B/16 across fine-grained classification tasks (Stanford Cars, Oxford Flowers).

---

## Strengths and Weaknesses

### Strengths
1. **Mathematical Rigor:** The paper successfully introduces formal statistical learning-theoretic capacity-control measures to weight-space model merging. Deriving tight empirical Rademacher complexity bounds for the trajectory space over network layers represents a very creative and valuable theoretical framework.
2. **Strict Smoothness Guarantees:** Applying Markov's Theorem for Polynomials combined with the chain rule on the logistic sigmoid parameterization to establish a strict Lipschitz continuous derivative bound (Equation 3.10) is a mathematically elegant solution to prevent high-frequency layer-wise oscillations.
3. **Rigorous Scientific Controls:** The paper includes excellent scientific controls, such as decoupling the geometric constraint of the polynomial trajectory from the capacity-limiting effect of norm-bounding (Consensus-Pulling) in Section 4.3.8. This elegantly demonstrates that both forces are essential for optimal generalization.
4. **Physical Scale Validation:** Evaluated on a real-world foundation model (CLIP ViT-B/16) using fine-grained tasks. The results show that RBPM successfully retains over 98.6% of the individual expert performance ceiling while outperforming coordinate-wise pruning baselines.
5. **Presentation and Structural Integrity:** The paper is exceptionally well-structured, clearly written, and mathematically complete. The appendix contains detailed step-by-step proofs and a sophisticated discussion of piecewise splines, local Rademacher complexity, and Neural ODEs.

### Weaknesses
1. **The Illusion of Average Accuracy (Severe Task Dominance):** In Table 1, the authors report a robust average accuracy of 38.85% for RBPM vs. 29.05% for Static Uniform, claiming "superior generalization." However, a close inspection of the individual columns reveals that **this +9.80% average gain is driven entirely by a single task (MNIST)**, where RBPM achieves 75.20% (+44.20% over Uniform). On the remaining three tasks, **RBPM actually degrades performance**:
   - FashionMNIST: 48.60% (vs. Uniform's 50.60%, **-2.00%**)
   - CIFAR-10: 17.20% (vs. Uniform's 19.60%, **-2.40%**)
   - SVHN: 14.40% (vs. Uniform's 15.00%, **-0.60%**)
   
   This indicates that the joint few-shot optimization is heavily dominated by MNIST's steep, clean gradients, causing the optimizer to overfit the ensembling trajectory to MNIST at the direct expense of the other three domains. While the authors propose integrating PCGrad to resolve this, the average accuracy drops to 35.70%, and the performance on CIFAR-10 and SVHN remains extremely poor (barely above random guessing and far below individual expert ceilings). This severe limitation of weight-space merging on heterogeneous domains should be discussed transparently, and the "superior average performance" claim should be tempered.
2. **Unexplained Omission of the CUB-200 Dataset:** In Section 7.1 ("Experimental Protocol for Scalability on Vision Transformers"), the authors define their ViT benchmark using three datasets: Stanford Cars, Oxford Flowers, and **CUB-200-2011 (Birds)**. However, Table 2 only presents results for Stanford Cars and Oxford Flowers. **The results for CUB-200-2011 are completely missing from the empirical section of the paper.** Selective reporting of datasets must be avoided, and the omission must be justified or resolved.
3. **Unfair "Apples-to-Oranges" Comparison:** The paper compares RBPM directly against zero-shot weight-space merging heuristics (Static Uniform, TIES-Merging, DARE-Merging, and Sparse Task Arithmetic). TIES, DARE, and Sparse Task Arithmetic are completely **zero-shot and unsupervised**, requiring zero labeled data and zero optimization. RBPM, however, utilizes a labeled calibration dataset of $M = 10$ samples per task and performs supervised Adam optimization. Comparing a supervised, data-dependent adaptation method to zero-shot data-free heuristics is unfair and overstates the benefits of the method. The paper should clearly partition the baselines and acknowledge RBPM's supervised information advantage.
4. **Overstatement of the "Provable Generalization Guarantee":** The paper claims to establish "provable out-of-distribution generalization guarantees" for the deep merged network. However, the direct scaling link to the polynomial degree $d$ in Equation 3.14 relies on a **first-order functional linearization** around $W_0$. In deep multi-layer neural networks, layer-to-layer representation interactions are highly non-linear, and the first-order Taylor expansion completely ignores the higher-order cross-layer interaction terms that drive representational interference. Thus, the bound in Equation 3.14 is an idealized linear proxy and does not constitute a true guarantee for the actual non-linear network.
5. **Shaky "Layer as a Sample" Abstraction in Theorem 3.1:** Theorem 3.1 derives the empirical Rademacher complexity of the trajectory space $\mathcal{H}_d$ over a sample of size $L$ (network layers). However, network layers are deterministic, highly ordered feedforward blocks. Treating layers as independent coordinates is a highly questionable mathematical abstraction because they are not sampled i.i.d. from a probability distribution.

---

## Detailed Ratings

### 1. Soundness: Fair
The mathematical derivations in the appendix are correct and highly thorough, and the consensus-pulling penalty is a clever formulation. However, the theoretical framework relies on highly questionable assumptions (e.g., treating layers as independent i.i.d. sample coordinates in Theorem 3.1 and using a first-order functional linearization that ignores non-linear cross-layer interactions). On the empirical side, the "superior average performance" is a complete illusion driven entirely by MNIST dominance (with performance degradation on the other 3 tasks), the CUB-200 dataset is omitted without explanation, and comparisons to zero-shot baselines are structurally unfair.

### 2. Presentation: Good
The paper is exceptionally well-structured and written with high professional clarity. The mathematical notation is rigorous and consistent. However, the presentation would be significantly improved by tempering the claims of "superior generalization" on the CNN benchmark to reflect the individual task degradations, and explicitly distinguishing between supervised calibration and unsupervised zero-shot ensembling.

### 3. Significance: Good
The theoretical significance is high: this is the first work to apply Rademacher complexity bounds to weight-space model merging, establishing a highly promising capacity-controlled merging paradigm. However, the practical significance is currently moderate. The physical gain of RBPM over completely unsupervised online adaptation (Online PolyMerge) on ViTs is marginal (+0.85% absolute), and it requires labeled calibration data. Additionally, on highly heterogeneous domains, adaptive merging fails to find a viable multi-task model (yielding random-guess performance on CIFAR-10 and SVHN).

### 4. Originality: Fair
The core mechanism—restricting layer-wise ensembling coefficients to follow low-degree polynomial trajectories—is directly borrowed from **PolyMerge (Croft & Vance, 2024)**. The calibration setting is borrowed from Vance et al. (2025). The spectrally-normalized Rademacher complexity bounds are standard applications of Bartlett et al. (2017). The paper's originality lies not in the individual components, but in their cohesive synthesis and the formal theoretical justification.

---

## Overall Recommendation: 3: Weak Reject

### Justification
The paper has clear merits, including outstanding mathematical clarity, elegant proofs, and a robust theoretical formulation that bridges learning theory and model merging. However, the empirical and methodological weaknesses currently outweigh these merits:
1. **Omission of CUB-200:** The unexplained omission of the CUB-200-2011 dataset (which was defined as a target benchmark) is a major experimental gap.
2. **MNIST Dominance Illusion:** The average accuracy gain on the CNN benchmark is a complete illusion driven by MNIST task dominance, while performance degrades on 3 out of 4 tasks. This severe representational limitation must be discussed transparently.
3. **Unfair Baselines:** The comparison between a supervised, few-shot calibrated method and zero-shot, data-free heuristics is unfair and must be properly qualified.
4. **Idealized Assumptions:** The paper overstates its non-linear generalization guarantees, which rely on first-order Taylor linearizations that ignore critical cross-layer interactions.

If the authors can include the CUB-200-2011 empirical results, transparently discuss task dominance and individual task performance degradation as a limitation, and qualify their theoretical linearization assumptions, I would be highly enthusiastic about recommending this paper for acceptance. In its current state, however, the paper requires revisions.
