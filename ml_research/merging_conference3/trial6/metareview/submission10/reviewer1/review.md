# Peer Review

## Summary of the Paper
The paper addresses the low-data calibration challenge of dynamic model-merging routing heads. In this regime (e.g., 16 samples per task), standard unregularized dynamic routers (like the Softmax-based BL-Router) are prone to catastrophic overfitting and representational collapse. 

To mitigate this, the paper proposes:
1. **Bounded Sigmoidal Router (BSigmoid-Router):** An independent, Softmax-free sigmoidal dynamic router that decouples activation pathways to avoid the zero-sum competitive constraint of Softmax.
2. **Task-Correlation Prior Regularization (TCPR):** A static prior regularization technique that centers pre-computed cross-task similarity matrices (either in parameter-space or representation-space) and regularizes the routing signature weights via cosine similarity.

The paper evaluates these ideas on a heterogeneous four-task image classification benchmark using a Vision Transformer backbone ($\mathtt{vit\_tiny\_patch16\_224}$). 

---

## Overall Recommendation

**Score: 2 (Reject)**

### Justification:
While the paper is well-written and contains an exceptionally honest post-mortem analysis of the proposed regularizer, the core contribution of the paper—**Task-Correlation Prior Regularization (TCPR)**—is mathematically and empirically shown to be a **failure**. 

The submission suffers from a fatal self-contradiction: the Abstract, Introduction, and Contributions sections extensively pitch TCPR as a highly effective, successful, and robust regularizer that prevents collapse. Yet, the Experiments and Conclusion sections openly demonstrate that TCPR fails to outperform an unregularized sigmoidal router at any scale. When active ($\beta \ge 1.0$), TCPR actively collapses performance due to the "Alignment-Interference Paradox." When inactive ($\beta = 10^{-6}$), it is "mathematically dead" and behaves identically to the baseline. 

Furthermore, the experimental setting is highly fragile: the "specialist experts" are severely under-trained (SVHN expert achieves a mere 23.20% accuracy), the calibration is done on a tiny 64-image set evaluated under a single random seed without any statistical significance tests, and the mathematical assumptions underlying the regularizer are flawed. The paper cannot be accepted in its current form. It requires a complete structural overhaul to reframe the narrative around its negative results and a major expansion of the experimental suite to include converged experts, statistical significance testing, and modern model-merging benchmarks.

---

## Strengths and Weaknesses

### Soundness: Poor
- **The Central Self-Contradiction:** There is a severe disconnect between the paper's claims and its empirical results. The Abstract claims that *"TCPR consistently prevents high-conflict task collapse... [and] provides a robust, scale-invariant pathway."* However, Section 4.4 and Section 5 state that *"the proposed static prior regularization fails to deliver empirical improvements over unregularized sigmoidal routing, and actively degrades performance at larger scales."* A paper cannot pitch a method as a success when its own results demonstrate it is ineffective and harmful.
- **Severely Under-Trained Experts:** The specialists are trained for only 2 epochs on 1000 images, resulting in accuracies of 73.20% on MNIST and a highly sub-optimal 23.20% on SVHN. Because these experts are barely better than random guessing and are filled with parameter noise, the dynamic router is essentially learning to merge noise. Model merging is designed to aggregate specialized capabilities from strong, converged models. Findings obtained from such weak experts are unlikely to generalize to realistic applications.
- **Flawed Cosine Weight Regularization:** Equation 16 regularizes the cosine similarity of the raw projection weights ($\mathbf{w}_i, \mathbf{w}_j$) of the routing head. This assumes that the intermediate representations $z(x)$ are uniformly distributed on a unit sphere. In reality, neural activations lie in highly structured, narrow subspaces. Forcing weight vectors to be orthogonal or aligned without considering the covariance structure of the feature space is theoretically unsound and explains why performance collapses when the regularizer becomes active.
- **Arbitrary Prior Centering:** Centering the similarity matrix by subtracting the off-diagonal mean (Eq 13) is a highly arbitrary heuristic. It forces arbitrary positive or negative correlations depending purely on the chosen set of tasks, making the regularization unprincipled and highly unstable.

### Presentation: Fair
- **Writing Quality:** The paper is exceptionally well-written. The mathematical notation is precise, the vocabulary is professional, and the structure is clean.
- **Narrative Mismatch:** Despite the excellent prose, the presentation is rated as "Fair" due to the misleading and contradictory narrative structure. Proposing a method as the primary novel contribution in the title, abstract, and introduction, only to spend the results section proving that it doesn't work, is highly confusing for the reader.

### Significance: Poor
- **Lacks Utility:** Because the proposed TCPR regularizer does not work, it has zero practical utility for practitioners.
- **Limited Scope:** The experiments are restricted to a tiny 5.7M parameter ViT on four toy image datasets (MNIST, FashionMNIST, CIFAR, SVHN). This is extremely far removed from modern model-merging applications where dynamic routing is practically relevant, such as combining LLMs or Vision-Language Models.

### Originality: Fair
- **BSigmoid-Router:** Replacing Softmax with independent Sigmoids is a logical architectural change in Mixture-of-Experts or dynamic routing to eliminate the zero-sum competitive constraint. While useful, it is highly incremental.
- **TCPR:** The mathematical formulation of the regularizer is technically novel, but its originality is undermined by its lack of empirical value.

---

## Detailed Evaluation of Ratings

- **Soundness:** Poor
- **Presentation:** Fair
- **Significance:** Poor
- **Originality:** Fair

---

## Questions and Actionable Feedback for the Authors

1. **Narrative Overhaul (Crucial):**
   The authors must completely rewrite the paper to align the introduction and abstract with the empirical results. Instead of proposing TCPR as a "successful method," the paper should be reframed as an empirical study on why static prior regularizations fail in dynamic calibration. The title should be changed to reflect this negative-result analysis (e.g., *"An Empirical Study of Dynamic Router Calibration: The Power of Sigmoidal Routing and the Failure of Static Prior Regularization"*). The analytical deconstruction in Section 4.4 is excellent and should be the central contribution of the paper.

2. **Run Robust Multi-Seed Evaluations:**
   Calibrating a router on a tiny set of 64 images makes the optimization extremely sensitive to random variance. The authors **must** report average accuracy and standard deviations across at least 5 to 10 random seeds (using different random calibration splits and weight initializations). Reporting results for a single seed ($\mathtt{seed=42}$) is statistically insufficient and invalidates the scientific conclusions.

3. **Train Experts to Proper Convergence:**
   Please train the specialist experts (especially SVHN and MNIST) to reasonable convergence (e.g., >95% for MNIST, >85% for SVHN). This will ensure that model merging is evaluated on actual specialized models, rather than on parameter noise. 

4. **Address the Covariance of Representation Space:**
   In your future work or revision, how do you plan to account for the covariance of intermediate representations $z(x)$ in your signature regularization? Forcing $\cos(\mathbf{w}_i, \mathbf{w}_j)$ to align with task similarities ignores the distribution of the activations. A principled regularizer must project the task similarities into the active activation subspace.

5. **Scale to Modern Benchmarks:**
   To make this paper truly impactful, please evaluate the unregularized `BSigmoid-Router` and the failure of static priors on modern, larger-scale model-merging settings. For example, merge CLIP models fine-tuned on diverse downstream domains, or evaluate on a multi-task instruction-following suite using LLMs.
