# Peer Review for "PAC-Bayes Merge: Trajectory-Regularized Model Merging"

## Summary of the Paper
The paper addresses post-hoc weight-space model merging, aiming to resolve the **Overfitting-Optimizer Paradox** (or transductive overfitting trap) that arises when ensembling coefficients are dynamically optimized on extremely scarce calibration data (e.g., 10 samples per task). 

To resolve this, the authors introduce **PAC-Bayes Merge**, which:
1. **Parameterizes layer-wise merging coefficients** using a continuous, low-degree (e.g., cubic) polynomial trajectory of network depth, significantly reducing the dimensionality of the search space.
2. **Models ensembling parameters** as a randomized Gaussian posterior distribution ($Q$) and proves that minimizing Alquier’s linear PAC-Bayesian bound under a spherical Gaussian prior ($P$) centered at the uniform ensembling consensus analytically yields a quadratic $L_2$ Consensus-Pulling penalty.
3. **Presents a non-isotropic variant (PAC-Bayes-FIM)** that weights coordinate penalties using diagonal elements of the empirical Fisher Information Matrix (FIM) evaluated at the uniform consensus point.
4. **Bridges the theory-to-practice gap** by implementing Monte Carlo expected risk optimization during training and expected prediction averaging over posterior-sampled parameters at test time (Randomized Ensemble).

The method is evaluated on a synthetic "sandbox" consisting of four physical classification datasets (MNIST, FashionMNIST, CIFAR-10, SVHN) projected via Johnson-Lindenstrauss (JL) embeddings to 192 features and passed through a 14-layer deep residual MLP backbone. The results show that PAC-Bayes Merge achieves a Joint Mean accuracy of **35.37%** (Deterministic Compiled), outperforming Static Uniform (33.35%), Ties-Merge (29.59%), and DARE-Merge (32.76%), while yielding a microscopic **0.10%** absolute improvement over the $L_1$-regularized Rademacher-Bounded Polynomial Merging (RBPM) baseline (35.27%).

---

## Strengths and Weaknesses

### Strengths
1. **Mathematical Rigor & Completeness:** The paper is exceptionally well-grounded in statistical learning theory. Deriving a smooth $L_2$ Consensus-Pulling penalty directly from Alquier's linear PAC-Bayesian bound under Gaussian priors is a elegant, mathematically watertight theoretical contribution.
2. **Clear Narrative & Presentation:** The writing is clean, highly professional, and easy to follow. Complex concepts like Alquier's temperature trade-offs, KL divergence derivations, and Fisher Information mappings are articulated with high clarity.
3. **Detailed Scalability Appendix:** The authors provide a highly constructive "scaling blueprint" in Appendix A, detailing how to implement PAC-Bayes Merge on standard computer vision backbones (ResNets, ViTs) and autoregressive LLMs with LoRA merging. This greatly helps bridge the gap between their theoretical framework and actual real-world deployment.
4. **Exhaustive Statistical Control:** Repeating experiments across 15 independent random seeds and providing complete results (mean, standard deviation, and a full scarcity sweep) represents a high standard of empirical honesty and reproducibility.

### Weaknesses (Practitioner Perspective)
Despite its mathematical beauty, the paper suffers from severe practical limitations and artificial constraints that limit its real-world utility and adoption:

1. **Test-Time Latency and Memory Overhead (The Randomized Ensemble Dilemma):**
   The theoretical guarantees of the PAC-Bayesian framework strictly apply to the randomized classifier $G_Q$. At test-time, evaluating $G_Q$ requires drawing $S_{\text{test}} = 5$ independent ensembling trajectory coordinates from the posterior and running 5 forward passes (Randomized Ensemble). This incurs a **5x test-time computational latency and memory overhead**. 
   - Post-hoc model merging is highly valued by practitioners precisely because it offers **zero runtime latency and zero additional memory footprint** compared to traditional ensembling or dynamic routing. Introducing a 5x overhead completely destroys this primary practical benefit.
   - While the authors propose a "Deterministic Compiled" model (evaluating a single static model at the posterior mean $\Theta^*$) to preserve zero test-time latency, the PAC-Bayesian generalization bounds **do not hold** for this deterministic model due to the non-convexity of deep neural networks. Therefore, the rigorous "learning-theoretic guarantees" claimed by the authors are lost in the only deployment mode that a practitioner would actually use.

2. **Highly Synthetic and Impractical "Sandbox" Setup:**
   The experimental evaluation is conducted on a highly contrived toy setup: a **14-layer MLP with a width of only 64** processing **Johnson-Lindenstrauss (JL) projected 192-dimensional features** of MNIST, FashionMNIST, CIFAR-10, and SVHN. 
   - This setup does not reflect contemporary machine learning. Practitioners apply model merging to large-scale deep neural networks (e.g., Vision Transformers, ResNets, Large Language Models) operating on raw high-dimensional inputs.
   - Because of this synthetic setup, the absolute accuracies are exceptionally poor and practically unusable. A Joint Mean accuracy of **35.37%**—with **CIFAR-10 accuracy at 12.89%** and **SVHN at 15.71%** (barely above the 10% random guessing baseline)—is not a deployable or useful system. It is unclear if the proposed regularizer would translate to any meaningful performance gains in high-performing, real-world systems.

3. **Inappropriate Baseline Comparison (Representation Space Mismatch):**
   The paper compares its method to prominent LLM-centric baselines like **Ties-Merge** and **DARE-Merge**, showing they perform poorly. 
   - However, Ties-Merge and DARE-Merge assume that task experts share a highly aligned pre-trained weight space (e.g., fine-tuned from a shared LLM base on identical input spaces). 
   - In the authors' setup, they apply **different seed-specific random projection matrices** to the input images of each task. This means the input layer of each task-specific expert must map completely different manifold structures. Under such severe, artificial coordinate mismatches, Ties-Merge and DARE-Merge are mathematically bound to fail. This makes the baseline comparison highly contrived and uninformative.

4. **Practically Insignificant Performance Margin ($L_1$ vs. $L_2$):**
   The core claim that the smooth $L_2$ penalty is superior to the sparse $L_1$ penalty (RBPM) because it "preserves continuous representative capacity" is theoretically appealing but practically unsupported.
   - Isotropic PAC-Bayes Merge (Deterministic Compiled) achieves **35.37 $\pm$ 2.81%** Joint Mean accuracy, whereas RBPM ($L_1$) achieves **35.27 $\pm$ 2.72%**.
   - A microscopic **0.10%** absolute difference, well within the large cross-seed standard deviations of over 2.7%, is statistically and practically indistinguishable. For a practitioner, this negligible improvement does not justify the added complexity of posterior sampling and theoretical derivations.

5. **Failure of the Explicit Regularizer under Extreme Scarcity ($M=2$):**
   The core motivation of the paper is that unregularized optimization collapses under extreme calibration data scarcity, requiring PAC-Bayesian regularization.
   - However, the scarcity sweep in `scarcity_results.json` reveals that under extreme scarcity (**$M = 2$ samples per task**):
     - **Offline Unconstrained (unregularized)** achieves a Joint Mean of **34.16 $\pm$ 3.13%**.
     - **PAC-Bayes (isotropic)** achieves **33.86 $\pm$ 3.36%**.
     - **PAC-Bayes-FIM** achieves **33.43 $\pm$ 3.40%**.
   - Under the most severe data constraints ($M=2$), **the unregularized offline optimizer actually outperforms the proposed explicit regularizer** by **0.30%** (over isotropic) and **0.73%** (over FIM). This raises serious doubts about the absolute necessity of the complex explicit PAC-Bayesian penalty in severe few-shot regimes, and suggests that implicit optimization dynamics (e.g., early stopping in AdamW) are already sufficient or superior.
   - Furthermore, the FIM variant performs the worst at $M=2$, illustrating that estimating Fisher Information on extremely small budgets introduces severe estimation noise that corrupts the prior and degrades optimization.

---

## Detailed Ratings

### Soundness: Good
The paper is theoretically and mathematically flawless. The derivations of the $L_2$ penalty and non-isotropic FIM-weighted bounds from Alquier's linear bound are mathematically sound and watertight. However, its soundness rating is limited to "Good" rather than "Excellent" because:
- The SWA equivalence theorem (Theorem 3.1) relies on the highly unrealistic assumption that disparate task experts (MNIST vs. SVHN) reside in a single shared basin of attraction, which contradicts Linear Mode Connectivity (LMC) findings.
- The theoretical bounds strictly apply only to the randomized ensemble (which incurs a prohibitive 5x latency overhead at test-time), whereas the zero-latency "Deterministic Compiled" deployment mode lacks these guarantees.

### Presentation: Excellent
The overall narrative, organization, and visual layout of the paper are exceptional. All equations are clearly defined, and the theoretical derivations are highly readable. The inclusion of a comprehensive scaling blueprint (Appendix A) and detailed FIM derivations (Appendix B) represents an outstanding level of completeness and presentation quality.

### Significance: Fair
The significance of the work is primarily theoretical, offering an elegant template for applying PAC-Bayesian bounds to post-hoc model merging. However, for practical and industrial applications, the significance is "Fair" (low):
- The evaluation is conducted on a highly synthetic, low-performing toy setup (width-64 MLP with JL-projected features).
- The empirical margin over standard $L_1$ regularization (RBPM) is practically negligible (0.10%).
- Under extreme scarcity ($M=2$), unregularized optimization outperforms the proposed method, and the 5x test-time latency of the randomized ensemble is a dealbreaker for real-world deployment.

### Originality: Good
The paper provides a creative and technically rigorous combination of PAC-Bayesian generalization theory and trajectory-constrained parameter-space fusion. It successfully establishes a novel, formal connection between Alquier's bound and quadratic center-pulling penalties, representing a solid step forward from prior heuristic regularizers.

---

## Overall Recommendation: 3 (Weak Reject)
The paper is a strong theoretical piece with elegant, watertight mathematical derivations and outstanding presentation. However, its overall merits are heavily outweighed by its severe practical shortcomings:
- The evaluation is limited to a synthetic toy setup with unusable absolute accuracies (e.g., 12.89% on CIFAR-10), making it impossible to know if the method scales to real-world architectures and raw image/token streams.
- The 5x test-time latency overhead of the Randomized Ensemble violates the primary motivation of model merging, while the zero-latency Deterministic Compiled mode lacks the claimed PAC-Bayesian guarantees.
- The empirical performance delta over the simpler, heuristic $L_1$ baseline is a negligible 0.10%, and unregularized optimization actually outperforms the proposed regularizer under extreme scarcity ($M=2$).

Unless the authors can demonstrate their method's efficacy on a real-world, high-performing benchmark (e.g., merging LoRA adapters on a 7B LLM or merging fine-tuned ResNet-50/ViT-B backbones on physical image datasets) and show a substantial and statistically significant practical advantage over standard unregularized optimization, this work remains a purely academic exercise of limited practical utility. I recommend a weak reject to encourage the authors to ground their beautiful theory in realistic, physical deep learning evaluations.
