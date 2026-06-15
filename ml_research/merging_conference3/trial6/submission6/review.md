# Peer Review: PAC-Bayes Merge

**Title:** PAC-Bayes Merge: Information-Theoretic PAC-Bayesian Bounds for Trajectory-Constrained Model Merging  
**Recommendation:** 4: Weak Accept (Technically solid paper that advances at least one sub-area of AI, with a contribution that others are likely to build on, but with some weaknesses that limit its immediate empirical impact, specifically its reliance on a synthetic representation sandbox).

---

## 1. Summary of the Paper

This paper presents **PAC-Bayes Merge**, a mathematically grounded framework that provides the first formal, information-theoretic foundation for trajectory-regularized model merging. 

In post-hoc model merging, task-specific expert neural networks (fine-tuned from a shared pre-trained base model) are combined directly in parameter space without joint multi-task training. While simple static uniform ensembling ($\alpha_k = 1/K$) or global task arithmetic coefficients serve as standard baselines, they fail to handle the significant layer-wise heterogeneity of deep neural networks. To address this, recent methods optimize layer-wise merging coefficients $\alpha_k(l)$ on a small calibration dataset $\mathcal{D}_{\text{cal}}$. However, this high-dimensional search space (e.g., $K \times L = 56$ parameters) optimized on extremely scarce calibration data (e.g., $M = 10$ samples per task) leads to severe transductive overfitting (which the authors term the *Overfitting-Optimizer Paradox*). Under unregularized optimization, layer-wise coefficients oscillate chaotically across adjacent layers, collapsing out-of-distribution generalization.

To resolve this, PAC-Bayes Merge integrates:
1. **Polynomial Trajectory Parameterization:** Restricting layer-wise ensembling coefficients to follow a continuous, low-degree polynomial trajectory across network depth ($z = \frac{l}{L-1} \in [0, 1]$), which acts as a depth-wise low-pass filter.
2. **PAC-Bayesian Learning-Theoretic Foundation:** Modeling trajectory coordinates as a randomized isotropic Gaussian posterior $Q \sim \mathcal{N}(\Theta, \sigma^2 I)$ centered at learnable coordinates $\Theta$, and specifying a spherical Gaussian prior $P \sim \mathcal{N}(\Theta_{\text{uniform}}, \sigma_0^2 I)$ centered at the scale-preserving uniform ensembling consensus $\Theta_{\text{uniform}}$. The authors prove that minimizing McAllester's PAC-Bayesian bound analytically justifies a smooth quadratic $L_2$ Consensus-Pulling penalty centered at the stable uniform consensus.
3. **Continuous Capacity Preservation:** Explaining how the quadratic $L_2$ penalty softly pulls parameters towards the consensus basin to preserve continuous representative capacity down intermediate layers, whereas heuristic $L_1$ penalties (e.g., in RBPM) corresponding to a Laplace-like prior induce coordinate sparsity and flatten trajectories.
4. **SWA Equivalence:** Establishing a formal link showing that uniform weight merging acts as a parametric low-pass filter that reduces the variance of SGD optimization noise by a factor of $K$.
5. **Monte Carlo Optimization & Evaluation:** Implementing randomized training (sampling trajectory coordinates during optimization) and test-time posterior ensemble evaluation (averaging predicted softmax probabilities across coordinates drawn from the posterior).

Evaluating on a customized 14-layer deep residual MLP multi-task representation sandbox (MNIST, FashionMNIST, CIFAR-10, SVHN modeled as 192-dimensional Gaussian vectors with correlation $\rho = 0.5$) under extreme calibration scarcity ($M = 10$ samples per task), the paper demonstrates that PAC-Bayes Merge achieves a Joint Mean accuracy of **47.77%** (Randomized Ensemble) and **47.74%** (Deterministic Compiled), outperforming Static Uniform (42.03%), properly tuned Ties-Merge (43.08%), DARE-Merge (41.91%), Offline Unconstrained Few-Shot Tuning (47.65%), and $L_1$-regularized trajectory merging (47.40%).

---

## 2. Main Strengths

* **Rigorous Information-Theoretic Foundation:** This work successfully bridges the gap between post-hoc weight-space model merging and statistical learning theory. Rather than proposing another empirical or intuitive heuristic, the authors provide a watertight PAC-Bayesian bound that mathematically derives a standard $L_2$ Consensus-Pulling penalty.
* **Outstanding Quality of Writing & Academic Rigor:** The paper is exceptionally well-written, clearly structured, and easy to follow. Each section is introduced with a logical roadmap. The authors demonstrate high intellectual honesty, especially in Section 3.4, where they present Theorem 3.1 (linking uniform weight merging to Stochastic Weight Averaging) and immediately critique its unrealistic single-basin modeling assumption under disparate task-specific settings. This level of self-reflection is refreshing and adds significant credibility to the work.
* **Practical Deployment Guidelines:** The comparison between the "Deterministic Compiled" model (deploying a single model statically compiled at the posterior mean $\Theta^*$ at test-time) and the "Randomized Ensemble" model (evaluating a 10-coordinate ensemble) yields a highly valuable practical insight. The compiled model performs within 0.03% of the randomized ensemble while completely avoiding the 10$\times$ forward-pass latency overhead. This demonstrates that randomized MC sampling can be used purely during optimization (to smooth the loss landscape) without sacrificing zero-latency deployment benefits.
* **Exhaustive Experimental Control and Ablations:** Despite the synthetic nature of the sandbox, the experiments are methodologically complete, featuring 5-seed evaluations, a paired two-tailed t-test (proving the statistical significance of the $L_2$ regularizer over $L_1$ with $p \approx 0.04$), and thorough ablation studies over the regularization weight $\lambda_{\text{PAC}}$ and the posterior noise standard deviation $\sigma$.

---

## 3. Main Weaknesses / Areas for Improvement

* **Weakness 1: Transition from McAllester's Bound to Quadratic $L_2$ Regularizer (Lagrangian Relaxation):**
  In Section 3.2, the authors apply McAllester's classical bound (Equation 10), which features the KL divergence inside a square root: $\sqrt{\frac{D_{\text{KL}}(Q \parallel P) + \ln(2\sqrt{N_{\text{img}}}/\delta)}{2 N_{\text{img}}}}$. 
  The authors minimize a linear surrogate $\mathcal{L}_{\text{ce}}(\Theta) + \lambda_{\text{PAC}} \mathcal{R}_{\text{PAC}}(\Theta)$, where $\mathcal{R}_{\text{PAC}}(\Theta) = \|\Theta - \Theta_{\text{uniform}}\|_2^2$ (Equation 14).
  Since the KL divergence itself is quadratic in $\Theta$, the complexity term in McAllester's bound is actually linear in $\|\Theta - \Theta_{\text{uniform}}\|_2$ (due to the square root). Minimizing a quadratic $L_2$ regularizer means we are optimizing a Lagrangian relaxation that penalizes the *square* of the complexity term (which corresponds to Catoni's or Alquier's bound rather than McAllester's directly). The authors should clarify this distinction, acknowledging that Equation (14) represents a Lagrangian relaxation of the complexity term rather than a direct linear surrogate of McAllester's square-root bound, and clarify the exact mathematical steps that yield the theoretical value of $\lambda_{\text{PAC}}$.
* **Weakness 2: Discrepancy in Sample Size Notation and Numerical Vacuosity Claims:**
  In Section 3.2, the authors define the multi-task calibration dataset $\mathcal{D}_{\text{cal}} = \{(x_i^{(k)}, y_i^{(k)})\}_{i=1}^{N_{\text{img}}}$ as containing $N_{\text{img}}$ samples. However, in the double-summation for $\mathcal{L}_{\text{ce}}(\Theta)$ (Equation 15), the index $i$ goes from $1$ to $N_{\text{img}}$ while $k$ goes from $0$ to $K-1$, implying there are $K \times N_{\text{img}}$ total samples. But in Section 4.1, the calibration set size is denoted as $M = 10$ samples per task (total 40 samples). This notation is inconsistent: is $N_{\text{img}}$ the total size of the calibration set (40) or the per-task size ($M = 10$)? If it is the per-task size, the definition of $\mathcal{D}_{\text{cal}}$ as having $N_{\text{img}}$ samples is incorrect.
  Furthermore, the paper claims to guarantee "non-vacuous control over out-of-distribution generalization under extreme data scarcity." However, under extreme scarcity ($M = 10$ per task, total $N = 40$), a quick calculation shows that the generalization gap term $\sqrt{D_{\text{KL}}/2N}$ is likely around $0.58$, making the upper bound on the true error rate exceed $1.0$ (since the empirical error is already $\approx 0.53$). A bound on an error rate that exceeds $1.0$ is numerically vacuous. The authors should tone down this claim: while the bound provides a highly principled *qualitative regularizer*, its finite-sample numerical value under extreme scarcity is vacuous, which is a common limitation of PAC-Bayesian deep learning bounds.
* **Weakness 3: Reliance on Simulated Data (Primary Empirical Limitation):** 
  The primary limitation affecting the significance of this paper is that the empirical evaluation is conducted inside a customized "representation sandbox." Here, image classification tasks are simulated using 192-dimensional Gaussian vectors centered around simulated task prototypes, rather than using physical datasets (e.g., actual MNIST/CIFAR-10 raw pixel arrays) and standard vision backbones (e.g., CNNs like ResNet-18 or Vision Transformers). While this simulated setup is highly rigorous and allows perfect control over variables (bypassing weight-space permutation mismatch and isolating coordinate conflict), evaluating on realistic, non-synthetic datasets and architectures is critical to establish broad empirical impact in the computer vision and deep learning communities.

---

## 4. Rating Breakdown

* **Soundness:** **Good.** The mathematical derivations are exceptionally elegant and cleanly executed, providing a highly principled foundation. However, the theoretical presentation has some minor notation discrepancies and overclaims regarding bound non-vacuosity that need to be resolved.
* **Presentation:** **Excellent.** The paper represents an extremely high standard of academic writing. The narrative is cohesive, the figures/tables are highly informative, and the roadmaps make it easy to read.
* **Significance:** **Good.** The paper successfully grounds heuristic parameter-space regularizers in formal statistical learning theory, which is a major step forward for model-merging literature. However, the scope of impact is currently constrained by the reliance on a synthetic representation sandbox.
* **Originality:** **Good.** While trajectory parameterization and consensus-pulling were previously introduced in literature (e.g., RBPM), this paper provides the first formal PAC-Bayesian derivation, the theoretical transition from $L_1$ to $L_2$ regularization, the randomized MC optimization, and the SWA linkage.

---

## 5. Detailed Feedback & Constructive Suggestions

1. **Address the Lagrangian Relaxation of the Bound:** 
   In Section 3.2, explain that minimizing $\|\Theta - \Theta_{\text{uniform}}\|_2^2$ is a Lagrangian relaxation of McAllester's bound (which contains the square root of the KL divergence). If a direct linear bound is desired, point out that Catoni's or Alquier's bound can be used to directly justify a linear KL penalty.
2. **Resolve the Sample Size Notation Discrepancy:**
   In Section 3.2, polish the notation to consistently use $M$ (or $N_{\text{img}}$) for the per-task sample size and $N_{\text{total}} = K \times M$ for the total sample size. Ensure that the double-summation in Equation (15) matches the definition of the calibration set $\mathcal{D}_{\text{cal}}$.
3. **Qualify the "Non-Vacuous" Generalization Control Claim:**
   In Section 1 and Section 3.2, clarify that while the PAC-Bayesian bound provides a mathematically non-vacuous qualitative regularizer (ensuring the complexity penalty is strictly bounded and theoretically correct), the actual numerical value of the bound under extreme few-shot scarcity ($N=40$) is likely vacuous (exceeding 1.0 for the error rate), which is a standard property of deep learning generalization bounds.
4. **Discuss the Capacity of Low-Degree Polynomials:**
   In Section 3.1, add a brief theoretical note explaining that restricting the trajectory to a low-degree polynomial ($d \le 2$) serves to bound the hypothesis space capacity (reducing the dimension $K(d+1)$ of the parameter space). This mathematically reduces the KL divergence term $D_{\text{KL}}$ in Equation (13) and directly leads to a tighter (less vacuous) generalization bound, providing a formal learning-theoretic explanation for why low-degree polynomials are optimal.
5. **Analyze the Local Linearity of the Loss Landscape:**
   Given that the *Deterministic Compiled* model at the posterior mean performs almost identically to the *Randomized Ensemble* model (47.74% vs 47.77%), add a short discussion in Section 4.2.1 explaining that this suggests an approximate local linearity or flatness of the loss landscape in the neighborhood of the posterior mean $\Theta^*$, making the expectation of the classification risk close to the risk at the expectation: $\mathbb{E}_{Q}[R(G_{\tilde{\Theta}})] \approx R(G_{\mathbb{E}[\tilde{\Theta}]})$.
