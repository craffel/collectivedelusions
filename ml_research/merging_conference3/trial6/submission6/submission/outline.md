# PAC-Bayes Merge: Paper Outline

## 1. Abstract
- **Context:** Post-hoc model merging has emerged as a lightweight, zero-retraining paradigm to integrate multiple task-specific expert models into a single multitask architecture.
- **Problem:** Optimizing merging coefficients on extremely small few-shot calibration datasets (e.g., $M = 10$ samples per task) leads to severe transductive overfitting, high-frequency inter-layer parameter oscillations, and degraded out-of-distribution generalization.
- **Proposed Solution:** PAC-Bayes Merge, the first formal, information-theoretic learning framework for trajectory-constrained model merging.
- **Mechanism:** We restrict ensembling parameters to a continuous, low-degree polynomial trajectory across network depth. We frame the trajectory parameters as the mean of a randomized Gaussian posterior and place a Gaussian prior centered at the stable, scale-preserving uniform consensus baseline $\Theta_{\text{uniform}}$.
- **Theoretical Insight:** Minimizing the PAC-Bayesian generalization bound mathematically derives a quadratic $L_2$ Consensus-Pulling penalty, which softly regularizes the trajectory coordinates without forcing coordinate sparsity.
- **Empirical Results:** Evaluated on genuine physical image datasets (MNIST, FashionMNIST, CIFAR-10, SVHN) projected via Johnson-Lindenstrauss embeddings into a 14-layer deep MLP residual backbone across 15 independent random seeds. PAC-Bayes Merge successfully suppresses transductive overfitting, outperforming Static Uniform, Ties-Merge, DARE-Merge, and Offline Unconstrained Tuning.

## 2. Introduction
- **Deep Learning Era:** The paradigm of pre-training and task-specific fine-tuning has led to highly specialized but disjoint models.
- **Model Merging:** Integrating these experts into a single multitask model without expensive retraining or catastrophic forgetting is a critical challenge.
- **Few-Shot Adaptation:** In resource-constrained settings, merging coefficients are tuned on tiny post-hoc calibration splits.
- **The Overfitting Trap:** Tuning unconstrained layer-wise parameters (e.g., 56 parameters for 14 layers across 4 tasks) on 40 samples results in severe transductive overfitting and chaotic coefficient oscillations across layers.
- **Our Contribution (PAC-Bayes Merge):
  - **Theoretical Grounding:** Replacing heuristic regularizers with a formal, information-theoretic PAC-Bayesian generalization guarantee.
  - **Quadratic $L_2$ Consensus-Pulling Penalty:** A mathematically derived regularizer that softly pulls ensembling trajectories back to the stable uniform consensus basin, preserving continuous representational capacity across layers.
  - **Sparsity vs. Capacity Trade-Off:** Proving why a smooth $L_2$ penalty performs comparably to or better than the Laplace-prior-derived $L_1$ penalty (RBPM) by avoiding artificial coordinate pruning and maintaining representation expressiveness.
  - **Empirical Sandbox Validation:** Demonstrating consistent generalization gains under extreme data scarcity and high inter-task noise.

## 3. Related Work
- **Post-Hoc Model Merging:** Reviewing task arithmetic, TIES-Merging, RegCalMerge, and PolyMerge. Highlight how existing methods rely on heuristic tuning or unconstrained optimisation.
- **Trajectory-Constrained Merging:** Discussing polynomial routing, RBPM, and dynamic wave superposition.
- **PAC-Bayesian Learning Theory:** Explaining how PAC-Bayes bounds have historically bounded generalization in neural network weights. Contrast this with our novel application directly to weight-space ensembling/merging coefficient trajectories.
- **Gap & Novelty:** Positioning PAC-Bayes Merge as the first to link information-theoretic PAC-Bayes bounds to weight-averaging trajectory regularization.

## 4. Methodology & Mathematical Formulation
- **Weight-Space Model Merging:** Formalizing task-expert weights $W_k^{(l)}$ and task vectors $V_k^{(l)} = W_k^{(l)} - W_0^{(l)}$.
- **Polynomial Trajectory Projection:** Mapping $K \times L$ parameters down to $K \times (d+1)$ by parameterizing the merging coefficients $\alpha_k(l)$ as sigmoids of low-degree polynomials over normalized depth.
- **PAC-Bayesian Generalization Bound:**
  - Establishing a randomized Gaussian posterior $Q(\tilde{\Theta}) \sim \mathcal{N}(\Theta, \sigma^2 I)$ and a Gaussian prior $P(\tilde{\Theta}) \sim \mathcal{N}(\Theta_{\text{uniform}}, \sigma_0^2 I)$.
  - Analytical proof showing that the Kullback-Leibler (KL) divergence between $Q$ and $P$ reduces to a quadratic $L_2$ distance penalty: $D_{\text{KL}}(Q \parallel P) = \frac{\|\Theta - \Theta_{\text{uniform}}\|_2^2}{2 \sigma_0^2} + \text{constant}$.
- **Stable Consensus Target Initialization:** Analytically deriving the initial bias $\theta_{\text{uniform}} = \ln(1/(K-1))$ to guarantee scale-preserving, uniform ensembling consensus at initialization.

## 5. Experimental Evaluation
- **Real-World Projected Representation Sandbox:**
  - Backbone: 14-layer representation manifold simulation using physical datasets (MNIST, FashionMNIST, CIFAR-10, SVHN) projected via random Johnson-Lindenstrauss mapping to 192 dimensions.
  - Multi-task feature space with inter-task coordinate overlap.
  - Non-convexity: 14-layer residual MLP with stable 0.1 residual branch scaling.
  - Calibration: $M=10$ samples per task (total 40 samples). Test: 500 samples per task (total 2000 samples).
  - Multi-seed aggregation (15 seeds).
- **Quantitative Performance:** Table comparing Expert Ceilings, Static Uniform, Ties-Merge, DARE-Merge, Offline Unconstrained, RBPM ($L_1$), and PAC-Bayes Merge ($L_2$).
- **Key Analyses:**
  - **Analysis A:** The Transductive Overfitting Trap (Offline Unconstrained overfitting tendencies).
  - **Analysis B:** Sparsity vs. Continuous Capacity (PAC-Bayes Merge $L_2$ compared with RBPM $L_1$ on Joint Mean).
  - **Analysis C:** Stochastic Weight Averaging (SWA) Effect (Static Uniform performance under extreme inter-task noise).
- **Ablation Studies:** Regularization sensitivity analysis over $\lambda_{\text{PAC}} \in \{0.001, 0.010, 0.50\}$.

## 6. Conclusion & Future Directions
- **Summary:** Reaffirming the value of mathematically grounded model merging.
- **Theoretical Impact:** Elevating model merging from heuristic parameter tweaking to a provably robust learning process.
- **Future Work:** Non-Gaussian priors, extensions to vision-language models, and dynamic routing architectures.
