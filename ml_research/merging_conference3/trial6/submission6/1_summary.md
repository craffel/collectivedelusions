# 1. Summary of the Submission

## High-Level Goal
The submission addresses the **Overfitting-Optimizer Paradox** (or transductive overfitting) in parameter-space model merging. When ensembling coefficients (e.g., layer-wise merging weights) are dynamically optimized on extremely scarce calibration data (e.g., 10 samples per task), the optimizer overfits to the transductive sampling noise, causing chaotic coefficient oscillations across adjacent layers and catastrophic out-of-distribution (OOD) generalization collapse.

## Key Proposed Idea
The authors propose **PAC-Bayes Merge**, a formal statistical learning-theoretic framework that regularizes the ensembling trajectory using Alquier's linear PAC-Bayesian generalization bound.

The method consists of the following key steps:
1. **Polynomial Trajectory Parameterization:** Normalize the network layer index to a continuous depth coordinate $z \in [0, 1]$. Parameterize the task-specific ensembling coefficients using a low-degree polynomial trajectory $p_k(z; \theta_k)$ of degree $d \le 3$. Bounded ensembling coefficients are obtained via a sigmoid mapping: $\alpha_k(l; \theta_k) = \sigma(p_k(l/(L-1); \theta_k))$.
2. **PAC-Bayesian Formulation:** Define a randomized Gaussian posterior classifier $Q(\tilde{\Theta}) \sim \mathcal{N}(\Theta, \sigma^2 I)$ centered around the learnable trajectory coefficients, and a spherical Gaussian prior $P(\tilde{\Theta}) \sim \mathcal{N}(\Theta_{\text{uniform}}, \sigma_0^2 I)$ centered at the stable, uniform ensembling consensus.
3. **Quadratic $L_2$ Consensus-Pulling Penalty:** Minimizing Alquier's linear PAC-Bayesian generalization bound with respect to the mean parameters $\Theta$ analytically yields a quadratic $L_2$ penalty pulling the coordinates toward the uniform baseline: $\mathcal{R}_{\text{PAC}}(\Theta) = \|\Theta - \Theta_{\text{uniform}}\|_2^2$.
4. **Theory-to-Practice Bridging:** Optimize the expected empirical risk using Monte Carlo sampling during training (expected risk) and test-time evaluation (posterior ensemble averaging) to strictly align optimization and evaluation with the randomized classifier assumptions.
5. **Non-Isotropic Fisher Extension:** Extend the framework using diagonal Gaussian prior/posterior distributions with coordinate-wise prior variances inversely proportional to their local empirical Fisher Information Matrix (FIM) sensitivities.

## Experimental Sandbox Context
The authors evaluate their method in a "projected representation sandbox":
- **Datasets:** MNIST, FashionMNIST, CIFAR-10, SVHN.
- **JL Projection:** Images are projected into $D_{\text{feat}} = 192$ features using random seed-specific Johnson-Lindenstrauss matrices.
- **Model:** A 14-layer deep MLP residual backbone with width 64 (~63,000 parameters) and residual branches scaled by $0.1$.
- **Budget:** Experts are trained on $N_{\text{train}} = 300$ samples, calibrated on $M = 10$ samples per task, and evaluated on $500$ disjoint test samples per task across 15 random seeds.

## Main Quantitative Results
- **Expert Ceilings:** 48.79 $\pm$ 1.02% Joint Mean accuracy.
- **Static Uniform Merging:** 33.57 $\pm$ 2.56%.
- **Ties-Merge:** 29.68 $\pm$ 3.48% (with optimal $p_{\text{trim}} = 0.80$).
- **DARE-Merge:** 33.24 $\pm$ 2.51% (with optimal $p_{\text{drop}} = 0.10$).
- **Offline Unconstrained Tuning:** 36.09 $\pm$ 2.53%.
- **RBPM ($L_1$ Trajectory Regularization):** 36.24 $\pm$ 2.18%.
- **Ours (Deterministic Compiled):** 36.22 $\pm$ 2.23%.
- **Ours (Randomized Ensemble):** 36.09 $\pm$ 2.23%.
- **Ours (FIM Deterministic Compiled):** 36.13 $\pm$ 2.18%.
- **Ours (FIM Randomized Ensemble):** 36.07 $\pm$ 2.17%.

Both deterministic compiled models statically merge weights for zero runtime latency, whereas the randomized ensembles use 5 Monte Carlo forward passes. Both isotropic and non-isotropic PAC-Bayes Merge models significantly outperform Static Uniform, Ties-Merge, and DARE-Merge, and match the performance of the sparse $L_1$-regularized RBPM baseline.
