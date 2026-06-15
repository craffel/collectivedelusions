# 1. Summary of the Paper

## Main Topic and Approach
The paper introduces **Dirichlet-PAC**, a mathematically rigorous learning-theoretic framework for test-time multi-task model serving (using parameter-efficient adapters like LoRA) on a shared frozen backbone. 

During test-time serving, a router must adapt its ensembling coefficients based on extremely scarce calibration streams (often fewer than 64 samples per task), making standard unregularized Empirical Risk Minimization (ERM) highly susceptible to transductive noise overfitting and temperature divergence. To address this, Dirichlet-PAC models the sample-specific ensembling weights directly as a random vector drawn from a Dirichlet distribution over the probability simplex $\Delta^{K-1}$. 

Under McAllester's PAC-Bayesian theorem, the authors derive a closed-form prediction-space generalization bound utilizing the exact analytical Kullback-Leibler (KL) divergence between Dirichlet distributions over the simplex itself. This acts as an elegant complexity-control penalty that prevents temperature parameters from exploding or collapsing, naturally promoting smooth, cooperative expert blending on task boundaries.

To drive this routing policy, the authors introduce **Subspace Energy Projection (SEP)**, an unsupervised, task-agnostic coordinate extraction system. It uses Singular Value Decomposition (SVD) on early-layer activations to construct orthonormal task-specific projection bases. Online queries are projected onto these bases and energy-normalized to map coordinates directly to the simplex.

Finally, the authors propose **Dirichlet-PAC Unsupervised (PEM-Div)**, which replaces the supervised loss with a Normalized Prediction Entropy Minimization (PEM) loss combined with batch-averaged ensembling weight diversity maximization, achieving a fully unsupervised test-time serving pathway.

## Key Findings and Empirical Results
- **Generalization and Stability:** Evaluated on a simulated 14-layer Analytical Coordinate Sandbox (ICS), Dirichlet-PAC achieves outstanding performance under extreme data-scarce splits ($N=64$). On orthogonal manifolds ($\rho=0.0$), it achieves **77.88% $\pm$ 1.19%** accuracy, outperforming standard unregularized Temp-Only ERM (**76.12% $\pm$ 1.86%**) and Gaussian PAC-ZCA (**75.67% $\pm$ 2.04%**), while dramatically reducing optimization variance.
- **The Success of Unsupervised PEM-Div:** The unsupervised PEM-Div variant achieves a remarkable **79.43% $\pm$ 1.05%** accuracy on orthogonal task manifolds, outperforming its supervised counterpart and matching or exceeding the supervised heuristic baseline SABLE (Raw Coords) (**79.02% $\pm$ 0.98%**), establishing a robust alternative for label-free edge serving.
- **Representation Corruption and Safety Valve:** Under high representation noise, standard routers overfit or collapse, corrupting internal activation vectors ("representation corruption"). Dirichlet-PAC’s energy-normalization serves as an "information-theoretic safety valve," naturally causing the Dirichlet posterior to collapse gracefully to a safe uniform distribution on high-noise queries.

## Explicitly Claimed Contributions (with Evidence)
1. **Simplex-Constrained PAC-Bayesian Theory:** The first learning-theoretic framework for test-time model ensembling operating directly on the probability simplex $\Delta^{K-1}$ using a Dirichlet policy (Sections 3.3, 3.5).
2. **Analytical Dirichlet KL Complexity Control:** Formal derivation of the exact analytical Dirichlet KL divergence within the PAC-Bayesian bound, providing a closed-form complexity penalty that stabilizes log-temperatures and acts as an elegant entropy regularizer (Section 3.4, Appendix A).
3. **Unsupervised Subspace Projection (SEP) with Energy Normalization:** Introduction of SEP combined with a novel energy-normalization protocol to project activations onto SVD subspaces and map coordinates to the simplex. The authors prove that this is mathematically scale-invariant and basis-independent (Section 3.2, Proposition 3.1).
4. **Superior and Stable Serving Performance:** Demonstration that Dirichlet-PAC significantly reduces optimization variance, resolves representation corruption under noise, and outperforms state-of-the-art baselines in the Analytical Coordinate Sandbox (Section 4.3).
5. **A Fully Unsupervised Serving Pathway (PEM-Div):** Formulation of a fully unsupervised prediction entropy minimization router that achieves state-of-the-art performance without labeled calibration data (Section 3.5, Section 4.3).
6. **Extensive Theoretical Analyses:** Rigorous first-principles derivation showing that representation clashing noise scales with ensembling entropy (Section 4.4); martingale-based sequential streaming extension (Section 5.1); and sensitivity analysis under quantization using the Wedin-Davis perturbation theorem (Section 5.2).
