# 1_summary.md - Summary of the Submission

## Overview of the Paper
This submission introduces **Dirichlet-PAC**, a mathematically grounded learning-theoretic framework for test-time multi-task model serving. The paper focuses on dynamic activation blending of parameter-efficient expert adapters (like LoRA) on a shared, frozen backbone. The authors address the severe data scarcity encountered during serve-time calibration streams (often containing fewer than 64 samples per task), where unregularized Empirical Risk Minimization (ERM) overfits to local transductive noise and leads to extreme routing temperature scales.

Rather than modeling routing parameters in unconstrained Euclidean space (which can lead to log-temperature explosion or sudden entropy collapse), Dirichlet-PAC models the sample-specific ensembling weight vector directly as a random variable drawn from a Dirichlet posterior over the probability simplex $\Delta^{K-1}$. By minimizing a PAC-Bayesian generalization bound under McAllester's theorem, they utilize the exact closed-form analytical Kullback-Leibler (KL) divergence between Dirichlet distributions on the simplex as a complexity-control penalty. 

To drive the routing policy in an unsupervised manner, the paper introduces **Subspace Energy Projection (SEP)**. SEP performs Singular Value Decomposition (SVD) on early-layer calibration activations to extract task-specific manifolds, projecting query activations onto these subspaces to extract coordinates that are subsequently normalized onto the simplex.

## Key Contributions
1. **Simplex-Constrained PAC-Bayesian Formulation:** The first learning-theoretic framework for test-time model ensembling operating directly on the probability simplex $\Delta^{K-1}$ via Dirichlet distributions.
2. **Closed-Form Dirichlet KL Penalty:** Analytical derivation and implementation of the Dirichlet-to-Dirichlet KL divergence within a prediction-space conditional PAC-Bayesian bound, stabilizing temperature calibration.
3. **Subspace Energy Projection (SEP) & Energy Normalization:** An unsupervised, basis-independent, and scale-invariant task manifold coordinate extraction method using SVD and coordinate normalization.
4. **Watertight Sample-Splitting & Discretization Proofs:** Resolves potential prior-data dependency violations by splitting calibration data into disjoint prior and optimization sets, and establishes uniform convergence over global temperatures via discretization and union-bound proofs.
5. **Empirical Evaluation:** Evaluated on a 14-layer Analytical Coordinate Sandbox (ICS) and validated on pre-trained transformer backbones (BERT-Tiny, BERT-Mini, BERT-Medium, BERT-Base) across various seeds.
