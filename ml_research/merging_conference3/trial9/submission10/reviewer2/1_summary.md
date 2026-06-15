# Intermediate Evaluation 1: Summary of the Paper

## 1. Main Topic and Scope
The paper addresses the challenge of **test-time multi-task model serving** utilizing parameter-efficient experts (specifically, Low-Rank Adaptation/LoRA adapters) on a shared frozen backbone. It focuses on **dynamic ensembling (activation-space blending)** at inference time. The core problem highlighted is the extreme scarcity of calibration data available at serve-time (typically fewer than 64 samples per task), which causes standard unregularized Empirical Risk Minimization (ERM) and unconstrained temperature-based routing policies to suffer from transductive noise overfitting, temperature parameter divergence ("log-temperature explosion"), and subsequent generalization collapse.

## 2. Proposed Approach
To address the serving-time data-scarcity and routing instability, the authors introduce **Dirichlet-PAC**, a learning-theoretic framework consisting of:
- **Simplex-Constrained Routing Policy:** Models the ensembling weight vector $\boldsymbol{\alpha}_b$ directly over the probability simplex $\Delta^{K-1}$ as a random variable distributed according to a Dirichlet posterior.
- **Analytical Dirichlet KL Complexity Control:** Utilizes McAllester's PAC-Bayesian theorem to derive a closed-form generalization bound. The complexity penalty is formulated using the exact analytical Kullback-Leibler (KL) divergence between Dirichlet distributions over the simplex, which acts as an information barrier to prevent temperature parameters from collapsing or exploding.
- **Subspace Energy Projection (SEP):** An unsupervised, task-agnostic dimensionality reduction protocol that performs Singular Value Decomposition (SVD) on early-layer activations from a prior calibration split to extract task-specific projection bases. Online queries are projected onto these bases to compute normalized, scale-invariant task coordinates.
- **Unsupervised Variant (PEM-Div):** An adaptation of the framework for label-free settings using Prediction Entropy Minimization combined with a batch-averaged weight entropy maximization (diversity) penalty.
- **Sample-Splitting Protocol:** Disjointly partitions the calibration set into a *Prior Calibration Split* (for SVD subspace learning) and an *Optimization Calibration Split* (for bound minimization) to maintain data-independence assumptions.

## 3. Key Findings
- **Synthetic Evaluation (Analytical Coordinate Sandbox/ICS):** 
  - On orthogonal manifolds ($\rho = 0.0$), Dirichlet-PAC achieves an average accuracy of **77.88%**, and PEM-Div achieves **79.43%**, compared to **76.12%** for unregularized Temp-Only ERM.
  - On overlapping manifolds ($\rho = 0.33$), Dirichlet-PAC achieves **76.32%**, and PEM-Div achieves **78.73%**, compared to **75.67%** for Temp-Only ERM.
  - The authors claim that Dirichlet-PAC drastically stabilizes optimization variance across seeds.
- **Physical Evaluation (BERT-Scale Backbones):**
  - Evaluated on multiple scales of pre-trained BERT backbones (`bert-tiny`, `bert-mini`, `bert-medium`, `bert-base-uncased`) with Multi-LoRA adapters on three text classification tasks.
  - The authors report that unregularized learned routers (ERM, PAC-ZCA) suffer from severe performance collapse (67%–74%), while Dirichlet-PAC stabilizes serving, achieving **92.00%** on BERT-Base and **99.33%** on BERT-Mini.

## 4. Explicitly Claimed Contributions and Accompanying Evidence
- **Contribution 1: Simplex-Constrained PAC-Bayesian Theory.** Operates directly on the probability simplex using a Dirichlet routing policy. *Evidence:* Mathematical derivations and formulation of prediction-space PAC-Bayes bounds in Section 3.
- **Contribution 2: Analytical Dirichlet KL Complexity Control.** Closed-form complexity penalty utilizing Gamma and digamma functions. *Evidence:* The closed-form analytical expression in Equation 11.
- **Contribution 3: Subspace Energy Projection (SEP) with scale-invariance and basis-independence proofs.** *Evidence:* Proposition 1 in Section 3.2 proving basis independence and scale-invariance of SVD-based normalized projection coordinates.
- **Contribution 4: Superior serving performance and stability on synthetic and real BERT models.** *Evidence:* Benchmark tables (Table 1, Table 2, Table 3) comparing Dirichlet-PAC against nine baselines, including weight-merging and activation-routing state-of-the-art.
