# Evaluation Task 1: Summary of the Paper

## Main Topic and Objective
The paper addresses the challenge of **layer-wise adaptive weight-space model merging** under extreme data constraints (such as few-shot calibration). Traditional layer-wise merging optimizes independent coefficients for each task across all network layers, creating a high-dimensional search space prone to transductive overfitting and poor out-of-distribution (OOD) generalization. 

To mitigate this, prior work proposed restricting coefficients to low-degree polynomial subspaces (e.g., Rademacher-Bounded Polynomial Merging or RBPM). However, polynomial curves suffer from boundary runaway (Runge's phenomenon), causing extreme, unstable ensembling weights at the critical first and last layers of deep neural networks.

To resolve these limitations, this paper proposes:
1. **Rademacher-Bounded Fourier Trajectory Merging (RB-FTM)**: Parameterizes layer-wise ensembling trajectories using a low-frequency continuous Fourier series across depth coordinates.
2. **Rademacher-Bounded Discrete Cosine Trajectory Merging (RB-DCTM)**: A non-periodic variant using a half-period cosine basis that eliminates periodic boundary identities while maintaining smoothness and boundary stability.

---

## Proposed Approach
Rather than treating layer-wise coefficients $\Lambda \in [0, 1]^{K \times L}$ as independent parameters, the authors project them onto a continuous, low-frequency spectral subspace of depth coordinates $z = l / (L-1) \in [0, 1]$:
$$\alpha_k(l) = \Pi_{[0,1]} \left( a_{k,0} + \sum_{f=1}^F (a_{k,f}\cos(2\pi f z) + b_{k,f}\sin(2\pi f z)) \right)$$
For the DCT variant, a half-frequency cosine-only series is used:
$$\alpha^{\text{DCT}}_k(l) = \Pi_{[0,1]} \left( a_{k,0} + \sum_{f=1}^F a_{k,f}\cos(\pi f z) \right)$$
The active capacity is regularized via a **Spectral Lasso ($L_1$) regularizer** applied strictly to the harmonic coefficients (excluding the baseline uniform weight $a_{k,0}$), which physically penalizes trajectory fluctuations:
$$\mathcal{L}(\Theta) = \mathcal{L}_{\text{CE}}(\mathcal{D}_{\text{cal}}; \Theta) + \gamma \sum_{k=0}^{K-1} \|\theta_{k,\text{harm}}\|_1$$

---

## Key Findings and Claims
1. **Trajectory Complexity Guarantees**:
   The authors prove mathematically that the empirical Rademacher complexity of the Fourier trajectory class scales with the spectral cutoff frequency $F$ and decays with network depth $L$ as $\mathcal{O}(\sqrt{\ln(F)/L})$, completely independent of the parameter count of the underlying deep network.
2. **DCT Boundary Independence & Tighter Bounds**:
   The Discrete Cosine basis (RB-DCTM) removes the periodic identity $\alpha(0) = \alpha(L-1)$ forced by the standard Fourier series, granting boundary value independence. Furthermore, its Rademacher complexity is strictly tighter than that of the Fourier class because it eliminates the sine components, reducing the basis size.
3. **Runge's Phenomenon Mitigation**:
   Both spectral classes successfully eliminate the boundary runaway oscillations of polynomial trajectories (such as RBPM, $d=2$), ensuring smooth, stable parameter trajectories.
4. **The Static Uniform Dominance Paradox**:
   Inside an idealized, perfectly aligned linear **Analytical Coordinate Sandbox (ACS)**, the parameter-free **Static Uniform** baseline (which simply averages experts using $1/K$ weights) consistently acts as a highly robust empirical upper bound (85.10% accuracy on CNN and 83.75% on CLIP). This is because any parameter adaptation in a perfectly aligned space introduces coordinate shearing and degrades predictions.
5. **Real-World ViT-B/16 Validation**:
   When merging actual Vision Transformer expert checkpoints (CIFAR-10 and CIFAR-100), where representation spaces are *not* perfectly coordinate-aligned, uniform merging suffers from representational collapse (71.30% average accuracy). In this heterogeneous setting, **RB-DCTM (F=2)** achieves the peak joint average accuracy of **74.90%**, outperforming Static Uniform (+3.60%), unconstrained optimization (+5.10%), and the polynomial competitor RBPM (+4.20%).

---

## Explicitly Claimed Contributions
* **Trigonometric Representation**: Projecting discrete ensembling weights onto a low-frequency continuous Fourier/DCT subspace across network depth, acting as a natural low-pass filter.
* **Learning-Theoretic Complexity Guarantees**: Deriving empirical Rademacher complexity bounds for both Fourier (Theorem 3.1) and DCT (Theorem 3.4) ensembling trajectory classes.
* **Analytical Spectral Lasso Regularization**: Applying an $L_1$ penalty strictly to harmonic coefficients to control active trajectory capacity without shrinking the baseline ensembling scale.
* **Boundary Runaway Mitigation**: Showing that trigonometric bases resolve the boundary oscillations of polynomial trajectories under depth-wise coordinate mapping.
* **ACS Sandbox and Real-World Evaluation**: Providing controlled experiments in a simulated coordinate sandbox and verifying the practical utility of the spectral regularizers on actual deep network parameters (ViT-B/16).
