# Paper Summary: Rademacher-Bounded Fourier Trajectory Merging (RB-FTM)

## Main Topic and Motivation
The paper addresses the challenge of **layer-wise adaptive weight-space ensembling** for multi-task expert networks. When merging parameters of task experts fine-tuned from a shared pre-trained initialization, tuning independent layer-wise coefficients on small calibration datasets (few-shot adaptation) leads to high-dimensional overparameterization and transductive overfitting. This overfitting causes poor generalization on the true multi-task distribution.

Prior attempts to constrain these ensembling trajectories across depth—such as Rademacher-Bounded Polynomial Merging (RBPM)—parameterize coefficients using low-degree polynomials. However, low-degree polynomials (e.g., quadratic curves, $d=2$) suffer from severe **boundary runaway** (similar to Runge's phenomenon). This forces ensembling coefficients at the network boundaries (first and last layers) to extreme values, catastrophically degrading low-level feature extraction and final classification performance.

## Proposed Approach
The authors introduce **Rademacher-Bounded Fourier Trajectory Merging (RB-FTM)** and its non-periodic counterpart, **Rademacher-Bounded Discrete Cosine Trajectory Merging (RB-DCTM)**. 
1. **Spectral Trajectory Parameterization**: Instead of optimizing $K \times L$ independent coefficients (for $K$ tasks and $L$ layers), ensembling trajectories are parameterized as a continuous spectral series composed of a small number of low-frequency harmonic sinusoids (Fourier basis) or half-period cosines (Discrete Cosine Transform basis).
2. **Learning-Theoretic Guarantees**: The authors prove that the empirical Rademacher complexity of these trigonometric trajectory classes over network depth coordinates is strictly bounded by the spectral cutoff frequency $F$ and network depth $L$, completely independent of the underlying network's parameter count.
3. **Spectral Lasso ($L_1$) Regularization**: To physically enforce this complexity bound, they introduce an $L_1$ penalty strictly on the harmonic coefficients of the trajectory, leaving the baseline uniform coefficient unpenalized to maintain activation scales.
4. **Boundary runaway mitigation**: RB-DCTM resolves the periodic boundary identity forced by standard Fourier series while providing a homogeneous Neumann boundary condition (flat derivative) that stabilizes initial feature extraction and final classification boundaries.

## Key Findings and Claimed Contributions
1. **Trigonometric Representation**: Projecting discrete ensembling coefficients onto a low-frequency continuous Fourier/DCT subspace prunes high-frequency noise and representation misalignment.
2. **Mathematical Complexity Bounds**: 
   - For Fourier Trajectories: $\widehat{\mathcal{R}}_L(\mathcal{H}_F) \le C_0 \sqrt{\frac{2 \ln(4F+2)}{L}}$
   - For DCT Trajectories: $\widehat{\mathcal{R}}_L(\mathcal{H}_F^{\text{DCT}}) \le C_0 \sqrt{\frac{2 \ln(2F+2)}{L}}$
   The DCT trajectory variant has a strictly tighter bound and eliminates periodic boundary identity constraints.
3. **Downstream Generalization Bridge**: The paper derives a downstream prediction generalization bound via covering numbers of the trajectory-parameterized network class, establishing an explicit $\widetilde{\mathcal{O}}(1/\sqrt{N})$ decay rate over data samples $N$ and showing that the predictive capacity scales as $\mathcal{O}(\sqrt{K/N})$ with respect to task count $K$.
4. **Elimination of Runge's Phenomenon**: Trigonometric trajectories naturally mitigate boundary runaway.
5. **Experimental Performance**:
   - In a synthetic **Analytical Coordinate Sandbox (ACS)** simulating representation propagation, RB-FTM and RB-DCTM outperform other tuned/adaptive baselines (unconstrained, globally-scaled, and polynomial).
   - In a real-world validation merging CIFAR-10 and CIFAR-100 expert Vision Transformers (ViT-B/16), RB-DCTM ($F=2$) achieves a **74.90%** joint average accuracy, outperforming the Static Uniform baseline ($71.30\%$), Globally-Scaled Task Arithmetic ($72.50\%$), and the polynomial competitor RBPM ($70.70\%$).
6. **The Static Uniform Dominance Paradox**: In the synthetic sandbox, the parameter-free **Static Uniform** baseline outperforms all adaptive/tuned ensembling methods. The authors show that in perfectly aligned spaces, any adaptation introduces anisotropic representation shearing, which distorts the global geometric topology. In real-world networks with coordinate misalignment, however, adaptive ensembling successfully recovers representation collapse.
