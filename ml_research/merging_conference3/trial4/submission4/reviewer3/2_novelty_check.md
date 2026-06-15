# 2. Novelty and Delta Check

This document provides a critical assessment of the novelty of **SpectralMerge**, detailing its "delta" relative to closely related prior work, and characterizing its overall contribution level.

## Characterization of Novelty
We characterize the novelty of this paper as **significant and conceptually refreshing**. While model merging itself is an active field, the dominant paradigm for parameterized merging has focused entirely on physical, spatial coordinate optimization (e.g., adjusting coefficients layer-by-layer) or heuristic weight-space modifications (e.g., pruning, sign alignment). By identifying the "spatial paradigm" as inherently redundant, ill-conditioned, and prone to severe overfitting under data scarcity, this paper introduces a paradigm shift: treating the sequence of layer-wise combining coefficients as a discrete 1D signal and optimizing its spectral representation. 

The novelty does not simply lie in applying the Discrete Cosine Transform (DCT-II) to deep learning parameters, but rather in **re-parameterizing the dynamic merging trajectory across layer depth** to enforce structural regularizations that are mathematically impossible or extremely difficult to represent in the spatial coordinate domain.

## Delta from Prior Work

### 1. Delta from Spatial Parameterized Merging (AdaMerging, RegCalMerge)
- **Prior Work**: Methods like AdaMerging and RegCalMerge optimize layer-wise combining coefficients as independent, unconstrained variables in the spatial depth coordinate space.
- **SpectralMerge Delta**: Instead of treating layer coefficients as independent variables, SpectralMerge maps them to orthogonal spectral coordinates via the DCT-II. This allows direct control over the spectral composition of the trajectory. High-frequency coordinates representing noisy oscillations can be analytically zeroed out (SpectralMerge-LP) or softly penalized (SpectralMerge-Reg), reducing the search-space dimensionality and preventing local overfitting.

### 2. Delta from Spatial Smoothing (PolyMerge)
- **Prior Work**: PolyMerge restricts merging coefficients to continuous polynomial curves over normalized network depth (e.g., degree $d=2$).
- **SpectralMerge Delta**:
  - **Numerical Conditioning**: Standard power series polynomials used in PolyMerge are highly collinear. The corresponding Vandermonde matrix is notoriously ill-conditioned, with its condition number growing exponentially as network depth $L$ or degree $d$ increases (leading to optimization instability). In contrast, the DCT-II basis functions are strictly orthonormal on uniform grids, guaranteeing a condition number of exactly $1.0$ (perfect conditioning) at *any* scale.
  - **Boundary Transitions**: PolyMerge is highly sensitive to boundary conditions and susceptible to Runge's phenomenon (wild oscillations near boundaries at moderately high degrees). DCT-II implicitly assumes an even symmetric boundary extension, which mathematically guarantees that the spatial derivative at the virtual boundaries of the network is exactly zero ($\frac{d\alpha}{dl} = 0$). This "boundary-flattening" effect protects critical input layers and classification heads from high-frequency gradient spikes.
  - **Soft Regularization**: PolyMerge only supports a hard degree cutoff (restricting to degree $d$). SpectralMerge supports soft spectral regularization (SpectralMerge-Reg) with a quadratic Spectral Decay Penalty ($\lambda_j = \mu \cdot j^2$), which softly penalizes high frequencies without completely removing the capacity to capture local, high-frequency transitions.

### 3. Delta from Weight-level Spectral Methods (STAR, SWUDI, ResMerge, SVC)
- **Prior Work**: Recent "spectral" model merging methods decompose high-dimensional model weight matrices directly (typically via SVD) to truncate noisy singular value directions or calibrate over-accumulation.
- **SpectralMerge Delta**: SpectralMerge does **not** perform spectral decomposition on the high-dimensional weight parameters or activations. Instead, it operates on the low-dimensional *1D sequence of layer-wise scaling coefficients* ($\vec{\alpha}_k \in \mathbb{R}^L$) across network depth. It is therefore computationally far lighter (computational overhead is $< 0.0001\%$ of a single forward pass, compared to SVD, which can be computationally heavy) and conceptually orthogonal to weight-level decomposition.

### 4. Delta from Frequency-domain Weight/Feature Filtering (FR-Merging, ToMe)
- **Prior Work**: FR-Merging filters task interference by transforming the actual weights of the model backbone into the frequency domain. Fourier ToMe operates in the frequency domain for token clustering.
- **SpectralMerge Delta**: SpectralMerge does not filter weights or activations in the frequency domain. It only filters the *layer-wise optimization trajectory of combining coefficients*. This allows it to act as an optimization-space regularizer rather than a fixed-backbone filter.

## Summary of Key Novel Insights
1. **The PEFT-Induced Step-Function Discontinuity**: The paper provides a highly sophisticated analysis of how Parameter-Efficient Fine-Tuning (such as localized fine-tuning on specific blocks/classification heads) affects model merging. Localized updates create a step-function discontinuity in parameter sensitivity across depth. Because a step-function has infinite frequency support, rigid low-pass hard cutoffs (LP) or dynamic hard schedulers (LP-Adaptive) catastrophically underfit and fail. However, soft spectral decay (Reg) successfully resolves this by allowing validation gradients to selectively activate localized high-frequency coordinates while still suppressing validation noise.
2. **Boundary Flattening Mathematics**: Showing that the symmetric boundary condition of the DCT-II guarantees flat spatial derivatives at virtual boundaries. This is a brilliant connection between Digital Signal Processing (DSP) mathematical properties and deep learning gradient flow/backpropagation, which explains why DCT-II outperforms other representations like the Discrete Sine Transform (DST).
