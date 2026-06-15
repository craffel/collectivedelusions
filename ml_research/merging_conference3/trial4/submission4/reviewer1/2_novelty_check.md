# 2. Novelty and Literature Delta Check

## Characterization of Novelty
The novelty of this submission is **incremental to moderate**. 

The core conceptual leap—constraining and smoothing layer-wise model-merging coefficients across network depth to prevent overfitting—was already introduced by **PolyMerge** (which parameterizes coefficients as continuous low-degree polynomials over depth). 

The primary contribution of this paper is replacing PolyMerge's polynomial basis functions (such as power series) with trigonometric basis functions via the **Discrete Cosine Transform (DCT-II)**. While this introduces several nice mathematical properties (such as perfect numerical conditioning and even-symmetry boundary conditions), the high-level idea of enforcing a smooth trajectory across layer depth to regularize parameterized merging remains conceptually identical to PolyMerge.

## The 'Delta' from Prior Work
The authors highlight several specific technical differences to establish a "delta" from existing literature:

1. **Delta from PolyMerge:**
   - **Numerical Conditioning:** PolyMerge uses a polynomial representation which requires optimizing coefficients of a Vandermonde matrix. This matrix is notoriously ill-conditioned, and its condition number ($\kappa$) scales exponentially with the polynomial degree $d$ and network depth $L$. SpectralMerge's DCT basis is strictly orthonormal on uniform grids, maintaining a condition number of exactly $1.0$ at any scale.
   - **Boundary Conditions:** PolyMerge polynomials can suffer from boundary run-away effects (Runge's phenomenon) at high degrees. SpectralMerge-LP uses low-frequency cosine components, and the DCT-II implicitly assumes an even symmetric boundary extension. This ensures that the reconstructed spatial derivative at virtual boundaries is exactly zero (flat), preventing artificial gradient discontinuities at the critical first and last layers of the network.
   - **DST comparison:** The authors explicitly contrast DCT-II with the Discrete Sine Transform (DST). The DST's odd boundary extension forces boundary coefficients to zero, which severely hurts performance on critical input/output layers and causes gradient spikes. The DCT-II's even extension avoids this.

2. **Delta from Weight-Space Frequency and Spectral Merging Methods:**
   - There is a rich body of concurrent and prior work applying frequency-domain or spectral techniques to model merging:
     - **FREE-Merging / FR-Merging:** Applies the Fourier Transform to filter out harmful frequency components of the *model weights* themselves.
     - **STAR / SVC:** Uses Singular Value Decomposition (SVD) on *weight matrices* or *task updates* to truncate noise-like singular values or calibrate spectral over-accumulation.
     - **SWUDI:** Uses eigendecomposition of weight updates to solve model merging as a regularized linear inverse problem.
   - **The Delta:** SpectralMerge does *not* apply any spectral or frequency transform to the model parameters or weights. Instead, it applies the DCT-II strictly to the **1D spatial sequence of layer-wise task-combining coefficients**. It is a parameterization of the search space for merging coefficients, not a weight-filtering technique.

## Critical Critique of the Claims of Novelty
While the mathematical details (boundary derivatives, orthonormality) are presented with impressive rigour, the overall novelty must be scrutinized:
* **Over-engineering of the DCT Boundary Argument:** The authors dedicate significant space to proving that the spatial derivative at virtual boundaries is mathematically zero, claiming this "completely protects the highly critical input-mapping layers." While mathematically correct, is this a real problem for PolyMerge or standard spatial optimization? Standard spatial optimizers handle physical boundary layers perfectly fine without virtual extensions. The boundary derivative analysis feels like a mathematical post-hoc justification for using DCT-II rather than a solution to a pre-existing, severe practical bottleneck.
* **Is "Spectral" an Overstatement?** The term "Spectral" is often associated in deep learning with the spectral properties of weight matrices (singular values, eigenvalues, SVD). By naming the framework "SpectralMerge," the authors risk semantic confusion with SVD-based merging methods (like STAR or SVC). A more precise name would be "Frequency-Domain Coefficient Merging."
* **Conceptual Proximity to Parameter Sharing:** Forcing coefficients to follow a low-pass trajectory (especially $F=1$) is extremely close to global parameter sharing (optimizing a single scale factor per task vector across all layers). The paper's novelty relies on showing that the minor layer-wise variations allowed by $F \in \{2, 3\}$ are both useful and robust.
