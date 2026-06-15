# Evaluation Phase 2: Novelty and Related Work Check

## 1. Assessment of Key Novel Aspects
The primary novel contribution of SpectralMerge is the re-parameterization of the optimization search space for layer-wise weight-merging coefficients. Specifically:
* **DCT-II on Layer Trajectories:** Treating the sequence of layer-wise combining coefficients $\vec{\alpha}_k$ across network depth $L$ as a discrete 1D spatial signal, and mapping it to the frequency domain via the Discrete Cosine Transform (DCT-II).
* **Analytical Frequency Constraints:** Formulating low-pass hard cutoffs (SpectralMerge-LP) and soft spectral decay penalties (SpectralMerge-Reg) to regularize the optimization trajectory directly in the spectral domain.
* **Block-wise/Layer-type Decompositions:** Extending this 1D spectral formulation to multi-scale block-wise and functional-category subsets to handle architectural heterogeneity.

## 2. The "Delta" from Prior Work
The proposed method sits at the intersection of parameterized merging (such as AdaMerging and RegCalMerge) and spatial coefficient smoothing (such as PolyMerge).

### A. Comparison with PolyMerge (Spatial Polynomials)
* **Basis Functions:** PolyMerge represents the layer trajectory using low-degree continuous polynomials. SpectralMerge represents it using trigonometric cosine functions (DCT-II).
* **Numerical Conditioning:** Polynomial regression involves a Vandermonde matrix, which becomes ill-conditioned as network depth $L$ or polynomial degree $d$ increases. SpectralMerge leverages the orthonormal DCT basis, which has a condition number of exactly $1.0$ at all scales.
* **Boundary Behaviors:** Polynomials are susceptible to boundary instabilities (Runge's phenomenon). The DCT-II implicitly assumes even symmetric boundaries, which mathematically forces spatial derivatives to be flat (zero slope) at virtual boundaries, stabilizing the first and last layers.
* **Critique of the Delta:** While the transition from polynomials to cosine bases yields elegant numerical properties, representing 1D trajectories using orthogonal trigonometric bases (e.g., Fourier or Cosine Series) instead of power-series polynomials is a standard and classical technique in signal processing, regression analysis, and spline theory. The mathematical "delta" is highly solid and elegant, but it is an evolutionary refinement of PolyMerge rather than an entirely unprecedented conceptual leap.

### B. Missing Literature: Frequency-Domain Model Merging
* **The Missing Link:** The authors state that they "challenge the traditional coordinate-dependent spatial paradigm of model merging and propose a novel frequency-domain parameterization." However, they completely miss a highly relevant and concurrent line of research that also operates in the frequency domain of model merging: **FREE-Merging (Fourier Transform for Efficient Model Merging)**, published at **ICCV 2025**.
* **The Conceptual Distinction:** 
  * **FREE-Merging (and its precursor FR-Merging)** applies Fourier Transforms directly to the *model parameters (weights/delta-weights)* of deep neural networks to filter out high-frequency spatial components that are associated with task interference and representation clashes. It is a parameter-space filtering and sparsification technique.
  * **SpectralMerge** applies the Discrete Cosine Transform (DCT-II) to the *layer-wise task-combining coefficients* (a 1D signal of length $L$, e.g., 12 or 18 coordinates). It is an optimization search-space re-parameterization and regularization technique.
* **Significance of the Gap:** By failing to cite or discuss FREE-Merging, the paper's claim of "introducing a novel frequency-domain parameter consolidation paradigm in model merging" is overstated. There is already work leveraging the frequency domain in model merging, albeit at the parameter level rather than the optimization coefficient level. Situating SpectralMerge relative to FREE-Merging is essential to clarify that FREE-Merging filters the high-dimensional backbone weights, while SpectralMerge regularizes the low-dimensional combining trajectory.

## 3. Characterization of Novelty
The novelty of this paper is best characterized as **Significant and Highly Practical, but Evolutionary**. 
* **Why it is Significant:** The paper takes a very clean, mathematically rigorous approach to solving a real and severe problem in model merging (validation overfitting under data scarcity). The mathematical justification for DCT-II (even-boundary derivatives, energy compaction, real-valued coordinates, and perfect conditioning) is exceptionally well-thought-out, and the empirical results on physical networks (ResNet-18) show massive blowout performance (+25.00% accuracy over PolyMerge and spatial search).
* **Why it is Evolutionary (not Revolutionary):** Re-parameterizing 1D functions via DCT or Fourier coefficients to enforce smoothness is a classic mathematical tool. Furthermore, the concept of frequency-domain manipulation in model merging has already been initiated by FREE-Merging. The paper's excessive hype ("paradigm-shifting", "visionary", "completely refuting") overstates the conceptual leap, which is a highly successful and elegant application of classical digital signal processing to parameterized model merging.
