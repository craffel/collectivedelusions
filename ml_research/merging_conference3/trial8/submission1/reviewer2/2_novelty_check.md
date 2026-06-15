# Novelty and Delta Assessment: HyperMerge

## Key Novel Aspects
The primary novelty of HyperMerge lies in **conceptualizing model merging and activation-space ensembling through the lens of non-Euclidean, hyperbolic geometry**. Specifically:
- **Non-Euclidean Representation Workspace:** Moving activation blending out of $\mathbb{R}^D$ and into the Poincaré Ball ($\mathbb{D}_c^D$) is a novel conceptual direction for modular deep learning.
- **Einstein Midpoints for Merging:** Utilizing the projective flat properties of the Beltrami-Klein model to perform ensembling via Lorentz-weighted Einstein midpoints is highly creative. It addresses the non-associative and non-commutative limitations of standard Möbius addition in the Poincaré Ball.

## Delta from Prior Work
1. **From Parameter-Space Merging (e.g., TIES, DARE):** While traditional merging fuses model weights statically off-line, HyperMerge is an online, test-time activation blending method.
2. **From Euclidean Dynamic Ensembling (e.g., PFSR, SABLE, SPS-ZCA):** 
   - PFSR, SABLE, and SPS-ZCA perform routing and ensembling in flat Euclidean space.
   - SABLE averages expert updates linearly in $\mathbb{R}^D$.
   - SPS-ZCA uses Euclidean centroid alignment.
   - HyperMerge replaces these flat operations with geodesic distances, exponential/logarithmic mappings, and Einstein midpoints in Klein space.

## Characterization of Novelty
The novelty can be characterized as **high in mathematical creativity but conceptually incremental in practice**. 

While the use of hyperbolic geometry and Klein-space algebra is mathematically sophisticated and highly original in this specific sub-field, it represents an application of existing Hyperbolic Neural Network (HNN) primitives (e.g., Ganea et al., 2018; Nickel & Kiela, 2017) to a new problem domain (model merging). 

Crucially, the practical "delta" is undermined by the performance of much simpler Euclidean baselines (like SABLE), which achieve identical or superior results without any hyperbolic machinery. This suggests that the heavy mathematical framework may be an elegant but unnecessary layer of complexity for this particular task.
