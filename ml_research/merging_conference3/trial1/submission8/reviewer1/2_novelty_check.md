# 2. Novelty and Originality Check

This submission investigates a niche intersection of model merging: combining manifold-based merging (like OrthoMerge) with representational isotropy/spectral balancing (like SAIM).

### Analysis of Novelty
* **The "Delta" from Prior Work**: 
  - *OrthoMerge* (Yang et al., 2026) introduced the framework of extracting rotations via Procrustes decomposition, projecting to the Lie algebra $\mathfrak{so}(d)$ via the inverse Cayley transform, averaging, and mapping back.
  - *SAIM* (2026) introduced SVD-based spectral balancing (isotropic smoothing) to reduce multi-task interference in Euclidean spaces.
  - This work attempts to merge these two ideas by performing spectral balancing directly on the skew-symmetric generators of the Lie algebra tangent space.

* **Characterization of Novelty**:
  - The novelty of the *proposed* successful method is low. The main "novel" attempt—performing SVD-based spectral balancing in the Lie algebra tangent space (RIMO)—catastrophically fails (collapsing performance to $13\%$).
  - The theoretical contributions consist of explaining why this combined approach fails (the Kernel and Spectrum Distortion Theorems). While the mathematical derivations are elegant, they essentially prove that standard SVD is geometrically incompatible with Lie algebra structures.
  - The final recommended mitigation, **RIMO-Pruned**, simply discards the small singular values (rank pruning) rather than balancing them, which essentially reverts the "isotropic balancing" idea and keeps the active subspaces intact.
  - The other proposed mitigations—real Schur decomposition and complex Hermitian eigen-decomposition—are standard mathematical tools applied to preserve skew-symmetry during decomposition.
  - From a conceptual standpoint, the paper's novelty lies primarily in identifying and analyzing a self-induced mathematical obstacle (the spectral pitfall of a highly complex pipeline) and proposing standard mathematical corrections to salvage it.
