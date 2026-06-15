# 2. Novelty Check and Delta from Prior Work

## Key Novel Aspects
The paper claims novelty in three distinct areas:
1. **The SVS Operator:** Utilizing SVD to project task vectors onto a low-rank spectral manifold offline to filter out sequential fine-tuning noise and resolve multi-task interference.
2. **Global Scaling Cancellation Theory:** A formal proof explaining mathematically why positive global weight scaling factors are neutralized by standard L2, LayerNorm, and RMSNorm layers.
3. **Entropy-SVS:** An information-theoretic, training-free scheme that dynamically allocates rank capacity across layers using the Shannon spectral entropy of task-specific singular value distributions.

## Delta from Prior Work
A close reading of the literature reveals that the core algorithmic component of this paper is highly overlapping with prior work, rendering the novelty of SVS quite limited:
1. **Overlap with Task Singular Vectors (TSV-Compress):** The authors openly acknowledge in Section 2 that Gargiulo et al. (2025) proposed *Task Singular Vectors* (TSV) and "introduced TSV-Compress to retain only the top singular components to denoise weight updates." Algorithmically, **Singular Value Slicing (SVS) is virtually identical to TSV-Compress.** Both perform SVD on task vectors post-hoc and truncate them to a specified rank $k$ to denoise the weight updates before linear combination. The "delta" in SVS is purely nomenclatural ("Slicing" instead of "Compressing") and does not represent an algorithmic advance.
2. **Frobenius Norm Rescaling (BWN):** Rescaling merged weight matrices based on Frobenius norm ratios (BWN) is an incredibly straightforward technique that resembles standard weight-rescaling heuristics in existing literature. For instance, TIES-Merging (Yadav et al., 2023) utilizes a scaling factor to restore the magnitude of merged vectors. 
3. **The Global Scaling Cancellation Theory:** While the mathematical derivations for LayerNorm, RMSNorm, and L2-norm are elegant, this theoretical "contribution" actually concludes that the proposed scale-preservation operator (BWN) is **entirely redundant and mathematically useless** in virtually all modern deep architectures (such as Transformers and joint encoders). While it serves a diagnostic purpose, proving that one's own algorithm (BWN) is redundant does not constitute a substantial algorithmic or practical novelty.
4. **Entropy-SVS:** The application of Shannon entropy to the normalized singular value distribution (spectral entropy) is a long-established concept in matrix analysis, signal processing, and dimensionality reduction. Its introduction into SVD-based model merging is a logical and straightforward application of classical information theory rather than a major conceptual breakthrough.

## Characterization of Novelty
The overall novelty of the submission must be characterized as **highly incremental and largely derivative**:
- **Algorithmic Novelty:** Low. SVS is an existing concept (TSV-Compress) under a new name. BWN is a basic norm-matching heuristic. Entropy-SVS is a straightforward application of classical spectral entropy.
- **Theoretical Novelty:** Moderate. The proofs of global scaling cancellation under LayerNorm, RMSNorm, and L2-norm are neat and mathematically sound, but they apply to standard properties of normalization layers that are well-known to practitioners (e.g., that LayerNorm is scale-invariant). Explicitly writing them down in the context of model merging is helpful but does not represent a major theoretical advance in machine learning.
- **Conceptual Novelty:** Low. The paper relies heavily on existing paradigms (Task Arithmetic, SVD-based compression, and standard normalization layers).
