# 2. Novelty and Literature Delta Check

## Characterization of Novelty
The novelty of the proposed **Singular Value Slicing (SVS)** is **incremental but conceptually solid**. The core mechanism—applying Singular Value Decomposition (SVD) to weight matrices/task vectors and truncating them to their principal components—is an established tool in linear algebra and model compression (e.g., in low-rank adaptation like LoRA, tensor-train decompositions, and low-rank approximation of weight matrices). 

However, the paper's significance and "delta" from prior work lies in three aspects:
1. **Application of SVD-based Denoising directly to Multi-Task Offline Model Merging:** Applying SVD specifically to task vectors ($T_t = W_t - W_0$) post-hoc as a "low-pass filter" to resolve destructive multi-task parameter interference.
2. **Global Scaling Cancellation Theory:** Formulating a mathematical explanation of why global scale-preservation schemes (such as Barycentric Weight Normalization) are redundant in standard Transformer backbones due to downstream normalization layers (L2-norm, LayerNorm, RMSNorm).
3. **Entropy-SVS:** Transitioning from uniform low-rank truncation (applying a static rank $k$ across all layers) to a dynamic, locally adaptive, and information-theoretic rank allocation based on the Shannon spectral entropy of each layer's singular values.

## Comparison and Delta from Key Prior Works

### 1. Task Arithmetic (TA) (Ilharco et al., 2022)
*   *Delta:* SVS modifies TA by first applying a low-rank SVD projection to each task vector $\tilde{T}_t = \mathcal{S}_k(T_t)$ before linear addition. 
*   *Novelty:* This addresses TA's primary vulnerability (destructive interference and fine-tuning noise) using continuous spectral approximation, whereas TA operates on raw full-rank vectors.

### 2. Heuristic Coordinate-Basis Pruning (TIES-Merging, DARE)
*   *Delta:* TIES-Merging and DARE prune task updates based on coordinate-wise magnitude (TIES) or random dropout (DARE) in the parameter index basis. SVS operates in the continuous spectral (eigenvalue) basis of the weight matrices.
*   *Novelty:* This is a fundamentally different conceptual approach (spectral-domain low-pass filtering vs. spatial coordinate-basis pruning). However, as discussed in the paper, coordinate-basis pruning actually outperforms SVS ($77.98\%$ for TIES and $75.18\%$ for DARE vs. $74.83\%$ for SVS).

### 3. Task Singular Vectors (TSV-Compress) (Gargiulo et al., 2025) & SVD-Merging (Stoica et al., 2025)
*   *Delta:* TSV-Compress also applies SVD to task vectors and retains top singular components for compression and denoising. SVD-Merging uses whitening/alignment in weight space.
*   *Novelty:* The paper explicitly builds on the SVD formulation of TSV-Compress but claims two key novelty deltas:
    1.  **Theoretical Justification:** Providing formal scale-invariance proofs in normalized networks (explaining why global scale preservation is redundant).
    2.  **Dynamic Rank Allocation (Entropy-SVS):** Allocating dynamic ranks using Shannon spectral entropy instead of uniform ranks.

## Critical Novelty Assessment (The Reviewer's Perspective)
While the authors clearly define their novelty deltas in Section 2, **there is a severe lack of empirical evaluation against direct spectral baselines**. 

Specifically:
*   The authors **do not compare SVS against TSV-Compress (Gargiulo et al., 2025)** or **SVD-Merging (Stoica et al., 2025)** in Table 1 or any other experiments.
*   Without an empirical comparison against these closely related spectral/SVD-based model-merging methods, it is impossible to evaluate whether SVS's proposed "Entropy-SVS" or specific slicing choices offer a statistically significant empirical advantage over prior SVD-based methods. 
*   Is the performance of SVS at $k=128$ significantly different from TSV-Compress at equivalent rank? Does Entropy-SVS trace a superior Pareto frontier compared to a baseline that proportionally scales rank (e.g., $k_l = \rho \cdot \min(m, n)$)? The paper lacks these critical baselines, making the empirical novelty of SVS less convincing than its theoretical framing.
