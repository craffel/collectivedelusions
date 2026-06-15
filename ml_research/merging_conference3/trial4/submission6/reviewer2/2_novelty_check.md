# Novelty and Delta Assessment

## 1. Characterization of Novelty
The novelty of this paper is primarily **deconstructive and conceptual** rather than additive. Instead of introducing a more complex, hyperparameter-heavy model merging technique, the paper applies Occam's razor to subtract the coordinate-wise sign-voting and dominant sign-election heuristics that have become standard in SOTA sparse model merging (such as TIES-Merging and DARE). 

This type of "reductive novelty" is rare in deep learning literature but has **extraordinary significance for practitioners**. Demonstrating that a stripped-down, two-step pipeline (layer-wise magnitude pruning + standard direct addition) matches or slightly exceeds the performance of highly complex, multi-stage pipelines helps prevent over-engineering, simplifies codebases, reduces computational overhead, and clarifies the underlying mechanics of weight-space model merging.

## 2. The "Delta" from Prior Work
The paper positions itself directly against **TIES-Merging** (Yadav et al., 2023) and **DARE** (Yu et al., 2024):
- **TIES-Merging Delta:** TIES-Merging claims that resolving sign conflicts via coordinate-wise sign voting and dominant sign election is essential to prevent parameter interference. The "delta" of this paper is the empirical and theoretical proof that this step is redundant. When task vectors are pruned to a reasonable density, coordinate collisions are extremely rare, and when they do occur, they are self-resolving.
- **DARE Delta:** DARE relies on random dropout and scaling to preserve the expected value of parameters, but still uses TIES's sign consensus heuristic to fuse the parameters. This paper shows that magnitude-based pruning (which is simpler and deterministic) combined with global scaling tuning is sufficient, without needing stochastic operations or sign consensus.
- **Task Arithmetic (TA) Delta:** While STA is conceptually close to standard Task Arithmetic (since it uses direct linear addition), the "delta" is the uniform layer-wise magnitude pruning and the critical correction of the **update under-scaling confounder**. The authors demonstrate that sparse task vectors suffer from magnitude attenuation, and once this is corrected via global scaling coefficient tuning ($\lambda = 0.8$ instead of $0.3$) or analytical rescaling ($100/s$), sparse linear addition easily matches more complex heuristics.

## 3. Key Novel Concepts Introduced
1. **The Update Under-Scaling Confounder:** The paper is the first to explicitly identify that the drop in performance in un-tuned sparse merging is caused by a drop in weight-space energy (magnitude attenuation) due to pruning, rather than a representation breakdown.
2. **Symmetric Noise Filtering Perspective:** The authors propose a novel conceptual shift: rather than viewing magnitude pruning as a precursor to "sign conflict resolution," they mathematically ground it as an SGD gradient noise filter. Pruning selectively removes high-frequency, low-magnitude optimization noise, preserving the salient task-specific representations and preventing weight-space drift.
3. **Empirical and Theoretical Mask Overlap Rate:** The paper provides a clean, closed-form probability bound for coordinate overlap under sparsity ($(s/100)^2$) and backs it up with the first empirical measurement of overlap across Vision Transformer layers.
