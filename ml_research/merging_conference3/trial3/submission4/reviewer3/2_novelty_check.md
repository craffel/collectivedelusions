# 2_novelty_check.md: Novelty Assessment

## Key Novel Aspects
1. **Joint Co-Optimization Framework:** Existing test-time coefficient optimization methods like AdaMerging assume a fully dense merged model. ZipMerge introduces the novel formulation of co-optimizing continuous merging coefficients $\Lambda$ and binary magnitude-pruning boundaries (mask $M$) simultaneously. 
2. **Analysis of Optimizer-Trajectory Geometry under Sparsity:** The paper offers novel insights into how sparsity levels affect first-order (STE) and zero-order (ES) optimizers. Showing that moderate sparsity (50%) favors ES due to STE's gradient approximation variance, while high sparsity (80%) favors STE due to focused, variance-reduced active paths, is a significant analytical contribution.
3. **Overfitting-Optimizer Paradox in Model Merging:** The paper formalizes and empirically traces the transductive overfitting of unsupervised entropy minimization on tiny calibration datasets, demonstrating that lower entropy on 64 calibration images can actively damage out-of-domain generalization.
4. **Orthogonal Procrustes SVD Alignment for PEFT Merging:** The derivation and empirical validation of Orthogonal Procrustes SVD Alignment for LoRA adapters is highly novel and represents a significant practical breakthrough. It rotates independently learned adapter spaces into a mutually compatible coordinate system *post-hoc*, closing over 67.5% of the performance gap with zero-data and negligible sub-millisecond overhead on CPU.

## Delta from Prior Work
- **Delta from AdaMerging:** AdaMerging optimizes layer-wise coefficients on a tiny calibration set but leaves the model fully dense. ZipMerge integrates weight-pruning directly into the test-time adaptation loop, co-designing the coefficients with a target sparsity percentile threshold.
- **Delta from Standard Pruning (Wanda, SparseGPT, Magnitude Pruning):** Traditional pruning methods compress a single pre-trained model. ZipMerge prunes a multi-task merged model on-the-fly during test-time adaptation, negotiating complex spatial and representational conflicts that standard single-model pruning ignores.
- **Delta from TIES-Merging:** TIES-Merging filters and prunes individual task-specific vectors *post-hoc* to resolve sign conflicts before merging. It does not compress the model to a target global physical footprint for deployment. ZipMerge prunes the merged model to a target global sparsity ratio and adapts the coefficients.
- **Delta from standard PEFT Merging:** The paper's SVD-based Orthogonal Procrustes Alignment is a substantial mathematical and empirical step forward. Unlike prior methods that require data or training to align representations, Procrustes resolves the coordinate basis mismatch *analytically* in a training-free manner.

## Characterization of Novelty
The novelty in this paper is **significant**. 

While the individual components—model merging, magnitude pruning, test-time adaptation, and Orthogonal Procrustes—are established in their respective areas, the paper's contribution is far from incremental:
- It connects these components to address a highly realistic system bottleneck (deploying merged models on edge hardware).
- It reveals critical empirical boundaries and failure modes (catastrophic collapse, Overfitting-Optimizer Paradox) that challenge standard academic assumptions.
- It proposes and validates simple, extremely low-overhead, and high-yield solutions (Prune-then-Merge as a spatial regularizer, Orthogonal Procrustes SVD Alignment) that provide immediate, actionable utility to real-world deployment. This makes the novelty exceptionally valuable for practical engineering.
