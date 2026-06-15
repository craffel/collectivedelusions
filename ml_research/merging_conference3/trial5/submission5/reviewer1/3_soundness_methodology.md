# Soundness and Methodology Evaluation

## Clarity of the Description
The paper is exceptionally well-written, structured, and clear. Every equation is clearly formulated, with precise definitions of all variables, dimensions, and operational parameters:
- The standard weight-space model-merging paradigm is mathematically specified.
- The input state projection and normalization are fully defined.
- The complex wave cosine activation of QWS-Merge is mathematically deconstructed.
- The three proposed classical alternative routing channels (Linear, Tanh, Softmax) are explicitly formulated.
- The global classical Linear Router baseline is fully written out in both vector and element-wise forms, ensuring perfect structural and algebraic symmetry.
- The mathematical closed-form proof of layer-averaging collapse is highly accessible and rigorous.

## Appropriateness of Methods
The methodological choices made by the authors are outstanding and highly appropriate:
1. **The Isolating Coordinate Sandbox:** This is a crucial and justified methodological control. Evaluating dynamic routers on full-scale Vision Transformers or LLMs couples routing dynamics with weight-space coordinate conflicts. The sandbox isolates routing performance by setting coordinate conflict to zero, providing a clean, noise-free diagnostic coordinate space.
2. **Comprehensive Baselines:** The authors include the simplest baselines (Uniform Merging and a global classical Linear Router), which are frequently crippled or omitted in other publications.
3. **Rigorous Optimization Controls:** The optimizer (AdamW), training epochs (100), learning rate ($10^{-2}$), calibration split size (64 samples), and projection dimension ($d=K=4$) are realistic and properly controlled.
4. **Scale-Validation Pilot:** The inclusion of an empirical Vision-Language scale-validation pilot (merging three CLIP-ViT-B/16 models) bridges the gap between the sandbox and real weight-space manifolds, validating that sandbox insights hold on real parameters.

## Technical Correctness and Robustness to Objections
The paper is technically flawless and displays exceptional methodological hygiene. The authors have anticipated and systematically addressed every potential technical objection through rigorous appendices:
- **Optimization Bias Objection:** To address if QWS-Merge fails due to an inappropriate learning rate, they perform a learning rate sensitivity sweep (Appendix E). This sweep confirms that QWS-Merge is highly non-convex and structurally unstable, collapse-prone regardless of learning rate.
- **Statistical Significance Objection:** To address if findings are random anomalies, they execute a 5-seed robustness audit (Appendix H) with full dataset regeneration. Results show high consistency.
- **Orthogonal Subspace Bias Objection:** To address if the orthogonal sandbox prototypes artificially favor linear projections, they run a task-correlation sweep (Appendix H). Classical routers consistently dominate across all levels of task correlation.
- **Layer-Averaging Redundancy Objection:** To address if their findings are bound to head merging, they design a deep layer-by-layer weight-merging framework with no averaging of coefficients (Appendix I). Even under this setup, QWS-Merge catastrophically collapses (10.60% Joint Mean), while their regularized Softmax router survives (23.90% Joint Mean) and the global Linear Router continues to dominate (35.50% Joint Mean).
- **Mathematical Explanation of Gradient Dynamics:** The authors provide a brilliant mathematical deconstruction of backpropagation gradient dynamics under data scarcity (Appendix I) to explain these behaviors. They show how backpropagating through 14 sequential layers on a tiny 64-sample split creates extreme gradient instability, causing unconstrained layer-wise routers to blow up or saturate. Softmax simplex constraints act as a regularizing barrier that prevents gradient explosion, while the global Linear Router bypasses the backpropagation chain completely, achieving superior stability.

## Reproducibility
The reproducibility of this paper is excellent. The authors provide exact mathematical formulations of all routing modules, the dataset generation parameters (subspace dimensions, class prototypes, noise factors), optimization details (learning rate, epochs, weight decay parameters), and scale-validation specifics. An expert reader has all the necessary information to reconstruct the experimental setup and replicate the findings.
