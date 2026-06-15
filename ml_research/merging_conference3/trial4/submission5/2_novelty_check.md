# Reviewer Report: 2_novelty_check.md

## 2. Novelty and Positioning Check

### Relation to Prior Work
The paper positions itself relative to several major paradigms in weight-space model merging:
1. **Task Arithmetic (Ilharco et al., 2022):** The standard linear addition of fine-tuned task vectors. The paper addresses its primary limitation—destructive representational interference—using a simple spatial regularizer (magnitude pruning) rather than unconstrained addition.
2. **TIES-Merging (Yadav et al., 2023):** A multi-stage heuristic that prunes small weights, elects a dominant sign across tasks, and averages only sign-compatible parameters. The paper argues that TIES introduces unnecessary complexity, and that simple deterministic magnitude masking is sufficient and can even outperform sign election.
3. **DARE-Merging (Yu et al., 2023):** A stochastic weight dropping and rescaling technique. SG-TA replaces this stochastic behavior with a deterministic, magnitude-based thresholding approach.
4. **Decoupled Prune-then-Merge (P-then-M):** Applies static magnitude pruning to individual experts before merging.

### Novelty of Core Methodology and Extensions
While pruning weights before merging (P-then-M) is a known technique, the paper introduces several crucial extensions and conceptual clarifications that represent significant novelty:

1. **Global vs. Layer-wise Masking (GQ vs. LQ):** 
   * The paper contrasts **Global Quantile (GQ) masking** with **Layer-wise Quantile (LQ) masking**. Standard P-then-M is equivalent to LQ masking, which enforces a rigid, homogeneous budget across all layers. 
   * By proposing GQ, the authors allow the network to dynamically distribute its parameter budget. Layers that adapted heavily during fine-tuning (e.g., attention projection and deep MLP layers) naturally retain more active updates, while stable layers are heavily sparsified. The empirical finding that GQ substantially outperforms LQ (61.40% vs. 57.81%) is a valuable insight.
   * The authors are exceptionally honest about the connection between P-then-M and SG-TA (LQ), proving their empirical equivalence under optimized calibration. This high-signal honesty isolates **global budget flexibility** as the true driver of their method's success.

2. **Task Vector Magnitude Normalization (TV-Norm):**
   * Multi-task model merging is frequently plagued by task dominance, where one task with larger-magnitude updates overshadows others.
   * The authors propose a pre-masking vector scale normalization (TV-Norm) by dividing each task vector by its mean absolute parameter shift. While conceptually straightforward, the paper's deep empirical evaluation of TV-Norm—showing a dramatic performance boost on MNIST (from 36.74% to 53.70%) and a comprehensive control sweep over validation pool sizes ($N_{\text{val}} \in [10, 20, 50, 100]$) to stabilize its high-variance calibration sweeps—makes it a substantial, original contribution.

3. **Sigmoid-Gated Soft Masking (SG-TA-Soft):**
   * Hard binary masking introduces representational discontinuities in the validation loss landscape. To address this, the authors propose continuous sigmoid gating.
   * While sigmoid gating itself is a standard function, applying it to task vectors for landscape stabilization in post-hoc model merging is a novel and elegant idea. Crucially, the authors empirically demonstrate that soft-gating dramatically reduces standard deviation across calibration seeds by nearly 2x (from $\pm 1.39\%$ to $\pm 0.75\%$), proving that landscape smoothing stabilizes validation-based hyperparameter tuning.

4. **Non-Uniform Calibration (CS & RS):**
   * Standard merging relies on uniform keep-ratios $k$ and uniform scaling coefficients $\alpha$ across all tasks.
   * The authors relax this assumption, expanding the search space to $2T$ dimensions. To solve the exponential complexity bottleneck ($\mathcal{O}(P^T)$), they propose and validate a highly scalable Coordinate Search (CS) algorithm that reduces search complexity to linear $\mathcal{O}(T \cdot P_{\text{local}})$, proving its optimization stability and sequence independence.

### Summary of Novelty Assessment
The individual components of SG-TA are simple and build upon existing operations (pruning, scaling, sigmoids, coordinate descent). However, their **systematic, decoupled integration, the conceptual contrast between global and layer-homogeneous pruning budgets, and the deep empirical analyses (such as landscape smoothing via soft-gating and validation sweeps for normalized vectors)** represent high-quality, original contributions. The paper is well-positioned, does not overclaim, and provides clear, honest comparative baselines that help isolate the physical causes of model merging performance.
