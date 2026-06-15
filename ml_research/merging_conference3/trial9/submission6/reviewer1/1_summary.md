# 1. Summary of the Paper

## Main Topic and Goal
The paper addresses the challenge of ensembling or merging task-specific low-rank projection operators (e.g., in parameter-efficient fine-tuning like LoRA) inside deep neural networks. Specifically, it focuses on the representation-space distortion and coordinate/norm decay that occurs when multiple task-specific projection matrices are linearly blended.

## Proposed Approach
To prevent "projected coordinate collapse" and "eigenvalue shrinkage" associated with flat Euclidean linear blending, the authors propose a framework named **Continuous Riemannian-Geometric Homotopical Model Merging via Grassmannian Geodesic Blending (C-Lie-MM)**. The method treats each projection operator as a point on the Grassmannian manifold $\mathcal{G}(d, D)$ and performs blending within the Riemannian geometry framework:
1. **Reference Space**: Uses the offline Karcher mean (approximated via the projection metric centroid from SVD of the average projection matrix) as a fixed reference point $Y_0 \in \mathcal{G}(d, D)$.
2. **Tangent Mapping**: Maps task-specific projection bases to the tangent space of $Y_0$ using a custom Cut-Locus-Aware Grassmannian logarithm map.
3. **Homotopic Interpolation**: Performs weighted linear combinations of these tangent matrices in the flat tangent space using sample-specific routing weights.
4. **Exponential Projection**: Projects the blended tangent vector back to the manifold via the Grassmannian exponential map to guarantee that the merged operator is symmetric, idempotent, and of rank $d$.
5. **Gibbs Routing**: Determines routing weights sample-wise using a temperature-calibrated softmax over projected activation norms.
6. **SVD-Free Polynomial Expansion**: Approximates the exponential map with Taylor or Chebyshev polynomial expansions to bypass online SVD during serving on edge/low-resource hardware.

## Key Findings & Claimed Contributions
1. **Mathematical Formalization of Coordinate Collapse**: Proves that flat linear averaging of projection matrices leads to eigenvalue shrinkage, causing exponential representation decay in deep networks.
2. **C-Lie-MM Framework**: Leverages exact Grassmannian logarithm and exponential maps on $\mathcal{G}(d, D)$ to preserve projection properties strictly ($\Delta_{\text{idem}} \approx 10^{-7}$).
3. **Empirical Validation**: Tests the method on a 14-layer Coordinate Sandbox. Uniform merging drops to 25.00% classification accuracy under severe manifold overlap, whereas C-Lie-MM maintains 70.30% accuracy, outperforming SABLE and PAC-ZCA baselines.
4. **Immunity to Heterogeneity Collapse**: Shows that sample-wise dynamic geodesic blending allows identical performance in both homogeneous and heterogeneous (mixed) streaming workloads.
5. **Out-of-Distribution Robustness**: Argues that under high uncertainty (routing weights distribute evenly), the Karcher mean acts as a central projection space to preserve shared semantics instead of collapsing.
6. **LoRA Integration Blueprint**: Evaluates simulated GLUE tasks styled after RoBERTa-Large. C-Lie-MM achieves 97.0% average accuracy, avoiding the coordinate collapse that degrades task arithmetic (49.8%) and TIES-Merging / SABLE (55.0%).
