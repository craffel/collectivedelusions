# 1. Summary of the Paper

## Main Topic and Goal
The paper tackles the problem of dynamic, test-time serving and merging of multiple task-specific Low-Rank Adaptation (LoRA) experts under heterogeneous input streams. Traditional serving of numerous adapters concurrently incurs significant system scheduling and memory overhead, while static parameter-space model merging suffers from "heterogeneity collapse" (accuracy degradation) when faced with conflicting tasks in real-time. To address these bottlenecks, the paper proposes a dynamic activation-space ensembling framework called **HyperMerge** (Hyperbolic Space Activation Routing and Fusion) that operates in a single, sample-wise forward pass.

## Proposed Approach
HyperMerge is built on the hypothesis that existing flat Euclidean representations ($\mathbb{R}^D$) in dynamic ensembling suffer from "representation crowding" and destructive inter-task "cross-talk" near the origin. It redefines the geometric substrate of model ensembling by projecting intermediate activation updates into a hyperbolic space—specifically, the Poincaré Ball model ($\mathbb{D}_c^D$)—where negative curvature provides exponential volume growth to segregate task manifolds and naturally accommodate hierarchical features.

The key components of the approach are:
1. **Differentiable Activation Mappings:** Using exponential ($\exp_{\mathbf{0}}^c$) and logarithmic ($\log_{\mathbf{0}}^c$) mapping functions to project intermediate LoRA expert updates into and out of the Poincaré Ball.
2. **Hyperbolic Distance and Routing:** Using hyperbolic geodesic distance to pre-computed task centroids to compute sample-wise ensembling coefficients via a temperature-scaled Softmax.
3. **Hyperbolic Centroid Alignment (HCA):** Projecting calibration embeddings to Klein space ($\mathbb{K}_c^D$) and calculating the Einstein midpoint to find mathematically optimal task centroids in closed form.
4. **Beltrami-Klein Symmetric Blending (BKSB):** Performing non-linear, permutation-invariant ensembling of expert activations by computing the Lorentz-weighted Einstein midpoint in Beltrami-Klein space before mapping back to the Poincaré Ball and the tangent space (Euclidean).
5. **Hyperbolic Out-of-Distribution Rejection (HOR):** A non-parametric outlier detector rejecting queries that lie beyond a distance threshold $\gamma_{\text{OOD}}$ from all task centroids.

## Key Findings
* **Task Segregation:** Projected activations are separated near the hyperbolic boundary, reducing representational crowding.
* **Stream Robustness:** Under fully heterogeneous streams, HyperMerge achieves 0.00% heterogeneity collapse, completely bypassing the need for heavy systems-level buffering like Micro-Batch Homogenization (MBH).
* **Competitive Accuracy:** Evaluated in a 14-layer Analytical Coordinate Sandbox, HyperMerge achieves **83.40% $\pm$ 5.15%** joint mean accuracy, performing on par with Euclidean ensembling baselines but with a mathematically consistent and order-independent formulation.
* **Sensitivity to Curvature:** Moderate negative curvature ($c \in [0.1, 0.5]$) is optimal for balancing task segregation and mapping distortion.

## Explicitly Claimed Contributions and Accompanying Evidence
1. **Challenge to the Euclidean Assumption:** The authors claim flat space causes representation crowding and cross-talk. *Evidence:* Demonstrated mathematically and verified in an "Overlapping Subspace Sandbox Regime," where projection to hyperbolic space increases distance between task centroids (from 1.6947 to 1.7315).
2. **HyperMerge Framework:** The first non-Euclidean model merging framework. *Evidence:* The theoretical formulation in Section 3 and implementation within the Analytical Coordinate Sandbox.
3. **HCA and BKSB Algorithms:** Formulas derived for computing closed-form centroids and performing permutation-invariant activation ensembling via Klein space Einstein midpoints. *Evidence:* Detailed derivations in Sections 3.5 and 3.6, with empirical validation of BKSB.
4. **Heterogeneity Immunity:** Showing that HyperMerge achieves absolute immunity to streaming patterns without sequential buffering bottlenecks. *Evidence:* Table 1, showing joint mean accuracy of 83.40% under both homogeneous and heterogeneous streams (0.00% collapse).
