# Paper Summary: HyperMerge

## Main Topic and Goal
The paper introduces **HyperMerge** (Hyperbolic Space Activation Routing and Fusion), a dynamic model ensembling framework designed to fuse task-specific model adapters (like LoRA) in activation space at test-time. Its primary goal is to address the limitations of static parameter-space merging (which suffers from inter-task interference under heterogeneous streams) and existing dynamic Euclidean ensembling methods. 

## Proposed Approach
HyperMerge shifts the geometric workspace of dynamic ensembling from flat Euclidean space ($\mathbb{R}^D$) to the **Poincaré Ball model of Hyperbolic Space** ($\mathbb{D}_c^D$). The core elements of the proposed approach are:
1. **Differentiable Activation Mappings:** Activations/updates are projected from Euclidean space to the Poincaré Ball via the exponential map at the origin ($\exp_{\mathbf{0}}^c$) and mapped back via the logarithmic map ($\log_{\mathbf{0}}^c$).
2. **Hyperbolic Centroid Alignment (HCA):** To compute task-specific centroids on a tiny calibration split, the method projects Poincaré activations to the Beltrami-Klein model ($\mathbb{K}_c^D$), computes the Einstein midpoint (weighted by Lorentz factors), and maps the resulting centroid back to Poincaré coordinates.
3. **Hyperbolic Distance-Based Routing:** At test-time, incoming queries are routed based on their geodesic distance in the Poincaré Ball to the pre-computed task centroids using a temperature-scaled Softmax.
4. **Beltrami-Klein Symmetric Blending (BKSB):** To ensemble expert updates symmetrically, the unscaled updates are mapped to Klein coordinates, blended via a Lorentz-weighted Einstein midpoint (using the routing weights), projected back to the Poincaré Ball, and then mapped back to Euclidean space to be added residually to the base model.
5. **Hyperbolic Out-of-Distribution Rejection (HOR):** A non-parametric outlier detector rejects queries whose minimum geodesic distance to any centroid exceeds a threshold.

## Key Findings and Claims
1. **Representation Crowding:** The authors claim that flat Euclidean space forces representations to crowd near the origin, leading to destructive cross-talk under heterogeneous streams. They claim that hyperbolic space, with its exponential volume growth, naturally segregates task manifolds.
2. **Superiority/Robustness:** The authors claim HyperMerge is immune to stream heterogeneity (0.00% heterogeneity collapse) and achieves a joint mean accuracy of **83.40% ± 5.15%** on a 14-layer Analytical Coordinate Sandbox, outperforming static Uniform merging and PFSR (without MBH).
3. **Equivalence in Overlapping Regimes:** Under a highly crowded "Overlapping Subspace Sandbox Regime," they claim HyperMerge remains highly competitive, segregating centroids further apart than flat Euclidean space.

## Explicitly Claimed Contributions
1. Challenging the Euclidean assumption in model merging and demonstrating that flat space introduces representation crowding.
2. Proposing HyperMerge, the first non-Euclidean model merging framework using Poincaré Ball geometry.
3. Formulating HCA and BKSB using Klein-space algebra to achieve permutation-invariant dynamic ensembling.
4. Demonstrating 83.40% accuracy on a simulated sandbox, eliminating heterogeneity collapse.
