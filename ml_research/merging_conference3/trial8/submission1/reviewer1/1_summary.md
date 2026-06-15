# 1. Summary of the Paper

## Main Topic and Approach
The paper introduces **HyperMerge** (Hyperbolic Space Activation Routing and Fusion), a non-Euclidean model merging framework designed for combining multiple task-specific Low-Rank Adaptation (LoRA) adapters at test time. The authors argue that existing dynamic ensembling methods, which operate in flat Euclidean space ($\mathbb{R}^D$), suffer from "representation crowding" near the coordinate origin and fail to respect the multi-scale, hierarchical nature of deep feature representations. 

To resolve these limitations, HyperMerge projects the activation updates of task-specific adapters into the Poincaré Ball model of hyperbolic space ($\mathbb{D}_c^D$), which features constant negative curvature. In this hyperbolic workspace, the paper introduces two primary mathematical procedures:
1. **Hyperbolic Centroid Alignment (HCA):** Computes robust, mathematically optimal task-specific reference centroids (Fréchet means) on a small calibration split by mapping Poincaré coordinates to Beltrami-Klein space, computing the Lorentz-weighted Einstein midpoint in closed-form, and projecting back to Poincaré space.
2. **Beltrami-Klein Symmetric Blending (BKSB):** Performs online, sample-wise activation ensembling by converting the projected Poincaré expert updates into Klein coordinates, computing a Lorentz-weighted Einstein midpoint using dynamic routing weights (obtained via temperature-scaled softmax over hyperbolic distances to task centroids), and projecting the merged state back to Poincaré space and then to flat Euclidean space.

Additionally, the paper introduces **Hyperbolic Out-of-Distribution Rejection (HOR)**, which uses a hyperbolic geodesic distance threshold to reject out-of-distribution queries at Layer 0.

## Key Findings and Empirical Evidence
The paper evaluates HyperMerge within a 14-layer, 192-dimensional Analytical Coordinate Sandbox with $K=4$ simulated task experts representing MNIST, Fashion-MNIST, CIFAR-10, and SVHN under two stream configurations (Homogeneous and Heterogeneous):
- **Homogeneous and Heterogeneous Accuracy:** HyperMerge achieves a flatline joint mean accuracy of **83.40% ± 5.15%** under both homogeneous and heterogeneous streams, completely avoiding "heterogeneity collapse" (0.00% collapse) without requiring stateful micro-batch buffering (unlike PFSR + MBH, which drops to 80.97% without MBH and requires complex buffering to reach 92.97%).
- **Comparison with Euclidean Baselines:** Under standard settings, HyperMerge performs on par with state-of-the-art Euclidean baselines (SPS-ZCA at 83.05% ± 4.95% and SABLE at 84.03% ± 5.15%). Under a highly crowded "Overlapping Subspace Sandbox Regime," HyperMerge achieves **76.62% ± 3.96%** (at $c=0.1$) and **76.50% ± 3.36%** (tuned), compared to SABLE (77.98% ± 2.12%) and SPS-ZCA (77.32% ± 1.98%).
- **Ablation Studies:**
  - Varying curvature $c$ from 0.001 to 1.0 shows that joint mean accuracy peaks at $c = 0.5$ (91.00%).
  - HOR achieves an OOD detection F1-score of **98.40%** at a distance threshold of $\gamma_{\text{OOD}} = 1.5$.
  - In the overlapping regime, the hyperbolic distance between the Task 0 and Task 1 centroids expands from 1.6947 in Euclidean space to 1.7315 in the Poincaré Ball ($c=0.1$), suggesting physical segregation on the manifold.

## Explicitly Claimed Contributions
The authors explicitly claim the following contributions:
1. **Geometric Challenge to the Euclidean Assumption:** Demonstrating that flat space ensembling leads to representation crowding and destructive cross-talk under heterogeneous deployment.
2. **Non-Euclidean Model Merging Paradigm:** Introducing the first model merging framework utilizing Poincaré Ball geometry to natively capture representation-level hierarchies.
3. **Rigorous Geometric Formulations (HCA and BKSB):** Providing closed-form, permutation-invariant, and mathematically consistent ensembling using Klein-space Einstein midpoints and Möbius algebra.
4. **Empirical Robustness:** Achieving 83.40% joint mean accuracy and absolute immunity (0.00% collapse) to stream heterogeneity on the Analytical Coordinate Sandbox.
