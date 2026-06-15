# Systematic Critique - Step 2: Novelty and Literature Positioning Check

## 1. Theoretical and Conceptual Novelty
The core conceptual contribution of shifting dynamic model ensembling and test-time activation routing from flat Euclidean space to non-Euclidean hyperbolic space (the Poincaré Ball) is highly original. In the rapidly growing literature of Parameter-Efficient Fine-Tuning (PEFT) ensembling, dynamic test-time routing, and activation merging (such as AdaMerging, PFSR, SABLE, and SPS-ZCA), representation spaces are invariably modeled in $\mathbb{R}^D$. Introducing negative curvature represents a creative and mathematically rigorous paradigm shift.

Specifically, the application of:
1. **Beltrami-Klein Einstein Midpoints** (Hyperbolic Centroid Alignment, HCA) to compute closed-form task barycenters (centroids) during calibration, and
2. **Beltrami-Klein Symmetric Blending (BKSB)** using Lorentz-weighted Einstein midpoints to perform non-linear, permutation-invariant activation ensembling,
represents an elegant adaptation of hyperbolic operations (originally proposed by Ganea et al., 2018 for standard feed-forward networks) to the modular deep learning framework.

## 2. Relationship to Prior Work
The paper positions itself very clearly in Section 2, reviewing:
* **PEFT and dynamic serving:** (LoRA, S-LoRA, Punica)
* **Static merging:** (Task Arithmetic, TIES-merging, DARE, ZipIt!)
* **Dynamic test-time adaptation:** (AdaMerging, PFSR, SABLE, SPS-ZCA)
* **Hyperbolic deep learning:** (Poincaré embeddings, HNNs, HGNNs)

The literature positioning is highly logical and clearly outlines the progression of the field from static parameter merging to dynamic activation blending, culminating in the need to resolve flat Euclidean "representation crowding" and "heterogeneity collapse".

## 3. Shortcomings in Theoretical Novelty and Contextualization
While the mathematical formulation is clean and elegant, there are notable conceptual gaps in how the novelty is motivated and justified:

1. **Hierarchy Confusion (Task-level vs. Representation-level):**
   The paper motivates the use of hyperbolic space by stating that it "naturally accommodates hierarchical task structures and power-law distribution spreads" and resolves "taxonomic and power-law mismatch". However, the tasks being merged—MNIST (handwritten digits), Fashion-MNIST (clothing items), CIFAR-10 (natural objects), and SVHN (street numbers)—are completely disjoint, flat, and independent tasks. They do not form a nested taxonomic hierarchy.
   While hyperbolic space is highly suited for tree-like hierarchical data (e.g., WordNet or biological taxonomies), the paper fails to explain why disjoint flat tasks would exhibit hierarchical structures requiring negative curvature. The justification conflates *internal representation-level hierarchy* within a single deep model and *task-level hierarchy* across a multi-task suite. To make the motivation rigorous, the authors should focus exclusively on the scale-invariant feature extraction hierarchy of deep networks, rather than making claims about task-level hierarchies.
2. **Heavy Reliance on Existing Mathematical Frameworks:**
   Although the application to model merging is novel, the core mathematical machinery (conformal factors, geodesic distances, exponential/logarithmic maps, Möbius addition, Möbius scalar multiplication, Poincaré-Klein conversions, Einstein midpoints) is adopted directly from Ganea et al. (2018) and standard hyperbolic geometry literature without significant algebraic or geometric modifications. The novelty lies in the pipeline configuration (the routing and ensembling wrappers) rather than any new hyperbolic algebraic primitives.
3. **Implicit Flat Assumption in Hybrid Setup:**
   The paper claims that the base model operates in flat Euclidean space while adapter updates are projected to the Poincaré Ball. The authors describe this hybrid design as "geometrically consistent" because the Euclidean representation space of the base model acts as the tangent space at the origin. However, if the underlying base model is trained under a flat Euclidean assumption, its intermediate representations $h_b^{(l-1)}$ are Euclidean. Linear projections of these representations via low-rank adapter matrices (which also live in Euclidean space) are fundamentally flat. Forcing these flat vector updates into a curved space and then mapping them back via log maps to perform flat addition with the base activation is more of an ad-hoc non-linear warping function than a geometrically consistent Riemannian flow. The paper does not address this conceptual gap.
