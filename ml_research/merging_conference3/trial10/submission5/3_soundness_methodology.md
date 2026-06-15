# Soundness and Methodology Check

The methodology of this paper is highly sound, mathematically rigorous, and carefully detailed, but contains minor terminological and structural issues that warrant critique.

## 1. Mathematical Formulations and Derivations
- **Spherical Norm Preservation:** The paper derives the spherical update step (Slerp) and mathematically verifies that the updated state remains exactly on the unit hypersphere:
  $$\|\mathbf{s}_t^{(l)}\|_2^2 = \cos^2(\theta) \|\mathbf{s}_t^{(l-1)}\|_2^2 + \sin^2(\theta) \|\mathbf{u}_t^{(l)}\|_2^2 + 2\cos(\theta)\sin(\theta) \langle \mathbf{s}_t^{(l-1)}, \mathbf{u}_t^{(l)} \rangle = 1$$
  This proof is correct and elegant.
- **Positive Orthant Persistence:** In Section 3.2 and Appendix A.2, the authors provide a formal proof of sign symmetry and positive orthant persistence. Because the state is initialized uniformly in the positive orthant ($1/\sqrt{K}\mathbf{1} > 0$) and the geodesic updates occur along the shortest great-circle path, the trajectory remains strictly within $\mathbb{S}^{K-1}_+$.
- **Terminological Nitpick ("Unitary"):** The use of the word "Unitary" in Unitary Geodesic Routing is a minor abuse of mathematical terminology. In linear algebra and quantum mechanics, a "unitary operator" (or unitary matrix) satisfies $U^* U = I$ and preserves inner products. The authors' routing state $\mathbf{s}_t^{(l)}$ is a "unit vector" (or unit-norm vector), not an operator. While the authors explicitly clarify this in Section 3.2 ("the term 'Unitary' ... refers strictly to the unit-norm constraint of our ensembling state vector ... rather than any quantum unitary operators"), using "Unitary" to mean "unit-norm vector" is terminologically non-standard and could confuse readers. "Spherical Geodesic Routing" or "Unit-Norm Geodesic Routing" would be mathematically more accurate.

## 2. Structural Analysis of Spatial-Temporal Coupling
- **Layer Mismatch Issue:** The spatial-temporal coupling initializes the *first adapted layer* of the current query using the *final layer's* state of the previous query:
  $$\mathbf{s}_t^{(L_{\text{frozen}})} = \mathbf{s}_{t-1}^{(L)}$$
  In deep networks, different layers capture different levels of feature abstraction (lower layers capture syntactic/local features, while higher layers capture semantic/global features). Initializing a low-level routing state with a high-level routing decision introduces a representational mismatch. The authors do not justify why they couple the final layer to the first adapted layer, rather than doing a layer-wise coupling ($\mathbf{s}_t^{(l)} = \mathbf{s}_{t-1}^{(l)}$) or propagating the state within matching layers. 

## 3. Control-Theoretic Soundness of Torque-Driven Agility
The physics-inspired Torque-Driven Agility is modeled as a first-order non-linear dynamical system with non-linear damping:
- By scaling the angular velocity directly with torque ($\theta = \eta \phi$) and omitting second-order acceleration terms, the system behaves as a first-order control loop.
- Control theory confirms that first-order loops are inherently stable and completely free from overshoot, oscillations, or kinetic momentum accumulation. This explains why UGR responds instantly to task boundaries without the "unwinding" lag of Momentum-Merge or ChemMerge.

## 4. High-Dimensional Scaling and Concentration of Measure
A potential theoretical bottleneck of hyperspherical geometry is the *concentration of measure* in high dimensions ($K \gg 10^2$), where random unit vectors are almost always orthogonal, causing the alignment cosine to degenerate and torque to remain constant.
- The authors address this directly in Section 3.5.
- They propose a top-$k$ active expert sub-manifold routing strategy (elaborated in Appendix A.4) where geodesic updates are restricted to the local $(k-1)$-dimensional spherical sub-manifold of the active experts, maintaining the sensitivity of the alignment cosine.

## Methodology Conclusion
The methodology is exceptionally strong and intellectually creative. Every mathematical claim is rigorously proven, and potential high-dimensional bottlenecks are addressed. However, the use of the term "Unitary" is non-standard, and the cross-layer spatial-temporal coupling represents a structural choice that lacks rigorous comparative justification.
