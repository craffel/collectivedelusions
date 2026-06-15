# 3. Soundness and Methodology

## Clarity of the Description
The methodology of HyperMerge is described with excellent mathematical clarity. The equations for the Poincaré Ball model (metric tensor, exponential map, logarithmic map), Möbius algebraic primitives (addition and scalar multiplication), and the specific procedures for Hyperbolic Centroid Alignment (HCA) and Beltrami-Klein Symmetric Blending (BKSB) are presented precisely and unambiguously. This level of detail is highly commendable and allows for easy conceptual tracing of the algebraic steps.

## Appropriateness of Methods and Mathematical Correctness
The algebraic operations used in the paper are mathematically correct and standard in the literature of Möbius gyrovector spaces and hyperbolic geometry. The use of the Beltrami-Klein model's Einstein midpoint formula to compute a closed-form, permutation-invariant barycenter is mathematically sound. Since geodesics in the Beltrami-Klein model are straight lines, the Einstein midpoint indeed serves as a valid and computationally efficient Fréchet mean, which resolves the non-associativity of sequential Möbius additions in the Poincaré Ball.

---

## Potential Technical Flaws and Conceptual Inconsistencies (Theoretical Scrutiny)

### 1. The "Small-Norm" Flatness Contradiction
The central motivation of HyperMerge is that flat Euclidean space is unsuited for modular deep learning because representations crowd around the origin, causing inter-task cross-talk. The authors propose to exploit the *exponential volume growth* of hyperbolic space to segregate task manifolds near the boundary of the Poincaré Ball.

However, this argument contains a fundamental physical and mathematical contradiction when applied to Low-Rank Adapters (LoRA):
- In the Poincaré Ball model with negative curvature $-c$, the volume of a sphere of radius $r$ scales as $V(r) \propto e^{(D-1)r}$ only for large radii. Near the origin ($\mathbf{x} \approx \mathbf{0}$), the volume scales polynomially, and the metric tensor is approximately Euclidean ($g_{\mathbf{x}} \approx 4 \mathbf{I}_D$ when $c=1$).
- LoRA updates $E_{k,b}^{(l)}$ are small displacement vectors, typically initialized with scaling factor $\alpha/r$ and exhibiting very small norms in practice ($\|E_{k,b}^{(l)}\|_2 \ll 1$).
- When these small updates are projected via the exponential map:
  $$\mathbf{v}_{k,b} = \exp_{\mathbf{0}}^c\left(E_{k,b}^{(l)}\right) \approx E_{k,b}^{(l)}$$
  the resulting points project to coordinates extremely close to the origin, where $\|\mathbf{v}_{k,b}\|_2 \ll 1/\sqrt{c}$.
- Consequently, the projected activations reside entirely in the locally flat, near-Euclidean region of the Poincaré Ball. The exponential volume growth of the manifold's boundary is **entirely inactive**. Thus, negative curvature cannot physically segregate these representation manifolds or resolve coordinate crowding in any meaningful way compared to flat Euclidean coordinates.

### 2. Taylor Expansion Proof of First-Order Equivalence to Euclidean Ensembling
We can formally prove that HyperMerge's BKSB operator is a negligible, high-order non-linear perturbation of standard Euclidean linear ensembling when the adapter updates are small. 

Let the adapter updates in the tangent space at the origin be $E_k \in T_{\mathbf{0}}\mathbb{D}_c^D$ (omitting sample index $b$ and layer index $l$ for clarity). Let the routing weights be $\alpha_k$ (such that $\sum_{k=1}^K \alpha_k = 1$). 

1. **Exponential Map Projection:**
   Using the Taylor expansion $\tanh(x) = x - \frac{1}{3}x^3 + \mathcal{O}(x^5)$, we expand the Poincaré projected coordinate $\mathbf{v}_k$:
   $$\mathbf{v}_k = \exp_{\mathbf{0}}^c(E_k) = E_k - \frac{c}{3}\|E_k\|_2^2 E_k + \mathcal{O}(c^2\|E_k\|_2^4 E_k)$$

2. **Poincaré to Klein Mapping:**
   Using the expansion $(1+x)^{-1} = 1 - x + \mathcal{O}(x^2)$, we project to Beltrami-Klein coordinates $\mathbf{w}_k$:
   $$\mathbf{w}_k = \frac{2 \mathbf{v}_k}{1 + c\|\mathbf{v}_k\|_2^2} = 2E_k - \frac{8}{3}c\|E_k\|_2^2 E_k + \mathcal{O}(c^2\|E_k\|_2^4 E_k)$$

3. **Lorentz Factor expansion:**
   $$\gamma_k = \frac{1}{\sqrt{1 - c\|\mathbf{w}_k\|_2^2}} = 1 + 2c\|E_k\|_2^2 + \mathcal{O}(c^2\|E_k\|_2^4)$$

4. **Einstein Midpoint Blending:**
   Let's expand the weighted Einstein midpoint in Klein space, $\mathbf{w}_{\text{merged}} = \frac{\sum_k \alpha_k \gamma_k \mathbf{w}_k}{\sum_k \alpha_k \gamma_k}$.
   - **Numerator expansion:**
     $$\sum_{k=1}^K \alpha_k \gamma_k \mathbf{w}_k = 2\sum_{k=1}^K \alpha_k E_k - \frac{4}{3}c\sum_{k=1}^K \alpha_k \|E_k\|_2^2 E_k + \mathcal{O}(c^2\|E\|_2^5)$$
   - **Denominator expansion:**
     $$\sum_{k=1}^K \alpha_k \gamma_k = 1 + 2c\sum_{k=1}^K \alpha_k \|E_k\|_2^2 + \mathcal{O}(c^2\|E\|_2^4)$$
   - **Barycenter quotient:**
     $$\mathbf{w}_{\text{merged}} = \left( 2\sum_{k=1}^K \alpha_k E_k - \frac{4}{3}c\sum_{k=1}^K \alpha_k \|E_k\|_2^2 E_k \right) \left( 1 - 2c\sum_{j=1}^K \alpha_j \|E_j\|_2^2 \right) + \mathcal{O}(c^2\|E\|_2^5)$$
     $$\mathbf{w}_{\text{merged}} = 2\sum_{k=1}^K \alpha_k E_k - 4c\left(\sum_{j=1}^K \alpha_j \|E_j\|_2^2\right)\left(\sum_{k=1}^K \alpha_k E_k\right) - \frac{4}{3}c\sum_{k=1}^K \alpha_k \|E_k\|_2^2 E_k + \mathcal{O}(c^2\|E\|_2^5)$$

5. **Klein to Poincaré Mapping:**
   $$\mathbf{v}_{\text{merged}} = \frac{\mathbf{w}_{\text{merged}}}{1 + \sqrt{1 - c\|\mathbf{w}_{\text{merged}}\|_2^2}} = \frac{\mathbf{w}_{\text{merged}}}{2} \left(1 + c \left\|\sum_{j=1}^K \alpha_j E_j\right\|_2^2\right) + \mathcal{O}(c^2\|E\|_2^5)$$
   $$\mathbf{v}_{\text{merged}} = \sum_{k=1}^K \alpha_k E_k + c\left\|\sum_{j=1}^K \alpha_j E_j\right\|_2^2 \sum_{k=1}^K \alpha_k E_k - 2c\left(\sum_{j=1}^K \alpha_j \|E_j\|_2^2\right)\left(\sum_{k=1}^K \alpha_k E_k\right) - \frac{2}{3}c\sum_{k=1}^K \alpha_k \|E_k\|_2^2 E_k + \mathcal{O}(c^2\|E\|_2^5)$$

6. **Logarithmic Map Projection:**
   Using the expansion $\text{artanh}(x) = x + \frac{1}{3}x^3 + \mathcal{O}(x^5)$, we project back to Euclidean space:
   $$E_{\text{merged}} = \log_{\mathbf{0}}^c(\mathbf{v}_{\text{merged}}) = \mathbf{v}_{\text{merged}} + \frac{c}{3}\|\mathbf{v}_{\text{merged}}\|_2^2 \mathbf{v}_{\text{merged}} + \mathcal{O}(c^2\|E\|_2^5)$$
   Substituting $\mathbf{v}_{\text{merged}}$:
   $$E_{\text{merged}} = \sum_{k=1}^K \alpha_k E_k + \frac{4}{3}c\left\|\sum_{j=1}^K \alpha_j E_j\right\|_2^2 \sum_{k=1}^K \alpha_k E_k - 2c\left(\sum_{j=1}^K \alpha_j \|E_j\|_2^2\right)\left(\sum_{k=1}^K \alpha_k E_k\right) - \frac{2}{3}c\sum_{k=1}^K \alpha_k \|E_k\|_2^2 E_k + \mathcal{O}(c^2\|E\|_2^5)$$

### Conclusion of Proof
As $\|E\|_2 \to 0$, the ensembling operator reduces exactly to:
$$E_{\text{merged}} = \sum_{k=1}^K \alpha_k E_k + \mathcal{O}(c \|E\|_2^3)$$
This proof shows that HyperMerge is **mathematically equivalent to flat Euclidean linear ensembling up to first-order and second-order terms**. The non-linear hyperbolic correction only appears at the cubic order $\mathcal{O}(c \|E\|_2^3)$. Since LoRA updates are small-norm, this correction term is microscopic. This explains why HyperMerge's empirical performance is virtually identical to (or slightly worse than) the Euclidean baselines.

### 3. Tangent Space Hybrid Contradiction
The authors assume that the flat Euclidean space $\mathbb{R}^D$ is the tangent space at the origin $T_{\mathbf{0}}\mathbb{D}_c^D$. However, the actual propagated state through the network is $h_b^{(l)} = h_{base, b}^{(l)} + E_{\text{merged}, b}^{(l)}$. Since both the base model representation $h_{base}$ and the merged update $E_{\text{merged}}$ are flat Euclidean vectors, the actual intermediate activations propagated to the next layer reside strictly in flat Euclidean space $\mathbb{R}^D$.
Therefore, any "hierarchical taxonomies" or "multi-scale internal feature manifolds" encoded by the deep network still reside in a flat Euclidean space during forward propagation. The claim that HyperMerge "accommodates representation-level hierarchies inside deep neural networks" is a logical error, as the hyperbolic mapping is transient and does not change the geometric substrate of the network's main representation stream.

---

## Reproducibility
The mathematical formulations are extremely detailed, and the pseudo-code provided in the text makes the implementation details highly reproducible. However, the evaluation is conducted entirely on a simulated 14-layer Analytical Coordinate Sandbox with synthetic coordinate partitions, which does not allow for a direct assessment of how this method behaves in actual, real-world deep learning pipelines (e.g., LLaMA, ViT). This synthetic setup limits the empirical reproducibility in standard deep learning benchmarks.
