# Peer Review

## Summary of the Paper
The paper presents **HyperMerge** (Hyperbolic Space Activation Routing and Fusion), a non-Euclidean framework designed for dynamic, test-time ensembling of multiple task-specific Low-Rank Adaptation (LoRA) adapters. The authors challenge the ubiquitous "Euclidean assumption" in modular deep learning, arguing that flat coordinate systems ($\mathbb{R}^D$) suffer from "representation crowding" near the origin and fail to respect the multi-scale, hierarchical nature of deep feature representations. 

To overcome these limitations, HyperMerge projects the activation-space updates of task-specific adapters into the Poincaré Ball model of hyperbolic space ($\mathbb{D}_c^D$), which features constant negative curvature. Within this hyperbolic workspace, the paper introduces two primary procedures:
1. **Hyperbolic Centroid Alignment (HCA):** Computes mathematically optimal task-specific reference centroids (Fréchet means) on a tiny calibration split by projecting Poincaré coordinates to Klein space and computing the Lorentz-weighted Einstein midpoint in closed form.
2. **Beltrami-Klein Symmetric Blending (BKSB):** Performs online, sample-wise activation ensembling by converting projected Poincaré expert updates into Klein coordinates, computing a Lorentz-weighted Einstein midpoint using dynamic routing weights (obtained via temperature-scaled softmax over geodesic distances to centroids), and mapping the merged result back to Poincaré and then to flat Euclidean space.

Additionally, the paper implements **Hyperbolic Out-of-Distribution Rejection (HOR)**, which uses a Poincaré geodesic distance threshold to reject out-of-distribution queries at Layer 0.

---

## Strengths and Weaknesses

### Strengths
1. **Mathematical Rigor and Elegance:** The use of the Beltrami-Klein model's Einstein midpoint formula to resolve the non-associativity of sequential Möbius additions in the Poincaré Ball is a mathematically sophisticated and highly elegant algebraic formulation.
2. **Excellent Clarity and Presentation:** The paper is exceptionally well-written, with precise mathematical notation, clean and consistent equations, and a cohesive, easy-to-follow narrative. Section 3.8 (distortion analysis) and Section 3.9 (numerical safeguards) demonstrate high technical diligence.
3. **Comprehensive Baselines:** The authors compare HyperMerge against a wide hierarchy of static and dynamic model merging baselines, including state-of-the-art Euclidean methods like SABLE and SPS-ZCA.
4. **Detailed Parametric Ablations:** The paper includes valuable ablation studies on the impact of curvature $c$, out-of-distribution rejection threshold $\gamma_{\text{OOD}}$, and a crowded "Overlapping Subspace" sandbox regime.

### Weaknesses
1. **The "Small-Norm" Boundary Contradiction:** The central motivation of HyperMerge is that flat Euclidean space suffers from representation crowding near the origin, which can be resolved by exploiting the exponential volume growth of hyperbolic space near its boundary. However, LoRA updates $E_{k,b}^{(l)}$ are small displacement vectors, exhibiting very small norms in practice ($\|E\|_2 \ll 1$). Consequently, their Poincaré projections reside entirely in the vicinity of the origin, where hyperbolic space is locally flat (approximately Euclidean). The exponential volume growth near the boundary is entirely inactive, invalidating the core motivation of using negative curvature to segregate representations.
2. **First-Order Equivalence to Euclidean Ensembling:** As proved via Taylor expansion below, BKSB mathematically converges to standard flat Euclidean linear ensembling up to first-order and second-order terms. The hyperbolic correction only appears at the cubic order $\mathcal{O}(c \|E\|_2^3)$. Since LoRA updates are small-norm, this correction is microscopically small, explaining why HyperMerge is empirically outperformed by, or performs on par with, flat Euclidean baselines.
3. **Lack of Real-World Evaluation:** The evaluation is conducted entirely on a synthetic "14-layer Analytical Coordinate Sandbox." While this sandbox is useful for theoretical exploration, it is highly idealized. To demonstrate practical utility, the method must be evaluated on real-world deep neural networks (e.g., LLaMA, ViT) with standard natural language or computer vision datasets.
4. **Unresolved Quantitative Reporting Discrepancy:** There is a major reporting discrepancy between Table 1 and Table 2. Table 1 reports a joint mean accuracy of **83.40%** for HyperMerge with $c=0.1$. However, Table 2 reports an accuracy of **89.30%** for the exact same configuration ($c=0.1$). This significant 5.90% discrepancy is left completely unaddressed in the text.
5. **Overstated Heterogeneity Robustness:** The authors present "absolute immunity to stream heterogeneity" as a unique contribution of HyperMerge. However, SABLE and SPS-ZCA also achieve exactly 0.00% heterogeneity collapse. This robustness is a general property of **all sample-wise activation-space ensembling methods** that route and fuse representations dynamically on a single forward pass, rather than a unique benefit of hyperbolic geometry.

---

## Soundness
**Rating: Fair**

**Justification:**
While the mathematical derivations and algebraic steps are technically correct on their own terms, the paper exhibits a fundamental conceptual inconsistency between its motivation and its application regime:

### 1. Proof of First-Order Equivalence to Euclidean Ensembling
Let the adapter updates in the tangent space at the origin be $E_k \in T_{\mathbf{0}}\mathbb{D}_c^D$, and let the routing weights be $\alpha_k$ (where $\sum_k \alpha_k = 1$). Under a moderate curvature $c$ and small-norm updates $\|E\|_2 \ll 1$:

1. **Poincaré Projection (Exponential Map):**
   Using $\tanh(x) = x - \frac{1}{3}x^3 + \mathcal{O}(x^5)$, we have:
   $$\mathbf{v}_k = \exp_{\mathbf{0}}^c(E_k) = E_k - \frac{c}{3}\|E_k\|_2^2 E_k + \mathcal{O}(c^2\|E_k\|_2^4 E_k)$$

2. **Klein Projection:**
   Using $(1+x)^{-1} = 1 - x + \mathcal{O}(x^2)$, we have:
   $$\mathbf{w}_k = \frac{2 \mathbf{v}_k}{1 + c\|\mathbf{v}_k\|_2^2} = 2E_k - \frac{8}{3}c\|E_k\|_2^2 E_k + \mathcal{O}(c^2\|E_k\|_2^4 E_k)$$

3. **Lorentz Factor:**
   $$\gamma_k = \frac{1}{\sqrt{1 - c\|\mathbf{w}_k\|_2^2}} = 1 + 2c\|E_k\|_2^2 + \mathcal{O}(c^2\|E_k\|_2^4)$$

4. **Einstein Midpoint Blending:**
   Expanding the numerator and denominator of $\mathbf{w}_{\text{merged}} = \frac{\sum_k \alpha_k \gamma_k \mathbf{w}_k}{\sum_k \alpha_k \gamma_k}$:
   $$\sum_{k=1}^K \alpha_k \gamma_k \mathbf{w}_k = 2\sum_{k=1}^K \alpha_k E_k - \frac{4}{3}c\sum_{k=1}^K \alpha_k \|E_k\|_2^2 E_k + \mathcal{O}(c^2\|E\|_2^5)$$
   $$\sum_{k=1}^K \alpha_k \gamma_k = 1 + 2c\sum_{k=1}^K \alpha_k \|E_k\|_2^2 + \mathcal{O}(c^2\|E\|_2^4)$$
   Dividing these terms:
   $$\mathbf{w}_{\text{merged}} = 2\sum_{k=1}^K \alpha_k E_k - 4c\left(\sum_{j=1}^K \alpha_j \|E_j\|_2^2\right)\left(\sum_{k=1}^K \alpha_k E_k\right) - \frac{4}{3}c\sum_{k=1}^K \alpha_k \|E_k\|_2^2 E_k + \mathcal{O}(c^2\|E\|_2^5)$$

5. **Poincaré Map Back:**
   $$\mathbf{v}_{\text{merged}} = \frac{\mathbf{w}_{\text{merged}}}{1 + \sqrt{1 - c\|\mathbf{w}_{\text{merged}}\|_2^2}} = \frac{\mathbf{w}_{\text{merged}}}{2} \left(1 + c \left\|\sum_{j=1}^K \alpha_j E_j\right\|_2^2\right) + \mathcal{O}(c^2\|E\|_2^5)$$
   $$\mathbf{v}_{\text{merged}} = \sum_{k=1}^K \alpha_k E_k + c\left\|\sum_{j=1}^K \alpha_j E_j\right\|_2^2 \sum_{k=1}^K \alpha_k E_k - 2c\left(\sum_{j=1}^K \alpha_j \|E_j\|_2^2\right)\left(\sum_{k=1}^K \alpha_k E_k\right) - \frac{2}{3}c\sum_{k=1}^K \alpha_k \|E_k\|_2^2 E_k + \mathcal{O}(c^2\|E\|_2^5)$$

6. **Logarithmic Map Back to Euclidean Space:**
   Using $\text{artanh}(x) = x + \frac{1}{3}x^3 + \mathcal{O}(x^5)$, we map back to the tangent space:
   $$E_{\text{merged}} = \log_{\mathbf{0}}^c(\mathbf{v}_{\text{merged}}) = \mathbf{v}_{\text{merged}} + \frac{c}{3}\|\mathbf{v}_{\text{merged}}\|_2^2 \mathbf{v}_{\text{merged}} + \mathcal{O}(c^2\|E\|_2^5)$$
   $$E_{\text{merged}} = \sum_{k=1}^K \alpha_k E_k + \frac{4}{3}c\left\|\sum_{j=1}^K \alpha_j E_j\right\|_2^2 \sum_{k=1}^K \alpha_k E_k - 2c\left(\sum_{j=1}^K \alpha_j \|E_j\|_2^2\right)\left(\sum_{k=1}^K \alpha_k E_k\right) - \frac{2}{3}c\sum_{k=1}^K \alpha_k \|E_k\|_2^2 E_k + \mathcal{O}(c^2\|E\|_2^5)$$

This rigorous derivation proves that up to first-order and second-order terms, the HyperMerge blending operator is mathematically equivalent to flat Euclidean linear ensembling:
$$E_{\text{merged}} = \sum_{k=1}^K \alpha_k E_k + \mathcal{O}(c \|E\|_2^3)$$
Since $\|E\|_2 \ll 1$, the cubic correction term is microscopic, explaining the lack of empirical hyperbolic advantage.

### 2. Tangent Space Hybrid Contradiction
The authors assume the flat Euclidean space $\mathbb{R}^D$ is the tangent space at the origin $T_{\mathbf{0}}\mathbb{D}_c^D$. However, the actual propagated state through the network is $h_b^{(l)} = h_{base, b}^{(l)} + E_{\text{merged}, b}^{(l)}$. Since both $h_{base}$ and $E_{\text{merged}}$ are flat Euclidean vectors, the actual intermediate activations propagated to the next layer reside strictly in flat Euclidean space $\mathbb{R}^D$.
Therefore, any "hierarchical taxonomies" or "multi-scale internal feature manifolds" encoded by the deep network still reside in a flat Euclidean space during forward propagation. The claim that HyperMerge "accommodates representation-level hierarchies inside deep neural networks" is a logical error, as the hyperbolic mapping is transient and does not change the geometric substrate of the network's main representation stream.

---

## Presentation
**Rating: Excellent**

**Justification:**
The overall presentation is outstanding. The manuscript is clearly written and beautifully structured. The mathematical derivations of both HCA and BKSB are presented precisely and unambiguously, and the notation is consistent. The authors write with exceptional prose and build their arguments smoothly, making the paper highly accessible. Section 3.9 on numerical safeguards and clamping to a bounding radius of $\frac{1}{\sqrt{c}} - \epsilon$ shows exceptional diligence and care for implementing hyperbolic operations stably.

---

## Significance
**Rating: Fair**

**Justification:**
The practical significance of the proposed method is low. In Table 1, SABLE (Early Routing) outperforms HyperMerge (84.03% vs. 83.40%). In Table 3 (Overlapping Subspace Sandbox Regime), SABLE again outperforms HyperMerge (77.98% vs. 76.62% / 76.50% tuned). 
Since HyperMerge introduces significant mathematical complexity (exponential/logarithmic projections, Klein coordinate maps, and Lorentz factor evaluations) but performs worse than a simple flat-space ensembling baseline, there is little incentive for deep learning practitioners to adopt it. However, the theoretical significance of demonstrating how to compute permutation-invariant, closed-form barycenters in model ensembling using Klein-space algebra is of moderate academic interest.

---

## Originality
**Rating: Good**

**Justification:**
The introduction of the Beltrami-Klein coordinate representation and its Einstein midpoint formula to resolve the non-associativity and non-commutativity of sequential Möbius additions in model ensembling is highly original. The authors identify a legitimate algebraic obstacle (ordering bias in sequential Möbius addition) and solve it using an elegant, closed-form barycentric formulation. However, because the resulting operator is first-order equivalent to flat Euclidean linear ensembling, the practical novelty of the hyperbolic space's volume growth is inactive during inference.

---

## Overall Recommendation
**Rating: 3: Weak reject**

**Justification:**
HyperMerge is an exceptionally well-written paper that introduces a highly elegant and mathematically sophisticated algebraic framework. The application of Beltrami-Klein Einstein midpoints to resolve the ordering bias of sequential Möbius additions in hyperbolic spaces is a notable theoretical contribution.

However, the weaknesses of the current submission outweigh its merits:
1. **Conceptual Contradiction:** There is a fundamental contradiction between the paper's core motivation (exploiting hyperbolic volume growth to segregate task representations near the boundary) and the application regime (small-norm LoRA updates that project entirely near the origin, where hyperbolic space is locally Euclidean).
2. **First-Order Equivalence:** The BKSB operator is mathematically shown via Taylor expansion to be equivalent to flat Euclidean linear ensembling up to first-order and second-order terms, rendering the "negative curvature advantage" inactive.
3. **No Empirical Superiority:** Flat-space Euclidean ensembling (SABLE) consistently outperforms HyperMerge in both standard (84.03% vs. 83.40%) and overlapping (77.98% vs. 76.62%) regimes.
4. **Quantitative Discrepancy:** The unexplained 5.90% discrepancy between Table 1 and Table 2 for $c=0.1$ must be resolved.
5. **Lack of Real-World Evaluation:** The evaluation is limited to a synthetic sandbox, failing to demonstrate practical generalizability on real large-scale models (like LLaMA or ViT).

**Constructive Suggestions for Revision:**
- **Incorporate the Taylor Expansion Analysis:** Acknowledge the small-norm flat-space limit of BKSB, and discuss why the hyperbolic correction remains very small.
- **Explain and Correct the Table 1 vs. Table 2 Discrepancy:** Clarify why the joint mean accuracy for $c=0.1$ is reported as 83.40% in Table 1 but as 89.30% in Table 2.
- **Real-World Benchmarks:** Evaluate HyperMerge on actual, non-synthetic datasets using pre-trained backbones (e.g., fine-tuning and ensembling LoRA experts for LLaMA on GLUE or ViT on domain-adaptation datasets).
- **Soften the Claims:** Align the claims with the empirical results. Instead of claiming to "shatter geometric limits," focus on the paper's true strength: presenting a mathematically rigorous, order-independent ensembling operator that serves as a continuous deformation of standard Euclidean ensembling.
