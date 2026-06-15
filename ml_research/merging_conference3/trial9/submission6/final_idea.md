# Idea Proposal: Lie-Algebraic Homotopical Model Merging via Grassmannian Geodesic Blending (Lie-MM)

## 1. Persona Alignment
As **The Theorist**, we approach the model ensembling and routing problem through a mathematically rigorous, Riemannian geometric lens. We argue that standard linear interpolation in parameter or activation space (e.g., standard Task Arithmetic, SABLE, or SPS-ZCA) implicitly assumes a flat Euclidean geometry. This flat assumption is fundamentally flawed: pre-trained representation manifolds and their low-rank projection bases lie on curved submanifolds, such as the **Grassmannian Manifold** $\mathcal{G}(d, D)$ of $d$-dimensional subspaces in $\mathbb{R}^D$. 

Linear blending of projection matrices $P_{\text{merged}} = \sum_k \alpha_k P_k$ violates the idempotency condition ($P_{\text{merged}}^2 \ne P_{\text{merged}}$), causing projected coordinate collapse and representation distortion. Lie-MM replaces flat heuristics with mathematically rigorous **Grassmannian Geodesic Blending (GGB)**, solving for ensembling parameters on the curved manifold itself. Our method is backed by formal Lie-algebraic exp/log mappings, ensuring that the merged operators are provably correct orthogonal projection operators at all times.

---

## 2. Core Techniques
1. **Grassmannian Representation Space:** We model task-specific subspaces as points on the Grassmannian manifold $\mathcal{G}(d, D)$, where each task $k$ is represented by an orthogonal basis matrix $V_k \in \mathbb{R}^{D \times d}$ ($V_k^T V_k = I_d$).
2. **Riemannian Logarithmic Mapping:** We map each task's projection basis $V_k$ onto the tangent space of a reference subspace $Y_0$ using the Grassmannian logarithm map $\log_{Y_0}(V_k)$, yielding a flat vector space of tangent matrices.
3. **Tangent Space Weighted Homotopy:** We perform linear interpolation of these tangent vectors using dynamic routing coefficients $\alpha_k$ computed from a temperature-calibrated Gibbs policy.
4. **Riemannian Exponential Mapping:** We project the blended tangent vector back onto the Grassmannian manifold using the Grassmannian exponential map $\exp_{Y_0}(H)$, ensuring that the merged basis $Y_{\text{merged}}$ remains strictly orthogonal ($Y_{\text{merged}}^T Y_{\text{merged}} = I_d$).
5. **Grassmannian Geodesic Projection:** We project the intermediate activations $z_b$ onto the merged Grassmannian geodesic subspace to yield the ensembled representation.

---

## 3. Mathematical Formulation

### A. The Grassmannian Manifold $\mathcal{G}(d, D)$
The Grassmannian $\mathcal{G}(d, D)$ is the set of all $d$-dimensional linear subspaces of $\mathbb{R}^D$. An element of $\mathcal{G}(d, D)$ is represented by an orthogonal matrix $Y \in \mathbb{R}^{D \times d}$ such that $Y^T Y = I_d$. The orthogonal projection matrix onto the subspace is given by $P = Y Y^T \in \mathbb{R}^{D \times D}$.

### B. Riemannian Logarithm Map on $\mathcal{G}(d, D)$
Let $Y_0, Y_1 \in \mathbb{R}^{D \times d}$ be two orthogonal projection bases representing points on the Grassmannian. The Riemannian logarithm map $\log_{Y_0}(Y_1)$ computes the tangent vector $H \in \mathbb{R}^{D \times d}$ at $Y_0$ along the geodesic from $Y_0$ to $Y_1$:
1. Compute the SVD of the orthogonal complement projection:
   $$ (I_D - Y_0 Y_0^T) Y_1 (Y_0^T Y_1)^{-1} = U \Sigma V^T $$
2. Compute the principal angles:
   $$ \Theta = \arctan(\Sigma) \in \mathbb{R}^{d \times d} $$
3. The tangent vector is given by:
   $$ H = \log_{Y_0}(Y_1) = U \Theta V^T \in \mathbb{R}^{D \times d} $$
Note that $Y_0^T H = 0$, satisfying the tangent space constraint.

### C. Riemannian Exponential Map on $\mathcal{G}(d, D)$
Given a tangent vector $H \in \mathbb{R}^{D \times d}$ at $Y_0$ with SVD $H = U \Theta V^T$, the Riemannian exponential map $\exp_{Y_0}(H)$ projects $H$ back onto the Grassmannian:
   $$ Y(t) = \exp_{Y_0}(t H) = Y_0 V \cos(t \Theta) V^T + U \sin(t \Theta) V^T $$
For $t=1$, $Y(1) \in \mathbb{R}^{D \times d}$ is strictly orthogonal, representing the geodesic destination subspace.

### D. Single-Step Grassmannian Barycenter Approximation
For multiple subspaces $\{V_k\}_{k=1}^K$ with ensembling weights $\{\alpha_k\}_{k=1}^K$ ($\sum_k \alpha_k = 1$), the exact Karcher mean is the subspace $Y_{\text{merged}}$ minimizing:
   $$ Y_{\text{merged}} = \arg\min_{Y \in \mathcal{G}(d, D)} \sum_{k=1}^K \alpha_k d^2_{\text{Grassmann}}(Y, V_k) $$
To execute this efficiently within a fast deep neural network forward pass, we employ a **Single-Step Riemannian Barycenter Approximation**:
1. Select the reference subspace $Y_0 = V_{\hat{k}}$ where $\hat{k} = \arg\max_k \alpha_k$ is the dominant task expert.
2. Map all other bases to the tangent space at $Y_0$:
   $$ H_k = \log_{Y_0}(V_k) \quad \forall k \in \{1, \dots, K\} $$
3. Blend the tangent matrices in the flat tangent space:
   $$ H_{\text{merged}} = \sum_{k=1}^K \alpha_k H_k $$
4. Project back to the manifold via the exponential map:
   $$ Y_{\text{merged}} = \exp_{Y_0}(H_{\text{merged}}) $$
5. Construct the strictly idempotent projection matrix:
   $$ P_{\text{merged}} = Y_{\text{merged}} Y_{\text{merged}}^T $$

---

## 4. Architecture Specifications
*   **Routing Layer ($l_{\text{route}}$):** Located at Layer 3 of the Vision Transformer or ResNet backbone, where representations $z_b \in \mathbb{R}^D$ are extracted.
*   **Subspace Dimension ($d$):** The principal component projection dimension, set to $d = 8$ or $d = 16$.
*   **Gibbs Temperature Routing:** Temperatures $\tau_k > 0$ are optimized using a log-temperature parameterization $w_k = \ln \tau_k$.
*   **Inputs:** $z_b \in \mathbb{R}^D$.
*   **Intermediate Representations:**
    *   Coordinate Vector: $\mathbf{e}_b = [\|V_1^T z_b\|_2, \dots, \|V_K^T z_b\|_2]^T \in \mathbb{R}^K$.
    *   Routing Coefficients: $\alpha_k = \exp(e_{k,b}/\tau_k) / \sum_j \exp(e_{j,b}/\tau_j)$.
    *   Merged Subspace Basis: $Y_{\text{merged}} \in \mathbb{R}^{D \times d}$ computed via the Single-Step Grassmannian Barycenter.
*   **Final Output:** Projected feature representation $z_{b, \text{projected}} = P_{\text{merged}} z_b = Y_{\text{merged}} (Y_{\text{merged}}^T z_b) \in \mathbb{R}^D$.

---

## 5. Baselines
We will evaluate Lie-MM against the following appropriate baselines:
1. **SABLE (Sample-wise Activation Blending):** Flat Euclidean activation blending without manifold projection constraints. This acts as our primary empirical baseline.
2. **SPS-ZCA (Single-Pass Centroid Alignment):** Stateless layer-wise projection baseline using standard centroid similarities.
3. **Unregularized ERM Router:** Traditional linear router trained without PAC or geometric regularization, demonstrating the necessity of the Grassmannian manifold constraint.
4. **PAC-ZCA:** The learning-theoretic temperature-only Gibbs routing baseline, verifying if curved manifold projection outperforms flat projections.

---

## 6. Step-by-Step Interaction
1. **Forward Propagation:** The input sample $x_b$ is propagated through the frozen backbone up to layer $l_{\text{route}}$, yielding hidden representation $z_b \in \mathbb{R}^D$.
2. **Coordinate Extraction:** $z_b$ is projected onto pre-extracted task subspaces to obtain the $K$-dimensional coordinate vector $\mathbf{e}_b$.
3. **Coefficient Evaluation:** The log-temperatures $\mathbf{w}$ are used to compute the dynamic Gibbs routing coefficients $\alpha_k$.
4. **Grassmannian Blending:** The bases $V_k$ are blended along Grassmannian geodesics using the Single-Step Barycenter to produce $Y_{\text{merged}}$.
5. **Activation Projection:** The hidden feature $z_b$ is projected onto the merged subspace via $z_{b, \text{projected}} = Y_{\text{merged}} Y_{\text{merged}}^T z_b$.
6. **Subsequent Expert Processing:** The projected activation is fed into the downstream expert adapters (e.g., parallel LoRAs), which process the representation to output the final predictions.
