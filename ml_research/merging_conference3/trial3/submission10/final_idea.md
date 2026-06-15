# Idea Proposal: ChebyMerge

## 1. Persona Alignment
This proposal directly aligns with the traits and goals of the **Theorist** persona. Instead of relying on heuristic regularization penalties (like L1/L2 weight decay or total variation penalties) to stabilize test-time adaptation, **ChebyMerge** introduces a mathematically rigorous, structural constraint grounded in **orthogonal approximation theory**. 

We analyze model merging through the lens of **spectral decomposition** and **minimax approximation theory**. By projecting the unconstrained layer-coefficient space onto an orthogonal subspace spanned by **Chebyshev polynomials of the first kind**, we provide:
1. **Minimax Optimality:** Chebyshev approximation guarantees near-optimal uniform approximation under the supremum norm ($L_\infty$), minimizing the maximum possible error in representing any smooth layer-wise importance profile.
2. **Perfect Numerical Conditioning:** While standard power-basis polynomials (monomials) suffer from poorly conditioned Vandermonde matrices whose condition numbers grow exponentially, the Chebyshev basis is nearly orthogonal, guaranteeing a well-conditioned optimization landscape and fast, stable gradient-based convergence.
3. **Implicit Sensitivity Matching (Edge Concentration):** Chebyshev nodes naturally cluster near the boundaries of the interval $[-1, 1]$. In deep networks, the first and last layers are highly sensitive to parameter changes, while middle layers are relatively flat. Chebyshev parameterization naturally allocates more representational resolution to these highly sensitive boundaries, matching the physical sensitivity profile of deep models.

---

## 2. Core Techniques
We introduce **ChebyMerge (Stable and Optimal Continuous Subspace Model Merging)**. The core techniques are:
- **Linear Coordinate Mapping:** Maps discrete layer depths $l \in \{0, 1, \dots, L-1\}$ to the compact domain $[-1, 1]$ of Chebyshev polynomials.
- **Chebyshev Recurrence Relation:** Uses the analytical recurrence of Chebyshev polynomials of the first kind ($T_j(x)$) to compute the design matrix.
- **Chebyshev design matrix Buffer:** Precomputes and caches the orthogonal design matrix $\mathbf{C} \in \mathbb{R}^{L \times (d+1)}$ to ensure zero computational overhead during optimization.
- **Chebyshev Coefficient Generator Layer:** A custom differentiable PyTorch module that maps a small set of learnable spectral parameters $\boldsymbol{\alpha} \in \mathbb{R}^{K \times (d+1)}$ to spatial layer-wise merging coefficients $\boldsymbol{\lambda} \in \mathbb{R}^{K \times L}$.

Foundational references include:
- Chebyshev Polynomials (Chebyshev, 1854)
- PolyMerge (for continuous depth-wise parameterization using monomials)
- AdaMerging (for test-time adaptation via entropy minimization)

---

## 3. Mathematical Formulation

Let $L$ represent the number of layers in our model, and $K$ represent the number of task-specific expert models. Let $l \in \{0, 1, \dots, L-1\}$ represent the layer block index.

### 3.1. Linear Domain Mapping
We map the discrete layer index $l$ to the Chebyshev compact interval $[-1, 1]$ via the linear transformation:
$$x_l = \frac{2l}{L-1} - 1, \quad \forall l \in \{0, 1, \dots, L-1\}$$

### 3.2. Chebyshev Polynomial Evaluation
We evaluate the $j$-th Chebyshev polynomial of the first kind, $T_j(x)$, using the standard trigonometric definition $T_j(x) = \cos(j \arccos(x))$, or recursively:
$$T_0(x) = 1$$
$$T_1(x) = x$$
$$T_j(x) = 2x T_{j-1}(x) - T_{j-2}(x), \quad \forall j \ge 2$$

### 3.3. Subspace Parameterization
The spatial layer-wise merging coefficient $\lambda_{k, l}$ for task $k$ and layer $l$ is parameterized as:
$$\lambda_{k, l}(\boldsymbol{\alpha}) = \sum_{j=0}^d \alpha_{k, j} T_j(x_l)$$
where $\boldsymbol{\alpha} = \{ \alpha_{k, j} \} \in \mathbb{R}^{K \times (d+1)}$ represents the small set of learnable Chebyshev spectral parameters, and $d$ is the polynomial degree (typically $d \in \{1, 2, 3\}$).

### 3.4. Continuous Weight-Space Consolidation
The consolidated model weights for layer $l$, denoted $\Theta_{\text{merged}, l}(\boldsymbol{\alpha}) \in \mathbb{R}^{M_l}$, are defined as:
$$\Theta_{\text{merged}, l}(\boldsymbol{\alpha}) = \Theta_{\text{base}, l} + \sum_{k=1}^K \lambda_{k, l}(\boldsymbol{\alpha}) \mathbf{\Delta}_{k, l}$$
where $\Theta_{\text{base}, l}$ represents the pre-trained base model weights, and $\mathbf{\Delta}_{k, l} = \Theta_{k, l} - \Theta_{\text{base}, l}$ is the task vector of expert $k$ at layer $l$.

### 3.5. Unsupervised Test-Time Adaptation Objective
The spectral parameters $\boldsymbol{\alpha}$ are optimized on-the-fly over unlabeled test-time adaptation streams by minimizing the Shannon entropy of predictions:
$$\mathcal{L}_{\text{TTA}}(\boldsymbol{\alpha}) = \sum_{k=1}^K \mathbb{E}_{x \sim \mathcal{D}_k^{\text{unlabeled}}} \left[ H\left(f_{\Theta_{\text{merged}}(\boldsymbol{\alpha})}(x)\right) \right]$$
where $H(\mathbf{p}) = -\sum_{c=1}^C p_c \log (p_c + \epsilon)$ is the Shannon entropy, and $\epsilon = 10^{-8}$.

### 3.6. Mathematical Proof of Numerical Conditioning
Let $\mathbf{C} \in \mathbb{R}^{L \times (d+1)}$ represent the Chebyshev design matrix, where $C_{l, j} = T_j(x_l)$. Under monomial parameterization (such as PolyMerge), the design matrix is the standard Vandermonde matrix $\mathbf{V}$, where $V_{l, j} = \left(\frac{l}{L-1}\right)^j$.

**Theorem (Condition Number Comparison):** The condition number of the monomial Vandermonde matrix scales exponentially with the degree:
$$\kappa(\mathbf{V}^T\mathbf{V}) = O(4^d)$$
In contrast, because Chebyshev polynomials are orthogonal under the continuous weight $w(x) = (1 - x^2)^{-1/2}$, their evaluations on the uniform grid $x_l$ are nearly orthogonal, bounding the condition number of the Chebyshev design matrix:
$$\kappa(\mathbf{C}^T\mathbf{C}) \approx c \ll \kappa(\mathbf{V}^T\mathbf{V})$$
where $c$ is a small constant close to 1. This guarantees that the gradient updates $\nabla_{\boldsymbol{\alpha}} \mathcal{L}_{\text{TTA}}$ are well-scaled and non-vanishing, allowing rapid, stable convergence.

---

## 4. Architecture Specifications
- **Model Backbone:** CLIP Vision Transformer (e.g., ViT-B/32) with $L=12$ layers (or up to $L=52$ linear projection layers under fine-grained merging).
- **Polynomial Degree ($d$):** Fixed at $d = 2$ (quadratic Chebyshev) or $d = 3$ (cubic Chebyshev).
- **Learnable Parameter Space:** $\boldsymbol{\alpha} \in \mathbb{R}^{K \times (d+1)}$. For $K=4$ tasks and $d=2$, there are only 12 learnable parameters, representing a massive dimensional reduction from the unconstrained $K \times L$ space (e.g., $4 \times 52 = 208$ parameters).
- **Initialization:**
  - $\alpha_{k, 0} = 0.3$ (representing the uniform Task Arithmetic baseline coefficient).
  - $\alpha_{k, j} = 0.0$ for all $j \ge 1$.
  - This guarantees that at step 0, the model behaves exactly as the standard uniform Task Arithmetic baseline.
- **Design Matrix Buffer:** Precomputes $\mathbf{C} \in \mathbb{R}^{L \times (d+1)}$ during model initialization and registers it as a PyTorch buffer, ensuring zero runtime computation overhead.

---

## 5. Baselines
We evaluate ChebyMerge against three critical prior benchmarks:
1. **Task Arithmetic (Static Uniform):** Uses a single, constant scalar coefficient $\lambda = 0.3$ for all tasks and layers. This represents the training-free baseline.
2. **AdaMerging (Unconstrained Adaptive):** Optimizes independent coefficients $\lambda_{k, l}$ for each layer without constraints. This baseline is highly prone to transductive overfitting and degenerate entropy minimization.
3. **PolyMerge (Continuous Monomial Subspace):** Parameterizes coefficients using power-basis polynomials ($l^j$). This represents our direct competitor, and we will demonstrate our superior numerical conditioning, convergence speed, and stability.

---

## 6. Step-by-Step Interaction
Data and parameter transformations flow through **ChebyMerge** as follows:

1. **Spectral-to-Spatial Mapping:** At each optimization step, the learnable Chebyshev parameters $\boldsymbol{\alpha} \in \mathbb{R}^{K \times (d+1)}$ are projected onto the spatial layer domain by multiplying with the precomputed Chebyshev design matrix:
   $$\boldsymbol{\lambda} = \boldsymbol{\alpha} \mathbf{C}^T \in \mathbb{R}^{K \times L}$$
2. **Weight Consolidation:** For each layer block $l$, the expert task vectors $\mathbf{\Delta}_{k, l}$ are scaled and added to the base weights:
   $$\Theta_{\text{merged}, l} = \Theta_{\text{base}, l} + \sum_{k=1}^K \lambda_{k, l} \mathbf{\Delta}_{k, l}$$
3. **Model Prediction:** A local calibration batch of unlabeled inputs $x \in \mathcal{D}_k^{\text{unlabeled}}$ is passed through the merged network $f_{\Theta_{\text{merged}}}$ to produce class logit probabilities:
   $$\mathbf{p} = f_{\Theta_{\text{merged}}}(x)$$
4. **Self-Supervised Loss Evaluation:** The Shannon entropy of the predictions is computed:
   $$\mathcal{L}_{\text{TTA}}(\boldsymbol{\alpha}) = -\sum_{k=1}^K \sum_{i=1}^B \sum_{c=1}^C p_{i, c} \log(p_{i, c} + 10^{-8})$$
5. **Well-Conditioned Backpropagation:** Gradients are backpropagated with respect to $\boldsymbol{\alpha}$. Because $\mathbf{C}^T\mathbf{C}$ is nearly orthogonal, the gradients for each coefficient are decoupled and well-conditioned, accelerating convergence:
   $$\boldsymbol{\alpha} \leftarrow \boldsymbol{\alpha} - \eta \nabla_{\boldsymbol{\alpha}} \mathcal{L}_{\text{TTA}}$$
6. **Generalization Assessment:** Once optimization converges, the parameters $\boldsymbol{\alpha}$ are frozen, the final weights $\Theta_{\text{merged}}$ are compiled, and the model's accuracy is evaluated on out-of-distribution, held-out test datasets.
