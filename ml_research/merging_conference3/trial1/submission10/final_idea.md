# FoldMerge: Neural Origami via Differentiable Weight-Space Diffeomorphisms for Multi-Task Manifold Folding

## 1. Persona Alignment
This technical implementation directly embodies the core philosophy of **The Visionary** persona described in `persona.md`. Instead of settling for incremental tweaks to existing Euclidean interpolation methods or standard manifold projections, **FoldMerge (Neural Origami)** fundamentally rethinks a foundational, unquestioned assumption of model merging: *that the path between multiple task-specific models in parameter space must be linear.*

Current state-of-the-art merging frameworks (like SyMerge, OrthoMerge, or SAIM) perform parameter aggregation using linear combinations in some representation space (Euclidean space or Lie algebra). However, due to the non-convex, curved, and disjoint nature of task basins of attraction, linear interpolation inevitably forces the merged model through high-loss barriers, degrading performance. 

FoldMerge takes a bold, paradigm-shifting risk: it discards linear averaging entirely. We propose that model merging should be treated as a **non-linear manifold folding process**. By training a highly expressive, differentiable weight-space diffeomorphism (implemented via normalizing flows), we geometrically bend and warp the disjoint coordinate systems of different task-specific parameter spaces. This "folds" their separate low-loss basins of attraction together into a single, unified basin. FoldMerge prioritizes extreme novelty and potential impact, drawing inspiration from geometric origami, differential topology, and generative coordinate networks to introduce a completely fresh perspective to deep learning model fusion.

---

## 2. Core Techniques
FoldMerge introduces several core algorithmic mechanisms that depart from traditional model merging:

1. **Differentiable Weight-Space Diffeomorphism ($g_\phi$):**
   We define a coordinate transformation network $g_\phi: \mathbb{R}^d \to \mathbb{R}^d$ parameterized by $\phi$, which is mathematically guaranteed to be a diffeomorphism (invertible and differentiable with a differentiable inverse $g_\phi^{-1}$). This is parameterized using a cascade of RealNVP coupling layers.
2. **Origami Weight Representation Space:**
   Instead of performing merging in the original parameter space, we map task-specific parameters $\{ \theta_k \}_{k=1}^K$ into a latent "Origami Space" via $z_k = g_\phi(\theta_k)$.
3. **Barycentric Merging in Origami Space:**
   In Origami Space, we compute the linear average (barycenter) of the warped coordinates: $\bar{z} = \sum_k \lambda_k z_k$.
4. **Inverse Diffeomorphism Decoding ($g_\phi^{-1}$):**
   The merged parameters are decoded by mapping the barycenter back via the inverse diffeomorphism: $\theta_{MTL} = g_\phi^{-1}(\bar{z})$.
5. **Local Jacobian Volume Regularization:**
   To ensure the coordinate warp remains smooth and preserves crucial geometric properties (such as the hyperspherical energy highlighted in OrthoMerge), we introduce a Frobenius-norm Jacobian penalty that forces the transformation to be locally volume-preserving and to deviate minimally from an identity mapping except where necessary to align the basins.

---

## 3. Mathematical Formulation

Let $f(x; \theta)$ be a model parameterized by task-specific layers/adapters $\theta \in \mathbb{R}^d$. We are given $K$ expert models fine-tuned on different tasks with weights $\theta_1, \theta_2, \ldots, \theta_K$.

Let $g_\phi: \mathbb{R}^d \to \mathbb{R}^d$ be our diffeomorphism parameterized by $\phi$.
The representation of the $k$-th task parameters in the folded Origami Space is:
$$z_k = g_\phi(\theta_k)$$

The merged representation in Origami Space is:
$$\bar{z} = \sum_{k=1}^K \lambda_k z_k$$
where $\lambda_k \geq 0$ and $\sum_{k=1}^K \lambda_k = 1$ are the merging coefficients.

The final merged parameter vector $\theta_{MTL}$ is reconstructed via:
$$\theta_{MTL}(\phi, \lambda) = g_\phi^{-1}\left( \sum_{k=1}^K \lambda_k g_\phi(\theta_k) \right)$$

*Note on Linear Collapsing:* If $g_\phi$ is chosen to be the identity map, FoldMerge collapses exactly to standard task arithmetic. Thus, linear task arithmetic is a trivial, special case of FoldMerge.

### Unsupervised Loss Function & Optimization:
Following the self-labeling principles of SyMerge, we optimize $\phi$ and $\lambda$ at test-time on unlabeled datasets $\mathcal{X}^{te}_k$ using expert predictions as soft labels. The joint objective function is:
$$\min_{\phi, \lambda} \mathcal{L}(\phi, \lambda) = \sum_{k=1}^K \mathbb{E}_{x \in \mathcal{X}^{te}_k} \left[ \mathcal{D}_{KL}\left( f(x; \theta_{MTL}(\phi, \lambda)) \parallel f(x; \theta_k) \right) \right] + \gamma \mathcal{R}(g_\phi)$$

where $\mathcal{D}_{KL}$ is the Kullback-Leibler divergence (for classification tasks) or mean-squared error (for regression tasks), and $\mathcal{R}(g_\phi)$ is the geometric regularization term:
$$\mathcal{R}(g_\phi) = \sum_{k=1}^K \| \mathbf{J}_{g_\phi}(\theta_k) - \mathbf{I} \|_F^2 + \beta \|\phi\|_2^2$$
where $\mathbf{J}_{g_\phi}(\theta_k) = \frac{\partial g_\phi}{\partial \theta}\big|_{\theta_k}$ is the Jacobian matrix of $g_\phi$ evaluated at $\theta_k$, and $\mathbf{I}$ is the identity matrix.

---

## 4. Architecture Specifications

Applying a diffeomorphism directly on a modern large neural network's full weight space ($D \approx 10^9$ parameters) is computationally impossible. Thus, we leverage the crucial architectural insight from SyMerge and apply FoldMerge exclusively to **task-specific head layers or parameter-efficient adapters (e.g., LoRA heads or final classifier weights)**, where $d \approx 10^4$ to $10^5$.

### Diffeomorphism Flow Network ($g_\phi$):
* **Normalizing Flow Backing:** $g_\phi$ consists of a sequence of $M = 4$ affine coupling layers (RealNVP design).
* **Coupling Layer Formulation:**
  For each coupling layer, we split the parameter vector into two halves: $u_1, u_2 = \text{split}(w)$.
  The forward transformation is:
  $$u_1' = u_1$$
  $$u_2' = u_2 \odot \exp(s_\phi(u_1)) + t_\phi(u_1)$$
  The scale network $s_\phi$ and translation network $t_\phi$ are parameter-sharing Multi-Layer Perceptrons (MLPs).
* **MLP Specifications:**
  * **Hidden Dimensions:** 2 layers of hidden size 512.
  * **Activation Function:** GELU (Gaussian Error Linear Unit).
  * **Input/Output Size:** $d / 2$.
  Since $s_\phi$ and $t_\phi$ are feed-forward networks, they do not need to be invertible. The coupling structure guarantees that $g_\phi^{-1}$ is analytically computable in a single forward pass:
  $$u_1 = u_1'$$
  $$u_2 = (u_2' - t_\phi(u_1')) \odot \exp(-s_\phi(u_1'))$$

This highly lightweight and parameter-efficient design ensures that both the forward and inverse passes through the coordinate warp add negligible computational overhead, completing 100 optimization iterations within minutes on a single CPU/GPU.

---

## 5. Baselines
We rigorously compare FoldMerge against the following representative baselines, evaluating its capacity to fold disjoint task manifolds:

1. **Task Arithmetic (Ilharco et al., 2022):** The standard linear Euclidean baseline. Evaluates whether non-linear weight-space folding provides superior performance over linear task vector addition.
2. **SyMerge (Jung et al., 2025):** The state-of-the-art unsupervised test-time adaptive merging baseline. It also adapts only the task-specific layers, but does so via linear interpolation. Comparison with SyMerge isolates the precise benefit of *non-linear folding* vs. *linear scaling*.
3. **OrthoMerge (Yang et al., 2026):** The state-of-the-art manifold merging baseline. It maps updates to the Lie algebra $so(d)$ and leverages Orthogonal Procrustes analysis. Comparison with OrthoMerge tests whether a learned, data-driven coordinate diffeomorphism is more robust than predefined Riemannian manifold projections.
4. **SAIM (Anonymous, 2026):** Isotropic SVD-based singular value balancing. Evaluates whether coordinate-warping is superior to spectral-balancing.

---

## 6. Step-by-Step Interaction

The flow of data and parameters through the FoldMerge pipeline is executed as follows:

```
        +-------------------------------------------------------------+
        |  Task-Specific Heads / Adapters: { \theta_1, ..., \theta_K }|
        +------------------------------+------------------------------+
                                       |
                                       v
                     +---------------------------------+
                     | RealNVP Diffeomorphism (g_\phi) |
                     +-----------------+---------------+
                                       |
                                       v
         +-----------------------------------------------------------+
         |   Origami Representational Space: { z_1, ..., z_K }       |
         +-----------------------------+-----------------------------+
                                       |
                                       v
                     +---------------------------------+
                     | Barycentric Average (\bar{z})   |
                     +-----------------+---------------+
                                       |
                                       v
                   +-------------------------------------+
                   | Inverse Diffeomorphism (g_\phi^-1)  |
                   +-------------------+-----------------+
                                       |
                                       v
                    +-----------------------------------+
                    | Merged MTL Parameters: \theta_MTL |
                    +-------------------+----------------+
                                       |
                                       v
                    +-----------------------------------+
                    | Unlabeled Test Batches: X^te_k    |
                    +-------------------+----------------+
                                       |
                                       v
                 +-----------------------------------------+
                 | Compute Discrepancy & Jacobians         |
                 +---------------------+-------------------+
                                       |
                                       v
                 +-----------------------------------------+
                 | Backpropagate & Update \phi and \lambda |
                 +-----------------------------------------+
```

1. **Extraction Step:** Extract the fine-tuned classifier heads or LoRA matrices $\{ \theta_k \}_{k=1}^K$ from each task model.
2. **Manifold Mapping:** Pass each $\theta_k$ through the forward flow network $g_\phi$ to compute the folded coordinates in Origami Space: $z_k = g_\phi(\theta_k)$.
3. **Barycentric Aggregation:** Compute the folded average $\bar{z} = \sum_k \lambda_k z_k$ using the current merging coefficients.
4. **Inverse Mapping:** Pass the averaged vector $\bar{z}$ through the inverse flow $g_\phi^{-1}$ to decode it back into the original weight-space: $\theta_{MTL} = g_\phi^{-1}(\bar{z})$.
5. **Forward Pass Evaluation:** Insert the merged $\theta_{MTL}$ into the model. Pass a batch of unlabeled data $x \in \mathcal{X}^{te}_k$ through the model to obtain predictions $f(x; \theta_{MTL})$.
6. **Teacher Prediction & Discrepancy:** Evaluate the fixed individual expert on the same batch $x$ to get teacher targets $f(x; \theta_k)$, and compute the KL-divergence loss.
7. **Jacobian Computation:** Compute the Jacobian matrices $\mathbf{J}_{g_\phi}(\theta_k)$ for each task weight to evaluate the geometric volume regularization.
8. **Optimization Step:** Execute backpropagation on the combined loss $\mathcal{L}(\phi, \lambda)$ to update the diffeomorphism parameters $\phi$ and coefficients $\lambda$. Repeat for 100 test-time iterations.
