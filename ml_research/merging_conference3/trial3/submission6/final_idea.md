# Curvature-Aware Analytical Model Merging (ACM)

## 1. Persona Alignment
This proposal is designed through the lens of **The Theorist** persona, who believes that empirical success is meaningless without solid mathematical foundation. Rather than relying on heuristic test-time adaptation (like entropy minimization) which is prone to transductive overfitting, sacrificial task bias, and lack of guarantees, **ACM** derives the optimal merging coefficients directly from the quadratic local approximation of the loss landscape. By projecting the parameter space onto the low-dimensional subspace of the task vectors, we compute the **full, non-diagonal, cross-parameter second-order curvature (Hessian)** of each task's loss landscape along the directions of the task updates. This allows us to derive a **provably optimal, closed-form analytical solution** $\Lambda = A^{-1}b$ for the merging coefficients, backed by a rigorous second-order Taylor formulation of joint multi-task generalization.

---

## 2. Core Techniques
1. **Low-Dimensional Subspace Projection:** We project the massive $D$-dimensional parameter space onto a $K$-dimensional subspace spanned by the task vectors $v_1^l, \dots, v_K^l$ at each layer $l$. This reduces the complexity of capturing full non-diagonal Hessian curvature from a prohibitive $O(D^2)$ memory and $O(D^3)$ inversion cost to a trivial $O(K^2)$ memory and $O(K^3)$ inversion cost.
2. **Analytical Closed-Form Weight Fusing:** Instead of iterative test-time optimization (gradient descent or evolutionary algorithms), we solve for the optimal layer-wise coefficients in a single mathematical step by solving a set of linear equations: $\Lambda^l = (A^l)^{-1} b^l$.
3. **Finite-Difference Projected Hessian Estimation:** We compute the projected Hessian-vector products $(v_i^l)^T H_k^l v_j^l$ without ever explicitly constructing the large Hessian matrix, using a single extra backward pass per task-direction pair:
   $$H_k v_j \approx \frac{\nabla \mathcal{L}_k(W_k + \epsilon v_j) - \nabla \mathcal{L}_k(W_k)}{\epsilon} \approx \frac{1}{\epsilon} \nabla \mathcal{L}_k(W_k + \epsilon v_j)$$
   where $\nabla \mathcal{L}_k(W_k) \approx 0$ at the local minimum.
4. **Hessian Block-Diagonal Cross-Layer Approximation:** We assume cross-layer second-order interactions are negligible (a standard block-diagonal Hessian approximation), allowing us to decompose and solve the optimization problem for each layer $l$ completely independently.
5. **Ridge Regularization (Tikhonov Regularization):** To ensure numerical stability and invertibility of the projected Hessian matrix $A^l$ even in extremely low-curvature or redundant parameter subspaces, we add a small quadratic penalty $\gamma \|\Lambda^l\|_2^2$, modifying the system to:
   $$\Lambda^{l, *} = (A^l + \gamma I)^{-1} b^l$$

---

## 3. Mathematical Formulation

Let $W_0 \in \mathbb{R}^D$ be the pre-trained base model, and $W_k \in \mathbb{R}^D$ be the expert model fine-tuned for task $k \in \{1, \dots, K\}$.
The task vector for expert $k$ at layer $l \in \{1, \dots, L\}$ is:
$$v_k^l = W_k^l - W_0^l$$
For each layer $l$, the merged model weights are defined by:
$$W^l(\Lambda^l) = W_0^l + \sum_{k=1}^K \Lambda_k^l v_k^l$$
where $\Lambda^l = [\Lambda_1^l, \dots, \Lambda_K^l]^T \in \mathbb{R}^K$ is the vector of merging coefficients for layer $l$.

We want to find the parameters $\Lambda = \{\Lambda^l\}_{l=1}^L$ that minimize the joint multi-task loss:
$$\min_{\Lambda} \mathcal{L}_{\text{joint}}(W(\Lambda)) = \min_{\Lambda} \sum_{k=1}^K \alpha_k \mathcal{L}_k(W(\Lambda))$$
where $\mathcal{L}_k$ is the loss of task $k$, and $\alpha_k > 0$ is the task weight (typically $\alpha_k = 1/K$).

Because $W_k$ is the local minimizer of task $k$, we assume the gradient vanishes: $\nabla \mathcal{L}_k(W_k) \approx 0$.
The second-order Taylor expansion of task loss $\mathcal{L}_k(W(\Lambda))$ around $W_k$ is:
$$\mathcal{L}_k(W(\Lambda)) \approx \mathcal{L}_k(W_k) + \frac{1}{2} (W(\Lambda) - W_k)^T H_k (W(\Lambda) - W_k)$$
where $H_k = \nabla^2 \mathcal{L}_k(W_k)$ is the Hessian of task $k$ at $W_k$.

Applying a block-diagonal approximation across layers (where $H_k \approx \text{diag}(H_k^1, \dots, H_k^L)$):
$$\mathcal{L}_k(W(\Lambda)) \approx \mathcal{L}_k(W_k) + \sum_{l=1}^L \frac{1}{2} (W^l(\Lambda^l) - W_k^l)^T H_k^l (W^l(\Lambda^l) - W_k^l)$$

Using the definition of the task vector matrix $V^l = [v_1^l, v_2^l, \dots, v_K^l] \in \mathbb{R}^{d_l \times K}$, the parameter shift from expert $k$ at layer $l$ is:
$$W^l(\Lambda^l) - W_k^l = V^l \Lambda^l - v_k^l$$
The joint optimization problem decomposes across layers into independent quadratic minimization objectives:
$$\min_{\Lambda^l} \sum_{k=1}^K \frac{\alpha_k}{2} (V^l \Lambda^l - v_k^l)^T H_k^l (V^l \Lambda^l - v_k^l)$$

Let's expand this quadratic function of $\Lambda^l$:
$$f(\Lambda^l) = \frac{1}{2} (\Lambda^l)^T A^l \Lambda^l - (\Lambda^l)^T b^l + \text{constant}$$
where:
$$A^l = \sum_{k=1}^K \alpha_k (V^l)^T H_k^l V^l \in \mathbb{R}^{K \times K}$$
$$b^l = \sum_{k=1}^K \alpha_k (V^l)^T H_k^l v_k^l \in \mathbb{R}^K$$

The elements of the projected Hessian matrix $A^l$ and vector $b^l$ are:
$$A^l_{ij} = \sum_{k=1}^K \alpha_k (v_i^l)^T H_k^l v_j^l \quad \text{for } i, j \in \{1, \dots, K\}$$
$$b^l_i = \sum_{k=1}^K \alpha_k (v_i^l)^T H_k^l v_k^l \quad \text{for } i \in \{1, \dots, K\}$$

Setting the gradient to zero and incorporating Ridge regularization ($\gamma > 0$):
$$\nabla_{\Lambda^l} f(\Lambda^l) + \gamma \Lambda^l = (A^l + \gamma I) \Lambda^l - b^l = 0$$
This yields the **exact analytical optimal solution** for the merging coefficients of layer $l$:
$$\Lambda^{l, *} = (A^l + \gamma I)^{-1} b^l$$

---

## 4. Architecture Specifications
- **Backbone Network:** Pre-trained ViT-Tiny (`vit_tiny_patch16_224`), containing $L=14$ parameter groups corresponding to:
  - Layer 0: Patch embeddings
  - Layers 1-12: The 12 independent Transformer blocks
  - Layer 13: Final layer normalization layer
- **Dimensionality:** 
  - Number of tasks $K = 4$ (MNIST, FashionMNIST, CIFAR-10, SVHN).
  - Search/Analytical dimension of $\Lambda^l$ is $K = 4$ per layer.
  - Overall parameter size of $\Lambda$ is $L \times K = 56$.
- **Projected Matrices Size:** $A^l$ is a $4 \times 4$ positive semi-definite matrix, and $b^l$ is a $4 \times 1$ vector. Inversion is extremely fast and numerically stable.

---

## 5. Baselines
We evaluate ACM against:
1. **Task Arithmetic (Uniform):** Static weight merging with uniform coefficients ($\lambda_k^l = 0.3$).
2. **AdaMerging (Iterative, Entropy-based):** Optimizing coefficients $\Lambda$ using first-order Adam GD to minimize unsupervised Shannon entropy of predictions on the test stream (15 steps, learning rate 0.02).
3. **RegCalMerge:** Adding Class-Capacity Normalization (CCN) and Scale-Normalized Entropy Weighting (SNEW) to AdaMerging.
4. **PolyMerge:** Parameterizing layer coefficients as low-degree polynomials and optimizing via entropy minimization.
5. **Q-Merge (Post-hoc):** Running unquantized model merging baselines and then quantizing to 8-bit and 4-bit to check noise robustness.
6. **ACM (Proposed):** Our direct, analytical closed-form solution.

---

## 6. Step-by-Step Interaction (The Forward Pass & Calibration Flow)

1. **Task-Specific Gradient Computation via Perturbation:**
   For each task expert $k \in \{1, \dots, K\}$:
     For each direction $j \in \{1, \dots, K\}$:
       a. Form the perturbed model weights $W = W_k + \epsilon (W_j - W_0)$, where $\epsilon > 0$ is a small finite-difference scale parameter (e.g. $\epsilon = 10^{-3}$).
       b. Feed a small calibration batch of task $k$'s dataset (e.g., 16 or 32 images) to the perturbed model.
       c. Perform a forward pass and compute the task-specific loss $\mathcal{L}_k$.
       d. Perform a backward pass to obtain the model gradient with respect to all layer weights:
          $$g_{k, j} = \nabla_{W} \mathcal{L}_k(W_k + \epsilon v_j)$$
2. **Layer-wise Dot Product Accumulation:**
   For each layer $l \in \{1, \dots, L\}$:
     Extract the layer-specific gradient component $g_{k, j}^l \in \mathbb{R}^{d_l}$.
     For each $i \in \{1, \dots, K\}$:
       Compute the scalar Hessian-vector product proxy:
       $$(v_i^l)^T H_k^l v_j^l \approx \frac{1}{\epsilon} \langle v_i^l, g_{k, j}^l \rangle$$
3. **Linear System Assembly:**
   For each layer $l$:
     Assemble the $K \times K$ matrix $A^l$ and $K \times 1$ vector $b^l$:
     $$A^l_{ij} = \sum_{k=1}^K \alpha_k (v_i^l)^T H_k^l v_j^l$$
     $$b^l_i = \sum_{k=1}^K \alpha_k (v_i^l)^T H_k^l v_k^l$$
4. **Closed-form Solution:**
   For each layer $l$:
     Add Ridge regularization and invert:
     $$\Lambda^{l, *} = (A^l + \gamma I)^{-1} b^l$$
5. **Final Model Fusing:**
   Fuses the task experts using the optimal solved coefficients:
   $$W^l_{\text{merged}} = W_0^l + \sum_{k=1}^K \Lambda_k^{l, *} v_k^l$$
6. **Inference:**
   Deploy the single merged model $W_{\text{merged}}$ directly. It is instantly ready for multi-task inference, completely training-free!
