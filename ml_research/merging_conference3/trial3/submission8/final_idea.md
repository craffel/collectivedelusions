# GP-BayesMerge: A Gaussian Process PAC-Bayes Framework for Robust Test-Time Model Merging

## 1. Persona Alignment
This proposal directly aligns with **The Theorist** persona:
- **Foundational Rigor:** Instead of proposing heuristic regularization terms (such as independent $L_2$ penalties or standard deviation minimization as in `RegCalMerge`), this approach is mathematically derived from first principles using the **PAC-Bayes generalization framework**.
- **Gaussian Process Priors:** By modeling the prior distribution of merging coefficients across neural network layers as a Gaussian Process over normalized depth, we provide a formal, continuous spatial model of deep network representations, rather than treating layers as disconnected parameter blocks.
- **Mathematical Unification:** We show that our derived precision-matrix quadratic form $\Sigma_{\ell}^{-1}$ mathematically unifies proximity constraints (distance from initialization) and spatial smoothness (correlation between adjacent layers) under a single positive-definite metric, offering clear guarantees of optimization stability and curvature-bound flat minima.

---

## 2. Core Techniques
The core mechanisms introduced in **GP-BayesMerge** are:
1. **PAC-Bayes Generalization Bounds:** Provides the theoretical surrogate objective for test-time adaptation, mapping the complexity penalty of the target risk to the KL divergence between the coefficient posterior and prior.
2. **Gaussian Process (GP) Parameter Prior:** Placing an RBF-kernelized GP prior over layer coefficients as a function of depth:
   $$\lambda_{\cdot, k} \sim \mathcal{GP}\left(\mu_0, \mathcal{K}(z, z')\right)$$
   where $z_l = l/L \in [0, 1]$ is the normalized layer depth.
3. **Precision Matrix Quadratic Regularization:** Converting the GP prior's covariance matrix $\Sigma_{\ell}$ into a precision-matrix regularization term:
   $$\mathcal{L}_{\text{GP}}(\Lambda) = \frac{1}{2} \sum_{k=1}^K (\lambda_{\cdot, k} - \mu_0)^T \Sigma_{\ell}^{-1} (\lambda_{\cdot, k} - \mu_0)$$
   where $\Sigma_{\ell}^{-1}$ acts as a spatial regularizer, smoothing out high-frequency optimization noise and solving the Overfitting-Optimizer Paradox.
4. **Calibration Engine Integration:** Integrates with Class-Capacity Normalization (CCN) and Scale-Normalized Entropy Weighting (SNEW) from `RegCalMerge` to eliminate sacrificial task bias on complex domains.

---

## 3. Mathematical Formulation

### 1. Merging Model Parameterization
Let $\theta_0$ be the pre-trained model parameter vector, and $\{\theta_k\}_{k=1}^K$ be the $K$ task-specific expert parameter vectors. For layer $l \in \{1, \ldots, L\}$, let $\theta_{0, l}$ and $\theta_{k, l}$ be the layer-specific parameter vectors. The merged parameters at layer $l$ are given by:
$$\theta_l(\Lambda) = \theta_{0, l} + \sum_{k=1}^K \lambda_{l, k} (\theta_{k, l} - \theta_{0, l})$$
where $\lambda_{l, k}$ is the merging coefficient for layer $l$ and task $k$.

### 2. Prior and Posterior over Coefficients
Let the prior distribution over the task-wise coefficient vector $\lambda_k = [\lambda_{1, k}, \ldots, \lambda_{L, k}]^T \in \mathbb{R}^L$ be a multivariate Gaussian:
$$P(\lambda_k) = \mathcal{N}(\mu_0, \Sigma_{\ell})$$
where $\mu_0 = \frac{1}{K} \mathbf{1}_L$ is the uniform Task Arithmetic coefficient, and $\Sigma_{\ell} \in \mathbb{R}^{L \times L}$ is the GP covariance matrix over normalized layer coordinates $z_l = l / L$:
$$[\Sigma_{\ell}]_{l, l'} = \sigma_p^2 \exp\left( - \frac{(z_l - z_{l'})^2}{2 \ell^2} \right) + \sigma_n^2 \delta_{l, l'}$$
Here, $\ell$ is the spatial lengthscale, $\sigma_p^2$ is the signal variance, and $\sigma_n^2$ is the jitter noise (set to $10^{-5}$ for numerical invertibility).

We define the posterior distribution $Q(\Lambda)$ as a Gaussian centered at the optimized coefficients with a narrow isotropic variance $\sigma_q^2 I$:
$$Q(\lambda_k) = \mathcal{N}(\lambda_k^*, \sigma_q^2 I)$$

### 3. PAC-Bayes KL Objective
The complexity penalty of the PAC-Bayes bound is proportional to the KL divergence:
$$\text{KL}(Q(\Lambda) \| P(\Lambda)) = \sum_{k=1}^K \text{KL}(Q(\lambda_k) \| P(\lambda_k))$$
Discarding terms independent of the optimization parameters $\lambda_k^*$, we obtain the GP-BayesMerge regularization term:
$$\mathcal{L}_{\text{GP}}(\Lambda) = \frac{\alpha}{2} \sum_{k=1}^K (\lambda_k - \mu_0)^T \Sigma_{\ell}^{-1} (\lambda_k - \mu_0)$$
where $\alpha$ is a scaling hyperparameter reflecting the PAC-Bayes trade-off.

### 4. Complete Optimization Objective
The joint loss optimized on a test-time calibration batch $X = \{X_1, \ldots, X_K\}$ is:
$$\mathcal{L}_{\text{total}}(\Lambda) = \sum_{k=1}^K w_k \cdot \tilde{\mathcal{H}}_k(\theta(\Lambda); X_k) + \frac{\alpha}{2} \sum_{k=1}^K (\lambda_k - \mu_0)^T \Sigma_{\ell}^{-1} (\lambda_k - \mu_0)$$
where:
- $\tilde{\mathcal{H}}_k$ is the Class-Capacity Normalized (CCN) entropy:
  $$\tilde{\mathcal{H}}_k = \frac{\mathcal{H}_k}{\log C_k}$$
- $w_k$ is the Scale-Normalized Entropy Weighting (SNEW):
  $$w_k = \frac{1}{\tilde{\mathcal{H}}_k^0}$$
  where $\tilde{\mathcal{H}}_k^0$ is the baseline normalized entropy under uniform task arithmetic.

---

## 4. Architecture Specifications

### 1. Backbone Model
- **Architecture:** Vision Transformer (e.g., ViT-B/16 or ViT-L/16) or ResNet-50.
- **Layers ($L$):** $L = 12$ Transformer blocks or $L = 50$ ResNet residual stages.
- **Task Experts ($K$):** $K = 4$ domain experts corresponding to the target datasets.

### 2. Optimization Variables
- **Parameters $\Lambda$:** A learnable matrix of size $L \times K$, initialized to $\frac{1}{K}$ everywhere.
- **Bounds:** Strictly constrained to $[0, 1]^L$ per task via clamping during optimization.

### 3. Hyperparameters
- **Lengthscale $\ell$:** Controls spatial smoothness. Typical range $\ell \in [0.05, 0.3]$.
- **Signal Variance $\sigma_p^2$:** Controls permissible parameter drift scale. Set to $1.0$.
- **Regularization Strength $\alpha$:** Typical range $\alpha \in [10^{-4}, 10^{-1}]$.
- **Jitter $\sigma_n^2$:** Fixed at $10^{-5}$ for matrix inversion.

---

## 5. Baselines
To validate GP-BayesMerge, we compare it against the following baselines:
1. **Task Arithmetic (Uniform Merging):** No optimization, $\lambda_{l, k} = \frac{1}{K}$ for all $l, k$. This is the baseline from which we measure improvement.
2. **Standard AdaMerging (Yang et al., 2024):** Unconstrained layer-wise entropy minimization. This serves to show the susceptibility of unregularized methods to the Overfitting-Optimizer Paradox.
3. **RegCalMerge (trial2_submission1):** Uses Elastic Spatial Regularization (ESR) with separate proximity ($\beta$) and spatial variance ($\gamma$) penalties. This baseline represents heuristic spatial smoothing.
4. **PolyMerge (trial2_submission3):** Subspace polynomial projection of coefficients. This represents hard constraint-based smoothing.
5. **Flat Spatial Averaging (trial1_submission7):** Forces coefficients to be identical across layers per task. Represents the infinite lengthscale limit ($\ell \to \infty$) of GP-BayesMerge.

---

## 6. Step-by-Step Interaction

### Phase 1: Setup and Initialization
1. Load the pre-trained base model $\theta_0$ and task-specific experts $\{\theta_k\}_{k=1}^K$.
2. Compute the normalized layer depths $z_l = l / L$ for $l = 1, \dots, L$.
3. Construct the RBF covariance matrix $\Sigma_{\ell}$ of size $L \times L$ using normalized coordinates.
4. Invert $\Sigma_{\ell}$ using a Cholesky decomposition or robust standard inversion to obtain the precision matrix $\Sigma_{\ell}^{-1}$.
5. Initialize the merging coefficients $\Lambda = \{\lambda_{l, k}\}$ to $\frac{1}{K}$.

### Phase 2: Calibration
1. Receive a small test-time calibration batch $X_k$ of size $N$ for each task $k$.
2. Forward-propagate the calibration batch through the merged model parameterized by current $\Lambda$.
3. Compute the Class-Capacity Normalized (CCN) entropy $\tilde{\mathcal{H}}_k$ and Scale-Normalized Entropy Weighting (SNEW) coefficients $w_k$ to build the multi-task loss.

### Phase 3: Optimization Step (TTA Loop)
1. Compute the task entropy loss $\mathcal{L}_{\text{entropy}}(\Lambda) = \sum_{k=1}^K w_k \cdot \tilde{\mathcal{H}}_k$.
2. Compute the GP-Bayes quadratic penalty:
   $$\mathcal{L}_{\text{GP}}(\Lambda) = \frac{\alpha}{2} \sum_{k=1}^K (\lambda_k - \mu_0)^T \Sigma_{\ell}^{-1} (\lambda_k - \mu_0)$$
3. Sum the losses: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{entropy}} + \mathcal{L}_{\text{GP}}$.
4. Perform gradient backpropagation with respect to $\Lambda$ to compute $\nabla_{\Lambda} \mathcal{L}_{\text{total}}$.
5. Update $\Lambda$ using the Adam optimizer with learning rate $\eta$ (e.g., $10^{-3}$), and clamp $\Lambda$ to $[0, 1]$.
6. Repeat for $T$ optimization steps (typically $T = 100$).

### Phase 4: Inference
1. Fuse the model parameters using the final optimized coefficients $\Lambda^*$.
2. Deploy the fused single model to perform joint inference on target streams.
