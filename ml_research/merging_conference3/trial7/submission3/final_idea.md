# Idea Proposal: Gaussian Process Dynamic Routing (GP-DR)

## 1. Persona Alignment
As **The Theorist**, I reject empirical heuristics that lack rigorous mathematical foundations. Current dynamic model merging methods—whether unregularized classical linear routers or wave-inspired quantum metaphors—depend heavily on parametric training loops that overfit catastrophically on scarce calibration data (e.g., $N=64$ samples) and collapse under vectorization ($B=1$) or task-stream shifts.

Gaussian Process Dynamic Routing (GP-DR) is the quintessential theorist's response. Instead of hoping that empirical regularization or gradient damping cures overfitting, GP-DR moves completely to a **non-parametric Bayesian framework**. GP-DR possesses **zero trainable routing parameters**, bypassing the "Overfitting-Optimizer Paradox" entirely. By utilizing a Gaussian Process prior over the low-dimensional representation space, GP-DR derives dynamic merging coefficients as a **closed-form posterior mean** and quantifies out-of-distribution (OOD) uncertainty via a **closed-form posterior variance**. This provides the model-merging field with its first provably smooth, mathematically bounded, and uncertainty-aware dynamic routing framework.

---

## 2. Core Techniques
GP-DR introduces and integrates the following core techniques:
1. **Non-parametric Bayesian Routing:** Dynamic merging coefficients are formulated as the posterior predictive mean of a Gaussian Process, utilizing the frozen $N=64$ calibration samples as coordinate landmarks.
2. **Closed-Form Posterior Inference:** By avoiding gradient descent, backpropagation, and trainable routing weights, the system runs via a single, stable closed-form matrix operation, eliminating optimization noise, learning-rate sensitivity, and seed-dependent local minima.
3. **Provable Uncertainty-Driven OOD Rejection:** The GP posterior variance acts as an exact, mathematically grounded metric of sample-wise out-of-distribution (OOD) density. If a test sample's posterior variance exceeds a theoretically bounded threshold, it is flagged as OOD and safely routed back to the uniform/base model, resolving OOD corruption without empirical cosine heuristics.
4. **Kernel-Bounded Coefficient Smoothness:** By selecting a positive-definite Mercer kernel (e.g., Radial Basis Function or Matern), we structurally guarantee that the dynamic routing function $\alpha(\psi)$ is Lipschitz-continuous and infinitely differentiable, mathematically preventing sequential routing jitter and coefficient collapse.

---

## 3. Mathematical Formulation

Let $\mathcal{D}_{\text{cal}} = \{(\psi(x)_i, y_i)\}_{i=1}^N$ represent the calibration dataset of size $N = 64$.
- Let $\psi(x)_i \in \mathbb{R}^d$ be the low-dimensional normalized representation of sample $i$.
- Let $y_i \in \mathbb{R}^K$ be the target dynamic blending vector for sample $i$. For classification tasks, $y_i$ is a one-hot vector indicating task membership: $y_{i, k} = 1$ if sample $i$ belongs to task $k$, and $0$ otherwise.

### Prior Distribution
We place a Gaussian Process prior on the routing function $f(\psi) = [f_1(\psi), \dots, f_K(\psi)]^T$, where each task's routing coordinate $f_k(\psi)$ is modeled as an independent GP:
$$f_k(\psi) \sim \mathcal{GP}\left(m(\psi), k(\psi, \psi')\right)$$
where $m(\psi) = \frac{1}{K}$ is the constant prior mean (defaulting to static uniform merging), and $k(\psi, \psi')$ is a positive-definite kernel. We employ the **Radial Basis Function (RBF) kernel**:
$$k(\psi, \psi') = \sigma_f^2 \exp\left(-\frac{\|\psi - \psi'\|_2^2}{2 \ell^2}\right)$$
where $\sigma_f^2$ is the signal variance, and $\ell$ is the characteristic lengthscale parameter.

### Gram Matrix and Cross-Covariance
Let $\mathbf{\Psi}_{\text{cal}} \in \mathbb{R}^{N \times d}$ denote the matrix of calibration inputs. The Gram covariance matrix $\mathbf{K} \in \mathbb{R}^{N \times N}$ is defined element-wise as:
$$\mathbf{K}_{i, j} = k(\psi_i, \psi_j)$$
For a new test sample representation $\psi_* \in \mathbb{R}^d$, we compute the cross-covariance vector $\mathbf{k}_* \in \mathbb{R}^{1 \times N}$:
$$\mathbf{k}_* = [k(\psi_*, \psi_1), \dots, k(\psi_*, \psi_N)]$$

### Closed-Form Posterior Inference
Under Gaussian noise assumption with variance $\sigma_n^2$, the joint distribution of the calibration targets $\mathbf{Y} \in \mathbb{R}^{N \times K}$ and the latent function values $f(\psi_*)$ is Gaussian. The predictive posterior distribution of the routing coefficients $\alpha(\psi_*) \in \mathbb{R}^K$ is:
$$\alpha(\psi_*) \mid \psi_*, \mathcal{D}_{\text{cal}} \sim \mathcal{N}\left(\mu(\psi_*), \sigma^2(\psi_*)\mathbf{I}\right)$$

The **posterior mean** vector $\mu(\psi_*) \in \mathbb{R}^K$ (serving as our dynamic merging coefficients) is solved in closed-form as:
$$\mu(\psi_*) = \mathbf{m}(\psi_*) + \mathbf{k}_* \left( \mathbf{K} + \sigma_n^2 \mathbf{I} \right)^{-1} \left( \mathbf{Y} - \mathbf{m}(X) \right)$$
where $\mathbf{m}(\psi_*) = [1/K, \dots, 1/K]$ and $\mathbf{m}(X) \in \mathbb{R}^{N \times K}$ is the prior mean matrix over training samples.

The **posterior variance** $\sigma^2(\psi_*) \in \mathbb{R}$ representing the model's epistemic uncertainty is:
$$\sigma^2(\psi_*) = k(\psi_*, \psi_*) - \mathbf{k}_* \left( \mathbf{K} + \sigma_n^2 \mathbf{I} \right)^{-1} \mathbf{k}_*^T$$

### Uncertainty-Guided OOD Rejection Fallback
Because the posterior variance $\sigma^2(\psi_*)$ is provably bounded between $[0, \sigma_f^2]$, we establish a mathematically sound threshold $\theta_{\text{OOD}} = \gamma \cdot \sigma_f^2$ (where $\gamma \in (0, 1)$). 
The final dynamic routing coefficients $\alpha^{\text{GP-DR}}_b$ are formulated as:
$$\alpha^{\text{GP-DR}}_b = \begin{cases} \mu(\psi_b) & \text{if } \sigma^2(\psi_b) \le \theta_{\text{OOD}} \\ [1/K, \dots, 1/K]^T & \text{if } \sigma^2(\psi_b) > \theta_{\text{OOD}} \end{cases}$$

This guarantees that any OOD sample (e.g., SVHN samples when only MNIST/CIFAR experts are active) having low kernel similarity to the calibration set triggers a high posterior variance, safely collapsing the coefficients back to the uniform prior, completely shielding the experts from destructive OOD interference.

---

## 4. Architecture Specifications

### Dimensionality and Transformations
1. **Penultimate Hidden Vector:** $z(x)_b \in \mathbb{R}^D$, representing the pooled hidden states from the backbone network ($D=192$ for ViT-Tiny, $D=768$ for ViT-Base).
2. **Dimension Projection:** To compress computational complexity, we project $z(x)_b$ to $d = K$ dimensions using a frozen projection matrix $P \in \mathbb{R}^{D \times d}$ (e.g., Random Gaussian Projection or PCA).
3. **State Normalization:** To guarantee scale invariance, the low-dimensional state representation $\psi(x)_b \in \mathbb{R}^d$ is projected onto the unit sphere:
   $$\psi(x)_b = \frac{z(x)_b P}{\|z(x)_b P\|_2 + \epsilon}$$
4. **GP Solver:** A single-layer matrix operator that pre-computes the inverse kernel matrix $\mathbf{M} = \left(\mathbf{K} + \sigma_n^2 \mathbf{I}\right)^{-1} \in \mathbb{R}^{N \times N}$ and the coefficient mapping matrix $\mathbf{W}_{\text{GP}} = \mathbf{M} \left(\mathbf{Y} - \mathbf{m}(X)\right) \in \mathbb{R}^{N \times K}$ offline.
5. **Runtime Computational Complexity:** At test time, for each sample $b$:
   - Project and Normalize: $O(Dd)$ FLOPs.
   - Cross-covariance computation: $\mathbf{k}_* \in \mathbb{R}^{1 \times N}$ takes $O(Nd)$ FLOPs.
   - Posterior Mean: $\alpha = \mathbf{m} + \mathbf{k}_* \mathbf{W}_{\text{GP}}$ takes $O(NK)$ FLOPs.
   - Posterior Variance: $\sigma^2 = k(\psi_*, \psi_*) - \mathbf{k}_* \mathbf{M} \mathbf{k}_*^T$ takes $O(N^2)$ FLOPs.
   With $N=64, K=4, d=4$, the complete routing pass requires less than $5 \times 10^3$ FLOPs, representing an extremely lightweight forward pass that is highly viable for real-time vectorized pipelines.

---

## 5. Baselines
To validate the performance of GP-DR, we compare it against:
1. **Static Uniform Merging:** The standard baseline representing parameter averaging with equal weights ($\alpha_k = 1/K$).
2. **Unregularized classical Linear Router:** Standard linear router trained on cross-entropy, exposing the extent of overfitting under low-data calibration.
3. **Task-Space Anchor Regularization (TSAR) / L2-Regularized Linear Routers:** High-performing parametric classical baselines that require standard backpropagation calibration.
4. **Parameter-Free Subspace Routing (PFSR):** The modern state-of-the-art non-parametric router based on direct cosine projections, which lacks Bayesian posterior uncertainty quantification.
5. **Quantum Wavefunction Superposition Merging (QWS-Merge):** The complex "wave-inspired" parametric router, used to demonstrate that simple closed-form Bayesian priors systematically outperform quantum metaphors.

---

## 6. Step-by-Step Interaction

### Step 1: Offline Calibration Setup (Zero-Shot Preparation)
1. Collect the 64 calibration samples and extract their penultimate hidden representations $z(x)_i \in \mathbb{R}^{D}$.
2. Generate the frozen dimension-projection matrix $P \in \mathbb{R}^{D \times d}$ and project the samples to get $\psi_i \in \mathbb{R}^d$.
3. Compute the Gram matrix $\mathbf{K} \in \mathbb{R}^{N \times N}$ using the RBF kernel.
4. Compute the inverse matrix $\mathbf{M} = \left(\mathbf{K} + \sigma_n^2 \mathbf{I}\right)^{-1} \in \mathbb{R}^{N \times N}$.
5. Construct the task-target matrix $\mathbf{Y} \in \mathbb{R}^{N \times K}$ and pre-compute the mapping weights $\mathbf{W}_{\text{GP}} = \mathbf{M} \left(\mathbf{Y} - \mathbf{m}(X)\right) \in \mathbb{R}^{N \times K}$. Store $\mathbf{\Psi}_{\text{cal}}$, $\mathbf{W}_{\text{GP}}$, and $\mathbf{M}$.

### Step 2: Inference Stream Processing (Real-time Forward Pass)
1. For an incoming test batch of size $B$ (or vectorized $B=1$), extract the feature representations $Z_* \in \mathbb{R}^{B \times D}$.
2. Project and normalize $Z_*$ to obtain the test coordinates $\mathbf{\Psi}_* \in \mathbb{R}^{B \times d}$.
3. For each sample $\psi_*$ in the batch:
   - Compute kernel similarities against all calibration coordinates: $\mathbf{k}_* = [k(\psi_*, \psi_1), \dots, k(\psi_*, \psi_N)] \in \mathbb{R}^{1 \times N}$.
   - Compute the posterior mean: $\mu(\psi_*) = [1/K, \dots, 1/K] + \mathbf{k}_* \mathbf{W}_{\text{GP}} \in \mathbb{R}^K$.
   - Compute the posterior variance: $\sigma^2(\psi_*) = \sigma_f^2 - \mathbf{k}_* \mathbf{M} \mathbf{k}_*^T \in \mathbb{R}$.
4. **Apply OOD Rejection:** If $\sigma^2(\psi_*) > \theta_{\text{OOD}}$, override the routing coefficients to $[1/K, \dots, 1/K]^T$. Otherwise, keep $\alpha = \mu(\psi_*)$.
5. **Dynamic Homogenization & Assembly:** Pass the generated coefficients $\alpha$ to the parameter assembly kernel to merge the active layers of the expert networks on-the-fly and compute the multi-task model forward pass.
