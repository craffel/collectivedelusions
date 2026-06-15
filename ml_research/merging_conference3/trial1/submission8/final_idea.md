# Idea Proposal: Limits of Representational Isotropy on Curved Manifolds (RIMO)

## 1. Persona Alignment
As a Theorist, I approach machine learning problems through the lens of mathematics, geometry, and spectral analysis. I am skeptical of purely heuristic parameter averaging (such as Task Arithmetic), which lacks geometric justification and disrupts the structural properties of trained representations. 

**RIMO** (\textbf{R}iemannian \textbf{I}sometry-respecting \textbf{M}anifold \textbf{O}perations) is grounded in a rigorous mathematical formulation: it performs model merging on the Riemannian manifold of the orthogonal group $\mathrm{O}(d)$ by utilizing its associated Lie algebra $\mathfrak{so}(d)$ as the vector space of infinitesimal generators. In this work, we conduct a deep theoretical diagnostic study into the limits of representational isotropy on curved manifolds. We expose a surprising and profound spectral balancing pitfall in Lie algebra tangent spaces: while spectral balancing (such as SAIM) is safe in flat Euclidean space, doing so on a manifold injects catastrophic high-dimensional rotational noise. We prove this via the Kernel Distortion Theorem and the Spectrum Distortion Theorem. To bypass this pitfall, we propose **RIMO-Pruned**, which utilizes rank-preserving spectral pruning to maintain the low-rank structure of tangent updates, preserving geometric and representational stability.

## 2. Core Techniques
RIMO integrates several foundational techniques and new geometric discoveries:
1. **Riemannian Manifold Mapping:** Mapping task-specific weight updates to the orthogonal group $\mathrm{O}(d)$ and its Lie algebra $\mathfrak{so}(d)$ via the Cayley transform, building on the framework established in **OrthoMerge** (Yang et al., 2026).
2. **Orthogonal-Residual Decoupling:** Extracting orthogonal rotational components from standard model parameters by solving the Orthogonal Procrustes problem (Gower & Dijksterhuis, 2004).
3. **The Tangent Space Spectral Pitfall:** We mathematically prove that attempting to perform isotropic spectral balancing inside $\mathfrak{so}(d)$ (interpolating singular values towards their mean) is highly destructive because the subsequent skew-symmetric projection step distorts the spectrum (Spectrum Distortion Theorem), and standard SVD solvers introduce non-symmetric coordinate gauges in multi-dimensional kernels (Kernel Distortion Theorem).
4. **Rank-Preserving Spectral Pruning:** Our proposed mitigation, **RIMO-Pruned**, which keeps inactive dimensions at exactly zero to prevent the injection of spurious rotations and coordinate warp.

## 3. Mathematical Formulation

Let $W_0 \in \mathbb{R}^{d_{in} \times d_{out}}$ be the pre-trained base model weights, and $\{W_k\}_{k=0}^{N-1}$ be the expert weights finetuned on $N$ downstream tasks.

### Step 1: Orthogonal Extraction via Procrustes Analysis
For each task $k$, we solve the Orthogonal Procrustes problem to decouple the finetuned weights into an orthogonal component $R_k \in O(d)$ and a residual component $\rho_k$:
$$R_k = \arg\min_{R} \|W_k - R W_0\|_F \quad \text{s.t.} \quad R^T R = I$$
The closed-form analytical solution is obtained via Singular Value Decomposition (SVD):
$$U_k \Sigma_k V_k^T = \text{SVD}(W_k W_0^T)$$
$$R_k = U_k V_k^T$$
The linear residual component representing the Euclidean displacement is captured as:
$$\rho_k = W_k - R_k W_0$$

### Step 2: Inverse Cayley Mapping to the Lie Algebra $so(d)$
We map the orthogonal matrices $\{R_k\}$ to their skew-symmetric Lie algebra representations $\{Q_k\}$ via the inverse Cayley transform:
$$Q_k = (R_k - I)(R_k + I)^{-1}$$
By algebraic properties of the Cayley transform, each $Q_k$ is strictly skew-symmetric: $Q_k^T = -Q_k$.

### Step 3: Magnitude-Corrected Aggregation
We perform a magnitude-corrected sum of the Lie algebra components to handle destructive interference:
$$Q_{com} = c \cdot \left( \frac{1}{N} \sum_{k=0}^{N-1} Q_k \right)$$
where the scaling factor $c \in \mathbb{R}$ is defined as:
$$c = \frac{\sum_{k=0}^{N-1} \|Q_k\|_F}{\left\| \sum_{k=0}^{N-1} Q_k \right\|_F}$$

### Step 4: Isotropic Spectral Balancing in $so(d)$
To prevent any single task direction from dominating the representation space, we perform Singular Value Decomposition on the skew-symmetric combined update:
$$U_{com} \Sigma_{com} V_{com}^T = \text{SVD}(Q_{com})$$
where $\Sigma_{com} = \text{diag}(\sigma_1, \sigma_2, \dots, \sigma_r)$. The mean singular value is computed as:
$$\bar{\sigma} = \frac{1}{r} \sum_{i=1}^r \sigma_i$$
We interpolate the singular value spectrum towards uniform isotropy:
$$\hat{\Sigma}_{com} = \bar{\sigma} I + (\Sigma_{com} - \bar{\sigma} I) \times \frac{1}{\sqrt{t}}$$
where $t$ is the current number of tasks (or a hyperparameter controlling the strength of the isotropic prior). 
We reconstruct the balanced component:
$$Q'_{com} = U_{com} \hat{\Sigma}_{com} V_{com}^T$$
To guarantee that the balanced component rigorously satisfies the algebraic constraints of the Lie algebra $so(d)$ under finite-precision arithmetic, we apply the projection operator:
$$\hat{Q}_{com} = \frac{1}{2}(Q'_{com} - (Q'_{com})^T)$$

### Step 5: Mapping Back to $O(d)$ and Residual Merging
The balanced skew-symmetric matrix is mapped back to the orthogonal group $O(d)$ via the forward Cayley transform:
$$R_{merged} = (I + \hat{Q}_{com})(I - \hat{Q}_{com})^{-1}$$
Simultaneously, the residual components $\{\rho_k\}$ are merged using standard Euclidean arithmetic:
$$\rho_{merged} = \frac{1}{N} \sum_{k=0}^{N-1} \rho_k$$
The final hybrid merged weights $W_{final}$ are constructed as:
$$W_{final} = R_{merged} W_0 + \rho_{merged}$$

## 4. Architecture Specifications
RIMO is a weight-level merging framework and is compatible with any model architecture that utilizes linear layers (including Transformers, MLPs, and CNNs). 
- **Target Layers:** Linear projection matrices in Self-Attention blocks (specifically $W_q, W_k, W_v, W_o$) and MLP blocks ($W_{gate}, W_{up}, W_{down}$).
- **Dimensionality:** For each target layer with dimension $d_{in} \times d_{out}$, if $d_{in} \ne d_{out}$, we apply the operations on block-diagonal square sub-matrices or pad/project the dimensions following the standard block-diagonal parameterization of OFT. Typically, $d = \min(d_{in}, d_{out})$.
- **Hyperparameters:**
  - $t$: The isotropic interpolation factor (defaults to $N$, the number of merged models).
  - Block size $b$: The block-diagonal size for SVD partition (typically 32 or 64 to ensure $O(b^3)$ computational efficiency).

## 5. Baselines
To rigorously evaluate the theoretical advantages of RIMO, we compare against the following baselines:
1. **Task Arithmetic (TA) (Ilharco et al., 2023):** Standard linear addition in Euclidean space. Appropriate as a baseline to demonstrate the benefit of geometry-preserving manifold merging.
2. **AdaMerging (Yang et al., 2024b):** Test-time adaptive model merging in Euclidean space. Compares RIMO against state-of-the-art unsupervised Euclidean optimization.
3. **OrthoMerge (Yang et al., 2026):** Riemannian manifold merging without isotropic spectral balancing. Helps ablate the impact of our Lie-algebraic isotropic balancing.
4. **SAIM (Anonymous, 2026):** Sharpness-aware isotropic model merging in Euclidean space. Helps evaluate whether isotropic balancing on the manifold yields better functional representation alignment than in Euclidean space.

## 6. Step-by-Step Interaction
The flow of data and operations in RIMO proceeds as follows:
1. **Input Stage:** Receive the pre-trained base model $W_0$ and the task-specific expert models $\{W_k\}$.
2. **Deconstruction Phase:** Solve the Procrustes SVD on $W_k W_0^T$ to decouple the weights into $R_k$ (orthogonal rotation) and $\rho_k$ (linear residuals).
3. **Manifold Mapping:** Apply the inverse Cayley transform to project $R_k$ into the Lie algebra $so(d)$ as a skew-symmetric matrix $Q_k$.
4. **Spectral Balancing:**
   - Compute the magnitude-corrected average $Q_{com}$ of $\{Q_k\}$.
   - Apply SVD on $Q_{com}$ to extract singular values.
   - Interpolate singular values towards the mean $\bar{\sigma}$ using the isotropic scale $1/\sqrt{t}$.
   - Reconstruct and project onto the skew-symmetric subspace to obtain $\hat{Q}_{com}$.
5. **Reconstruction Phase:** Map $\hat{Q}_{com}$ back to $R_{merged}$ via the forward Cayley transform, average the residuals to obtain $\rho_{merged}$, and compute the final hybrid weights $W_{final} = R_{merged} W_0 + \rho_{merged}$.
6. **Output Stage:** Return the merged model $W_{final}$ for downstream inference or evaluation.
