# Unitary Geodesic Routing (UGR) for Stable Test-Time Model Ensembling

## 1. Persona Alignment
This proposal strongly aligns with the **Visionary** persona through its radical departure from standard Euclidean state-space representations and its direct inspiration from the geometric principles of quantum mechanics and differential geometry. 
- **Rethinking Fundamental Assumptions:** Prior stateful routing methods (such as ChemMerge, PAC-Kinetics, and Momentum-Merge) assume that routing updates must occur in a flat Euclidean space, requiring post-hoc Softmax normalization to project weights onto the probability simplex. We reject this assumption, demonstrating that ensembling weights can be modeled directly as coordinates on a curved hypersphere $\mathbb{S}^{K-1}$.
- **Unconventional and Risky:** Operating on non-Euclidean manifolds is mathematically elegant but computationally risky due to the complexities of curved transitions. However, we resolve this by deriving a closed-form, extremely efficient geodesic rotation operator (spherical interpolation), completely bypassing expensive matrix exponential calculations.
- **Novelty over Incremental Tweaks:** Rather than tweaking the EMA decay rates of Momentum-Merge or adding more complex chemical ODE terms to ChemMerge, UGR establishes an entirely new geometric routing paradigm that is Softmax-free, norm-preserving, and inherently stable.

## 2. Core Techniques
Unitary Geodesic Routing introduces several core techniques to test-time model ensembling:
- **Spherical State Representation:** The ensembling state is modeled as a unit-norm vector $\mathbf{s}_t^{(l)}$ on the $(K-1)$-sphere, $\mathbb{S}^{K-1} \subset \mathbb{R}^K$.
- **Softmax-Free Simplex Projection:** Ensembling weights are mapped directly as the coordinate-wise squared magnitudes of the state vector: $\alpha_{k, t}^{(l)} = (s_{k, t}^{(l)})^2$. This guarantees that the weights reside strictly on the probability simplex $\Delta^{K-1}$ with zero mathematical artifacts, mirroring the Born rule of quantum mechanics.
- **Closed-Form Geodesic Rotation (Spherical EMA):** State transitions are modeled as continuous rotations along the shortest geodesic path on the sphere. Using a custom closed-form Rodrigues-like formulation, we compute spherical interpolation (Slerp) without instantiating high-dimensional rotation matrices or numerical ODE solvers.
- **Torque-Driven Adaptive Agility:** The rotation speed naturally scales with the angular distance (torque) between the current state and the incoming activation signal. Under highly confident task switches, the torque explodes, instantly overriding stateful inertia and suppressing representational lag without any hand-crafted thresholds.
- **Spatial-Temporal Geodesic Coupling:** We establish a continuous 2D state-space by initializing the first adapted layer's state of the current query using the final layer's state of the previous query: $\mathbf{s}_t^{(L_{\text{frozen}})} = \mathbf{s}_{t-1}^{(L)}$.

## 3. Mathematical Formulation

### 3.1 State and Probability Space
Let $K$ be the number of expert adapters. We define the state vector at time step $t$ and adapted layer $l$ as:
$$\mathbf{s}_t^{(l)} \in \mathbb{S}^{K-1} = \left\{ \mathbf{s} \in \mathbb{R}^K : \|\mathbf{s}\|_2 = 1 \right\}$$

The active ensembling weights $\boldsymbol{\alpha}_t^{(l)} \in \Delta^{K-1}$ are computed coordinate-wise via:
$$\alpha_{k, t}^{(l)} = \left(s_{k, t}^{(l)}\right)^2$$

This formulation satisfies the probability simplex constraints exactly:
$$\sum_{k=1}^K \alpha_{k, t}^{(l)} = \sum_{k=1}^K \left(s_{k, t}^{(l)}\right)^2 = \left\|\mathbf{s}_t^{(l)}\right\|_2^2 = 1 \quad \text{and} \quad \alpha_{k, t}^{(l)} \ge 0 \quad \forall k$$

### 3.2 Bottom-Up Target Vector
Let $\mathbf{e}_t^{(l)} \in [0, 1]^K$ be the raw coordinate similarity projections (e.g., extracted via SABLE nearest-centroids or Subspace Energy Projections). We map this bottom-up signal onto the unit sphere to construct our target vector $\mathbf{w}_t^{(l)} \in \mathbb{S}^{K-1}$:
$$\mathbf{w}_t^{(l)} = \frac{\mathbf{e}_t^{(l)}}{\left\|\mathbf{e}_t^{(l)}\right\|_2 + \epsilon}$$
where $\epsilon = 10^{-6}$ is a numerical stability constant.

### 3.3 Geodesic Rotation Operator
To transition the state from layer $l-1$ to layer $l$ under an inertia parameter $\eta \in [0, 1]$, we perform a geodesic rotation along the great circle connecting $\mathbf{s}_t^{(l-1)}$ and $\mathbf{w}_t^{(l)}$:

1. **Compute Alignment:** Measure the cosine of the angle between the current state and the target vector:
   $$c_t^{(l)} = \left(\mathbf{s}_t^{(l-1)}\right)^T \mathbf{w}_t^{(l)} \in [-1, 1]$$

2. **Handle Collinearity:** If $\left|c_t^{(l)}\right| \ge 1 - 10^{-6}$, the state and target are already aligned, and we set:
   $$\mathbf{s}_t^{(l)} = \mathbf{s}_t^{(l-1)}$$

3. **Orthonormalize Target Component:** If they are not collinear, extract the component of $\mathbf{w}_t^{(l)}$ that is orthogonal to $\mathbf{s}_t^{(l-1)}$:
   $$\mathbf{v}_t^{(l)} = \mathbf{w}_t^{(l)} - c_t^{(l)} \mathbf{s}_t^{(l-1)}$$
   Normalize it to obtain the orthogonal unit vector $\mathbf{u}_t^{(l)}$:
   $$\mathbf{u}_t^{(l)} = \frac{\mathbf{v}_t^{(l)}}{\left\|\mathbf{v}_t^{(l)}\right\|_2}$$

4. **Angle Interpolation:** Compute the full geodesic angular distance $\phi_t^{(l)}$ and scale it using the step size parameter $\eta$:
   $$\phi_t^{(l)} = \arccos\left(c_t^{(l)}\right)$$
   $$\theta_t^{(l)} = \eta \cdot \phi_t^{(l)}$$

5. **Spherical Update:** Rotate the state along the geodesic path:
   $$\mathbf{s}_t^{(l)} = \cos\left(\theta_t^{(l)}\right) \mathbf{s}_t^{(l-1)} + \sin\left(\theta_t^{(l)}\right) \mathbf{u}_t^{(l)}$$

Since $\mathbf{s}_t^{(l-1)}$ and $\mathbf{u}_t^{(l)}$ are orthonormal, the updated state is guaranteed to lie on the unit sphere:
$$\left\|\mathbf{s}_t^{(l)}\right\|_2^2 = \cos^2\left(\theta_t^{(l)}\right) \left\|\mathbf{s}_t^{(l-1)}\right\|_2^2 + \sin^2\left(\theta_t^{(l)}\right) \left\|\mathbf{u}_t^{(l)}\right\|_2^2 = \cos^2\left(\theta_t^{(l)}\right) + \sin^2\left(\theta_t^{(l)}\right) = 1$$

## 4. Architecture Specifications
- **Input:** Pooled intermediate representation $h_t^{(l-1)} \in \mathbb{R}^D$ entering adapted layer $l$.
- **Calibration Parameters:** Task centroids $\mu_k^{(l)}$ computed offline via a tiny calibration subset ($N_{\text{cal}} = 64$).
- **Learnable/Tunable Hyperparameters:**
  - **Inertia Coeff ($\eta$):** Bounded in $[0, 1]$ (default: $\eta = 0.50$). This controls the rotation speed. At $\eta = 1$, the router is stateless (rotating instantly to the target); at $\eta = 0$, the router is static (maintaining initial weights).
  - **Gating Temperature ($\tau$):** Used to scale similarity scores during bottom-up coordinate extraction (default: $\tau = 0.10$).
- **State Initialization:**
  - At time step $t=1$, we initialize the boundary condition uniformly:
    $$\mathbf{s}_1^{(L_{\text{frozen}})} = \frac{1}{\sqrt{K}} \mathbf{1} \implies \boldsymbol{\alpha}_1^{(L_{\text{frozen}})} = \frac{1}{K} \mathbf{1}$$
  - For $t > 1$, we propagate the state across time steps by initializing the first adapted layer with the final layer state of the previous sample:
    $$\mathbf{s}_t^{(L_{\text{frozen}})} = \mathbf{s}_{t-1}^{(L)}$$
- **Output:** Dynamically blended activations:
  $$h_t^{(l)} = h_t^{(l-1)} W_{\text{base}}^{(l)} + \sum_{k=1}^K \left(s_{k, t}^{(l)}\right)^2 \left( h_t^{(l-1)} A_k^{(l)} B_k^{(l)} \right)$$

## 5. Baselines
We will evaluate Unitary Geodesic Routing against five state-of-the-art baselines inside the calibrated representation sandbox:
1. **Static Uniform Merging:** Parameter-free baseline that applies constant ensembling weights $\alpha_k = 1/K$ everywhere.
2. **SABLE (Stateless Nearest-Centroid):** Evaluates the performance ceiling under zero-inertia local plasticity, highlighting the trade-off between local joint accuracy and high-frequency routing jitter.
3. **ChemMerge (Stateful Biochemical Kinetics):** The complex metaphorical SOTA baseline. Comparing against ChemMerge will show whether our mathematically pure, Softmax-free, and ODE-free UGR can match or exceed its performance ceiling.
4. **Momentum-Merge (Constant Euclidean EMA):** The minimalist SOTA. Comparing against Momentum-Merge will directly isolate the benefits of curved spherical manifold interpolation over flat Euclidean updates.
5. **PAC-Kinetics:** The learning-theoretic stateful router. Comparing against PAC-Kinetics will evaluate UGR's performance under rigorous optimization constraints.

## 6. Step-by-Step Interaction
For each query $X_t$ in a continuous, sequential stream:

1. **Backbone Feature Extraction:** Pass the input through the frozen shared layers of the backbone network up to layer $L_{\text{frozen}}$ to obtain $h_t^{(L_{\text{frozen}})}$.
2. **Temporal Coupling:** If $t=1$, set $\mathbf{s}_1^{(L_{\text{frozen}})} = \frac{1}{\sqrt{K}} \mathbf{1}$. If $t > 1$, set $\mathbf{s}_t^{(L_{\text{frozen}})} = \mathbf{s}_{t-1}^{(L)}$.
3. **Layer-by-Layer Activation Flow:** For each adapted layer $l = L_{\text{frozen}}+1, \dots, L$:
   - **Extract Hidden State:** Receive $h_t^{(l-1)}$ from the previous layer.
   - **Compute Bottom-Up Target:** Calculate the cosine similarity between $h_t^{(l-1)}$ and each task's layer-wise centroid $\mu_k^{(l)}$. Apply Softmax with temperature $\tau$ to extract raw weights $\mathbf{e}_t^{(l)}$, then normalize to the unit sphere to obtain $\mathbf{w}_t^{(l)}$.
   - **Compute Torque and Rotate:** Calculate the alignment $c_t^{(l)} = (\mathbf{s}_t^{(l-1)})^T \mathbf{w}_t^{(l)}$. If they are collinear, keep the state unchanged. Otherwise, orthonormalize the target component to get $\mathbf{u}_t^{(l)}$, scale the angular distance to $\theta_t^{(l)} = \eta \cdot \arccos(c_t^{(l)})$, and rotate the state:
     $$\mathbf{s}_t^{(l)} = \cos\left(\theta_t^{(l)}\right) \mathbf{s}_t^{(l-1)} + \sin\left(\theta_t^{(l)}\right) \mathbf{u}_t^{(l)}$$
   - **Activation Blending:** Compute the squared elements of the state to obtain ensembling weights $\alpha_{k, t}^{(l)} = (s_{k, t}^{(l)})^2$, and execute the single-pass activation blending forward pass to obtain $h_t^{(l)}$.
   - **Propagate:** Pass $h_t^{(l)}$ to the next layer.
4. **State Persistence:** Store the final state $\mathbf{s}_t^{(L)}$ to initialize the boundary condition of the next sample $X_{t+1}$.
