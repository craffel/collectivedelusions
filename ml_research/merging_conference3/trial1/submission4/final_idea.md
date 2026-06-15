# Idea Proposal: FluidMerge (Fluid-Dynamic Parameter Coalescence)

## 1. Persona Alignment
*FluidMerge* aligns perfectly with **The Visionary** persona. Instead of making incremental, coordinate-wise adjustments or tuning standard static merging coefficients in flat Euclidean space (the conventional way merging is approached), *FluidMerge* represents a radical paradigm shift: it models the entire model merging process as a continuous-time physical fluid-dynamic simulation. We view the parameter space as a non-convex Riemannian landscape through which expert models flow and coalesce as viscous fluids. By drawing inspiration from the physics of Navier-Stokes fluid mixtures and advection-diffusion flows, this approach takes a high-risk, curiosity-driven leap to solve parameter interference and discover optimal multi-task basins that simple linear weight math can never reach.

## 2. Core Techniques
The core algorithms and mechanisms introduced in `FluidMerge` include:
1.  **Continuous-Time Parameter Advection:** Parameters are treated as mobile particles advecting through a velocity field. Task-specific advection forces $\mathbf{F}_k(\theta)$ are generated using gradients of the task-specific expert-supervised losses.
2.  **Layer-wise Laplacian Viscous Regularization:** A parameter-space Laplacian diffusion operator $\mathbf{D}(\theta) = \Delta \theta$ acts as physical fluid viscosity. It couples adjacent weight coordinates (such as adjacent channels or neurons), preventing individual weight updates from tearing apart the functional structure of the network and ensuring cohesive representation alignment.
3.  **Self-Supervised Force Guidance:** Since ground-truth labels are unavailable at test time, advection forces are guided by a self-labeling objective using predictions from the individual, fine-tuned expert teachers (extending the teacher-student paradigm of *SyMerge* (Jung et al., 2025)).
4.  **Numerical Flow Solver:** An integration scheme (such as 2nd-order Runge-Kutta or Euler integration) is employed to solve the continuous trajectory $\theta(t)$ from $t=0$ (the pre-trained base model) to $t=1$ (the fully coalesced synergistic model).

## 3. Mathematical Formulation

Let $\theta_0$ represent the weights of the pre-trained base model, and $\theta_1, \theta_2, \dots, \theta_K$ represent the weights of the fine-tuned task-specific experts.

We define a virtual time trajectory $\theta(t)$ for $t \in [0, 1]$, where the evolution of the weights is governed by the following continuous-time advection-diffusion ordinary differential equation (ODE):

$$\frac{d\theta(t)}{dt} = \sum_{k=1}^K w_k(t) \mathbf{F}_k(\theta(t)) + \nu \mathbf{D}(\theta(t))$$

where:
*   $w_k(t)$ is a time-varying weight matching the importance of task $k$ at time $t$.
*   $\nu$ is the viscosity coefficient controlling the strength of the parameter diffusion.
*   $\mathbf{F}_k(\theta(t))$ is the task-gradient advection force field, defined as the negative gradient of the self-labeled teacher loss for task $k$:
    $$\mathbf{F}_k(\theta) = -\nabla_\theta \mathcal{L}_k(\theta; X_{\text{unlabeled}})$$
    where $\mathcal{L}_k(\theta) = \mathcal{H}(C_k^{\text{merged}}(X_{\text{unlabeled}}; \theta), C_k^{\text{ft}}(X_{\text{unlabeled}}))$ is the cross-entropy loss between the merged model's predictions and the expert model's fixed predictions.
*   $\mathbf{D}(\theta(t))$ is the structural diffusion operator, discretized as a spatial Laplacian over the weight coordinates. For a 2D weight matrix $W \in \mathbb{R}^{d_{\text{in}} \times d_{\text{out}}}$, the Laplacian at index $(i, j)$ couples adjacent weights to smooth out gradient discontinuities:
    $$[\mathbf{D}(W)]_{i,j} = W_{i+1,j} + W_{i-1,j} + W_{i,j+1} + W_{i,j-1} - 4W_{i,j}$$

### Numerical Discretization (Euler Integration)
We discretize the continuous trajectory over $N$ steps with step size $\Delta t = 1/N$:

$$\theta_{n+1} = \theta_n + \Delta t \left( \sum_{k=1}^K w_k(t_n) \mathbf{F}_k(\theta_n) + \nu \mathbf{D}(\theta_n) \right)$$

where $\theta_1 = \theta_0$ (the pre-trained model initialized at step 1), and $\theta_{N+1}$ yields the final merged weights $\theta_{\text{final}}$.

## 4. Architecture Specifications
*   **Inputs:**
    *   Pre-trained base model weights $\theta_0$.
    *   Fine-tuned expert model weights $\theta_1, \theta_2, \dots, \theta_K$.
    *   Unlabeled test data batch $X_{\text{unlabeled}}$.
*   **Intermediate Representations:**
    *   *Advection Field Tensor:* Same shape as the network weights, storing the current composite gradient forces.
    *   *Viscosity Tensor:* Same shape as the network weights, storing the layer-wise Laplacian diffusion values.
*   **Parameters:**
    *   Viscosity coefficient $\nu \in [10^{-4}, 10^{-2}]$ controlling local smoothing.
    *   Step size $\Delta t \in [0.05, 0.2]$ governing integration fidelity.
*   **Outputs:**
    *   The coalesced multi-task model weights $\theta_{\text{final}}$ that reside in the optimal, smooth intersection of the task-specific loss basins.

## 5. Baselines
We evaluate `FluidMerge` against the following state-of-the-art baselines:
1.  **Task Arithmetic (Ilharco et al., 2023):** Standard linear addition in Euclidean space. Appropriate as the foundational baseline to demonstrate the superiority of non-linear physical trajectories.
2.  **Ties Merging (Yadav et al., 2024):** Resolves sign conflicts and performs pruning. Appropriate to verify if physical viscosity performs better than heuristic parameter pruning.
3.  **SyMerge (Jung et al., 2025):** The direct precursor that adaptively tunes a single task-specific layer under self-labeling. Appropriate to show the benefit of merging all layers continuously.
4.  **OrthoMerge (Yang et al., 2026):** Merges orthogonal transformations in Lie algebra. Appropriate to compare manifold-constrained merging with continuous fluid-dynamic mixing.

## 6. Step-by-Step Interaction
1.  **Initialization:** Set $n = 0$ and initialize the merged weights $\theta_n = \theta_0$ (the pre-trained base checkpoint).
2.  **Forward Pass & Soft Labels:** For each task $k$, pass the unlabeled batch $X_{\text{unlabeled}}$ through the fixed expert model $\theta_k$ to obtain fixed soft-label predictions $C_k^{\text{ft}}$.
3.  **Current Prediction:** Pass the unlabeled batch through the current merged model $\theta_n$ to obtain current predictions $C_k^{\text{merged}}(\theta_n)$.
4.  **Force Computation:** Compute the self-labeled teacher cross-entropy loss $\mathcal{L}_k(\theta_n)$ and its gradient with respect to $\theta_n$ to construct the advection forces $\mathbf{F}_k(\theta_n)$.
5.  **Diffusion Computation:** Compute the Laplacian $\mathbf{D}(\theta_n)$ on the weight coordinates to calculate the viscosity forces.
6.  **Euler Update:** Combine the forces and update the weights to obtain $\theta_{n+1}$.
7.  **Iterate:** Repeat steps 3–6 for $N$ steps until $n = N$.
8.  **Output:** Return $\theta_{N+1}$ as the final synergized model.
