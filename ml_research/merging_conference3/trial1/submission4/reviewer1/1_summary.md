# Comprehensive Summary of FluidMerge

## 1. Main Topic and Scope
This paper proposes **FluidMerge** (Fluid-Dynamic Parameter Coalescence), a continuous-time framework for post-hoc model merging and test-time adaptation (TTA). The work aims to combine multiple task-specific expert checkpoints—fine-tuned from a shared pretrained base model—into a single, unified, multi-task model without full joint retraining. 

To move beyond the limitations of flat Euclidean merging operators (e.g., static coordinate averages, masking heuristics), the paper frames the trajectory of parameters during test-time adaptation as a continuous-time advection-diffusion fluid flow. The parameters are modeled as a mobile fluid phase flowing and coalescing through a viscous vector field in a non-convex high-dimensional parameter landscape.

---

## 2. Methodology and Technical Approach
The shared parameters $\theta(t)$ are modeled over a virtual time horizon $t \in [0, T]$. The trajectory is governed by a continuous-time advection-diffusion ordinary differential equation (ODE):

$$\frac{d\theta(t)}{dt} = \sum_{k=1}^K w_k(t) \mathbf{F}_k(\theta(t)) + \nu \mathbf{D}(\theta(t))$$

Where:
*   $w_k(t)$: Relative task importance weights at virtual time $t$ (set uniformly as $1/K$ in the experiments).
*   $\mathbf{F}_k(\theta(t))$: **Advection Force Field** that pulls parameters along low-loss streamlines. It is defined as the negative gradient of a self-supervised cross-entropy loss between the student's predictions and stationary, expert-teacher soft labels on unlabeled task-specific test data.
*   $\mathbf{D}(\theta(t))$: **Structural Diffusion (Viscosity)** operator designed to prevent representation tearing. 
    *   *Spatial Laplacian Viscosity (Flawed Baseline):* Computes a discrete 2D spatial Laplacian over weight coordinates.
    *   *Fisher-Information-based Viscosity (Fisher-Ours):* Computes the empirical diagonal Fisher Information coordinates $F_i^{(0)}$ at initialization, generating a coordinate-free restorative force:
        $$[\mathbf{D}_{\text{Fisher}}(\theta)]_i = - F_i^{(0)} \left( \theta_i - \theta_i(0) \right)$$
*   $\nu$: Viscosity coefficient controlling the diffusion-advection balance.

The paper solves this ODE numerically using a **1st-order Euler discretization** with step size $\Delta t = 0.1$ and $N=100$ steps (virtual horizon $T = 10.0$). 

To align classification decision boundaries, downstream task-specific linear heads are optimized in tandem with the shared encoder using standard discrete **Adam optimization**. The system thus operates as a hybrid continuous-discrete dynamical system.

---

## 3. Key Findings
1.  **The Domain Shift Barrier:** Initializing the continuous adaptation directly from the raw pretrained base weights $\theta_0$ yields near-random accuracy (~6.23% average top-1 accuracy). This occurs because task classification heads are highly sensitive to the coordinate-shifted representations of their respective experts.
2.  **Calibration Collapse and Overfitting:** Attempting to adapt the model starting from the raw base weights leads to a rapid explosion of Expected Calibration Error (ECE > 90%). The linear heads quickly overfit to arbitrary high-frequency noise in the unaligned representation space, becoming extremely confident in incorrect predictions.
3.  **Boundary Condition Resolution:** Initializing the continuous trajectory at the **Task Arithmetic** average—termed *Expert-Weighted Initial Boundary Conditions* ($\theta(0) = \theta_{\text{TA}}$)—places the parameters inside high-performing multi-task basins, completely bypassing the representation reconstruction bottleneck.
4.  **Fisher Viscosity Stabilization:** Applying coordinate-dependent spatial Laplacians causes representation tearing and structural decay (54.76% accuracy). In contrast, coordinate-free Fisher-Information Viscosity (EWC) successfully stabilizes the trajectory, achieving the highest average accuracy (**59.34%**) and the lowest calibration error (**7.18% ECE**), outperforming standard $L_2$ weight anchoring and state-of-the-art TTA baselines.

---

## 4. Explicitly Claimed Contributions and Supporting Evidence
The paper explicitly lists four main contributions:
*   **The FluidMerge Perspective:** Modeling model merging as a continuous-time physical fluid flow. Supported by the physical analogies mapping gradient forces to advection and regularizers to viscosity.
*   **Fisher-Information Viscosity & EWC Isomorphism:** Formulating a coordinate-free, permutation-invariant viscosity based on the empirical Fisher metric, and proving its exact mathematical isomorphism under discretized Euler integration to Elastic Weight Consolidation (EWC). Supporting evidence is provided by the step-by-step mathematical derivation in Section 3.6.
*   **Expert-Weighted Initial Boundary Conditions:** Initializing the fluid at the Task Arithmetic average to resolve the domain shift barrier. Supported by the empirical boundary analysis in Section 4.3 (showing a collapse to ~5% from base weights vs. >59% recovery from TA boundary conditions).
*   **Empirical Resolution of the Domain Shift Barrier:** Evaluating FluidMerge against SyMerge, AdaMerging, and Task Surgery across 8 classification datasets using a Vision Transformer (ViT-B-32). Supported by comprehensive metrics in Table 1, showing that FluidMerge (Fisher - Ours) achieves statistically significant improvements in Top-1 Accuracy and ECE under a standardized "Synergy-Refinement Protocol."
