# Paper Summary

## Main Topic
The paper addresses the challenge of combining multiple task-specific expert checkpoints fine-tuned from a shared pretrained base model into a single unified model without retraining (model merging). It proposes **FluidMerge** (Fluid-Dynamic Parameter Coalescence), which views the parameter trajectory during test-time adaptation (TTA) as a continuous-time advection-diffusion fluid flow, bridging physical principles with deep learning optimization.

## Proposed Approach
FluidMerge models the parameter trajectory $\theta(t)$ over a virtual time horizon $t \in [0, T]$ using a continuous-time advection-diffusion ordinary differential equation (ODE):
$$\frac{d\theta(t)}{dt} = \sum_{k=1}^K w_k(t) \mathbf{F}_k(\theta(t)) + \nu \mathbf{D}(\theta(t))$$

Where:
- **Advection ($\mathbf{F}_k$):** Self-supervised task-specific losses (based on cross-entropy with soft-probability pseudo-labels provided by task-specific teacher experts) generate velocity fields pulling weight particles along low-loss streamlines towards task-specific basins.
- **Diffusion ($\mathbf{D}$):** Regularization acting as fluid viscosity to smooth updates. The authors contrast an intuitive but flawed coordinate-dependent 2D spatial Laplacian operator with a coordinate-free, permutation-invariant **Fisher-Information-based Viscosity** based on the empirical diagonal Fisher Information Matrix.
- **Expert-Weighted Initial Boundary Conditions:** Instead of starting from raw base weights $\theta(0) = \theta_0$, the continuous optimization is initialized from the standard Task Arithmetic (TA) average ($\theta_{\text{TA}}$). This bypasses representation misalignment.
- **Dynamic Classification Head Optimization:** The task-specific classification heads are jointly optimized in tandem with the shared encoder weights using an Adam optimizer on the self-supervised loss, forming a hybrid continuous-discrete optimization system.

## Key Findings
1. **Domain Shift Barrier:** Initializing test-time adaptation directly from the raw base weights leads to random-guess classification performance (~5% accuracy) because pretrained representations are unaligned with the task-specific classification heads. 
2. **Pseudo-Label Overfitting:** Attempting post-hoc adaptation on unlabeled data from raw base weights results in rapid overfitting of classification heads to teacher pseudo-labels, causing Expected Calibration Error (ECE) to explode to over 90%.
3. **Synergy-Refinement Success:** Initializing from the Task Arithmetic boundary condition resolves the domain shift barrier, allowing continuous-time trajectory refinement. Under this protocol, FluidMerge with Fisher-Information-based Viscosity achieves **59.34%** average top-1 accuracy across 8 datasets, outperforming static Task Arithmetic (57.74%), head-only tuning (58.12%), and competitive TTA baselines (AdaMerging: 58.04%, Task Surgery: 58.23%, SyMerge: 58.42%, and $L_2$ weight anchoring: 58.48%).
4. **Fisher Viscosity Stabilization:** Fisher Viscosity selectively dampens high-sensitivity parameters, which successfully prevents calibration collapse (maintaining an average ECE of **7.18%**) and out-of-distribution tearing compared to spatial Laplacians (54.76% accuracy, 16.02% ECE) or standard $L_2$ anchoring (58.48% accuracy, 8.75% ECE).

## Explicitly Claimed Contributions (with Evidence)
- **FluidMerge Perspective:** Reimaginig model merging as a continuous-time advection-diffusion fluid flow. Evidence: Mathematical formulation of the ODE and discretization via Euler integration (Section 3).
- **Fisher-Information Viscosity & EWC Isomorphism:** Formulating a permutation-invariant viscosity regularizer and proving its exact mathematical isomorphism to standard Elastic Weight Consolidation (EWC) under Euler discretization. Evidence: Mathematical proof showing the equivalence to standard EWC-regularized gradient descent (Section 3.6).
- **Expert-Weighted Initial Boundary Conditions (Task Arithmetic Initialization):** Addressing representation reconstruction bottlenecks by initializing the fluid at the Task Arithmetic state. Evidence: Table 1 vs. Table 2 results, showing a jump from ~5% to >57% accuracy.
- **Empirical Resolution of the Domain Shift Barrier:** Showing that the joint optimization of representation geometry and classifier boundaries achieves state-of-the-art results. Evidence: Extensive evaluation on 8 classification datasets using a ViT-B-32 backbone (Section 4).
