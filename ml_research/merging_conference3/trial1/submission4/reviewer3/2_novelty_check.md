# Evaluation Component 2: Novelty and Related Work Check

## 1. Characterization of Novelty
The novelty of this submission is **conceptual and metaphorical**, rather than representing a completely new set of mathematical primitives. 

* **Conceptual Novelty (Significant):** Re-imagining the trajectory of neural network parameters during post-hoc test-time adaptation as a continuous physical fluid-dynamic system (advection-diffusion flow) is a highly creative, elegant, and engaging framing. Bridging deep learning weight optimization with the physics of viscous flows is a fresh perspective that expands how we conceptualize parameter trajectory geometry.
* **Technical/Algorithmic Novelty (Moderate/Incremental):** When stripped of the physical metaphors, the successful, high-performing components of the proposed "FluidMerge" method consist almost entirely of existing, standard deep learning techniques:
    1. **Expert-Weighted Initial Boundary Conditions** is mathematically identical to starting optimization from standard **Task Arithmetic** (Ilharco et al., 2023).
    2. **Fisher-Information-based Viscosity** is mathematically isomorphic to a standard **Elastic Weight Consolidation (EWC)** penalty (Kirkpatrick et al., 2017) that anchors parameters to the Task Arithmetic state during adaptation.
    3. **Advection Force Field** is a standard self-supervised student-teacher cross-entropy objective using teacher soft labels, building closely on the self-labeling framework of **SyMerge** (Jung et al., 2025).
    4. **Spatial Laplacian** is a flawed, coordinate-dependent baseline that is highly prone to representation tearing (achieving only 54.76% accuracy) and is rejected by the authors themselves due to neural permutation invariance.

Thus, the actual algorithmic "delta" is combining full-encoder TTA self-training (starting from Task Arithmetic) with a diagonal empirical Fisher (EWC) penalty and joint linear head adaptation under a continuous-time ODE solver (Euler discretization). While this combination is well-motivated and empirically effective, the individual mathematical components are heavily grounded in established work.

---

## 2. Positioning and Contextualization Gaps (Scholarly Critique)
As a scholarly review of the literature, the submission has several **critical citation and contextualization gaps** that fail to accurately describe the landscape of the field:

### A. Missing Gradient Flow and Inverse Problem Perspectives (SWUDI / WUDI)
The paper positions FluidMerge as "the first to model the trajectory of the network weights themselves as a continuous, viscous fluid flow." However, it fails to discuss or cite recent and concurrent works that formalize model merging using gradient flow and spectral theory:
* **WUDI-Merging (ICML 2025):** *"Whoever Started the Interference Should End It: Guiding Data-Free Model Merging via Task Vectors"* is a major related work that theoretically demonstrates how task vectors form an approximate linear subspace and addresses parameter interference without training data.
* **SWUDI (arXiv, June 2026):** *"Closed-Form Spectral Regularization for Multi-Task Model Merging"* is an extremely relevant, recent work. SWUDI explicitly treats multi-task model merging as a noisy linear inverse problem solved using **gradient flow**. It proves that iterative descent acts as an implicit **spectral regularizer** and models the gradient-flow trajectory using a closed-form "soft exponential filter" ($1 - e^{-t\lambda_k}$) that corresponds to the stopping time of the gradient flow.
* **The Gap:** By omitting SWUDI and WUDI, the authors miss a highly relevant alternative paradigm that *also* views model merging through continuous-time gradient flows, but achieves a data-free, closed-form spectral solution that is 28–72$\times$ faster. The authors must discuss and position FluidMerge relative to these spectral/gradient flow formulations.

### B. Over-Claim of First-of-its-Kind Metaphor
The paper claims: *"FluidMerge is the first to model the trajectory of the network weights themselves as a continuous, viscous fluid flow."* While the "viscous fluid" terminology is unique to this paper, prior works in continuous-time optimizer design (e.g., Muehlebach & Jordan, 2021) and Hamiltonian/Lagrangian mechanics applied to parameter spaces (e.g., continuous-time momentum as physical drag forces) have modeled network weight trajectories as physical dynamics. The authors should tone down this claim and situate their physical analog within the broader literature on physical and continuous-time perspectives of parameter optimization.

### C. Permutation Symmetries and Coordinate-Free Representations
The paper discusses the "permutation invariance paradox" and "coordinate-free viscosity," citing Git Re-Basin (Ainsworth et al., 2023). However, it does not discuss how other manifold merging approaches (such as GeoMerge, OrthoMerge, and EpiMer) handle these symmetries in relation to continuous trajectories. A deeper discussion on why a diagonal Fisher Information matrix is permutation-invariant (since it relies on coordinate-wise gradients) whereas the spatial Laplacian is not, would strengthen the theoretical positioning of the paper.
