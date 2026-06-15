# Peer Review of Conference Submission

## Paper Summary
This submission addresses the challenge of multi-task model merging—combining task-specific expert neural networks (fine-tuned from a shared pretrained base model) into a single unified model without joint retraining. To address representation misalignment and coordinate conflicts, the authors propose **FluidMerge** (Fluid-Dynamic Parameter Coalescence), a framework that treats parameter test-time adaptation (TTA) as a continuous-time advection-diffusion fluid flow. 

In FluidMerge, self-supervised student-teacher cross-entropy gradients act as "advection forces" pulling parameters towards task-specific basins, while parameter-space diffusion acts as "fluid viscosity" to prevent representational tearing. To resolve neural permutation symmetries, the authors replace a coordinate-dependent 2D spatial Laplacian with a coordinate-free, function-sensitive viscosity based on the diagonal empirical Fisher Information Matrix. Under discretized Euler integration, this Fisher viscosity is proved to be mathematically isomorphic to running gradient descent under an Elastic Weight Consolidation (EWC) penalty relative to the starting weights. The authors also formulate Expert-Weighted Initial Boundary Conditions, which are mathematically equivalent to initializing optimization from the standard Task Arithmetic average (Ilharco et al., 2023).

Through a standardized **Synergy-Refinement Protocol** on a Vision Transformer (ViT-B-32) across 8 classification datasets, FluidMerge-Fisher achieves an average Top-1 accuracy of **59.34%**, outperforming static Task Arithmetic (57.74%), L2 weight anchoring (58.48%), and dynamic TTA baselines (AdaMerging, SyMerge, Task Surgery). However, under a diagnostic **Boundary Stress-Test** initialized at the raw pretrained base weights, all methods fail completely (~5% accuracy) and suffer from Expected Calibration Error (ECE) explosion (>90%) due to pseudo-label overfitting under unaligned representations.

---

## Strengths and Weaknesses

### Strengths
1. **Engaging and Creative Metaphor:** Re-imagining parameter trajectories during test-time adaptation as continuous-time physical fluid-dynamic flows is an exceptionally novel and engaging narrative that bridges deep learning optimization with fluid mechanics.
2. **Outstanding Scientific Transparency:** The authors demonstrate exemplary scientific honesty. Rather than hiding behind physical jargon or overselling their metaphors, they explicitly deconstruct their methods and identify their exact equivalents in standard deep learning literature (Expert-Weighted Boundary Conditions $\equiv$ Task Arithmetic initialization; Fisher Viscosity $\equiv$ continuous-time EWC). This prevents hype and provides direct, demystified value.
3. **Mathematical and Methodological Rigor:** The paper is mathematically solid. It formulates a complete continuous-time ODE framework, specifies detailed Euler discretization steps, and derives a rigorous proof of the mathematical isomorphism between diagonal Fisher-Information viscosity and the EWC penalty.
4. **Comprehensive Benchmarking & Standardized Protocols:** The empirical evaluation uses 8 diverse datasets on a ViT-B-32 backbone and establishes two highly fair, standardized protocols: the Synergy-Refinement Protocol (starting at TA) and the Boundary Stress-Test (starting at base weights). This standardization makes the empirical results highly robust.
5. **Statistical Rigor:** The authors validate their findings using paired two-tailed t-tests across datasets (reporting highly significant improvements, $p < 0.0001$ and $p < 0.001$) and report results over 3 random seeds with extremely low variability (standard deviation $\le 0.15\%$).

### Weaknesses
1. **Critical Related Work and Positioning Gap (Major Scholarly Concern):** While the paper positions FluidMerge as "the first to model the trajectory of the network weights themselves as a continuous, viscous fluid flow," it completely omits a highly relevant and concurrent/recent line of work that formalizes model merging using continuous-time gradient flows and spectral regularization:
    * **WUDI-Merging (ICML 2025):** *"Whoever Started the Interference Should End It: Guiding Data-Free Model Merging via Task Vectors"* theoretically demonstrates how task vectors form an approximate linear subspace and addresses parameter interference without training data.
    * **SWUDI (arXiv, June 2026):** *"Closed-Form Spectral Regularization for Multi-Task Model Merging"* is an extremely relevant work. SWUDI explicitly treats model merging as a noisy linear inverse problem solved by continuous-time **gradient flow**. It proves that iterative descent acts as an implicit spectral regularizer and models the trajectory using a closed-form "soft exponential filter" ($1 - e^{-t\lambda_k}$) that corresponds to the stopping time of the gradient flow.
    * **The Gap:** SWUDI and WUDI offer highly competitive, data-free, closed-form spectral solutions that are **28–72$\times$ faster** and require **50% less memory** than iterative methods. By failing to discuss or cite these spectral gradient-flow formulations, the authors fail to accurately describe the landscape of the field and position their contribution.
2. **Disconnect between Viscosity Metaphor and Diagonal Fisher Implementation:** The authors correctly reject the 2D spatial Laplacian because it violates permutation invariance, but they replace it with a diagonal empirical Fisher Information matrix.
    * In physical fluid dynamics, viscosity is a spatial force that couples adjacent fluid particles, transferring momentum and smoothing out shear forces.
    * A diagonal Fisher matrix assumes complete parameter independence. Consequently, the viscosity force $-F_i^{(0)}(\theta_i - \theta_i(0))$ acts as a coordinate-wise **point-wise restorative spring force (harmonic anchor)**, completely losing any cross-parameter coupling.
    * This means that the high-performing model (FluidMerge-Fisher) **does not actually implement the "viscosity" or physical fluid coupling** originally motivated. It is simply independent parameter-wise EWC anchoring.
3. **Lack of Numerical Stability Analysis:** The continuous-time ODE is solved using a simple 1st-order fixed-step Euler discretization with a relatively large step size of $\Delta t = 0.1$ and $N = 100$ steps. Fixed-step Euler solvers are prone to numerical drift and instability in highly non-convex spaces. The authors do not analyze or quantify the discretization error or trajectory stability of their solver.
4. **Extremely High Computational Cost vs. Modest Gain:** FluidMerge-Fisher requires **20.5 minutes** of premium NVIDIA A100 GPU compute and **14.8 GB** of memory. It yields only a **1.60%** improvement over static Task Arithmetic (0 seconds compute) and a **1.22%** improvement over the "Static TA + Head Tuning" baseline (which takes seconds). For practical edge deployments, full-encoder backpropagation for 20 minutes is completely unviable. While the authors transparently acknowledge this and frame FluidMerge as a "high-capacity research tool," this bottleneck severely limits its practical impact.

---

## Detailed Ratings

### Soundness: Good
The mathematical formulations, Euler discretization updates, and EWC isomorphism derivations are technically correct, and the experiments are rigorous and well-designed. However, the soundness rating is capped at "Good" because:
1. There is a conceptual disconnect between the viscous fluid metaphor and the diagonal Fisher implementation (which loses all parameter-coupling forces and acts as standard independent coordinate-wise anchoring).
2. The paper lacks any analysis of the numerical drift, discretization error, or stability of the 1st-order Euler solver, which represents a significant risk when running full-encoder optimization on a non-convex ViT landscape.

### Presentation: Good
The paper is exceptionally well-written, easy to follow, and engaging, with highly detailed tables and diagnostic analyses. However, the rating is "Good" rather than "Excellent" because of the **critical citation and positioning gap** regarding recent continuous-time gradient-flow model merging theories (such as SWUDI and WUDI-Merging). Additionally, the claims regarding "FluidMerge" resolving the domain shift barrier should be toned down, as the Boundary Stress-Test shows that the Task Arithmetic initialization is what actually bypasses the barrier.

### Significance: Fair
While FluidMerge is of significant theoretical interest as a "maximum-capacity upper bound" for adaptive parameter blending and provides valuable scientific insights into representational alignment, its direct practical significance is low. A 20.5-minute full-encoder backpropagation optimization that yields only a 1.22% accuracy boost over simple, zero-cost head tuning is highly unlikely to be deployed in realistic test-time adaptation settings.

### Originality: Good
Re-imagining model merging as a continuous viscous fluid flow is highly creative and original. However, the rating is "Good" because the actual high-performing technical implementation is a repackaging of standard elements: Task Arithmetic initialization, EWC regularized gradients, and self-supervised student-teacher training.

---

## Overall Recommendation

**Rating: 3: Weak Reject**

**Justification:** 
The paper has clear merits, including an engaging physical metaphor, exemplary scientific honesty regarding mathematical equivalences, and a highly rigorous experimental evaluation. However, the weaknesses currently outweigh these merits. 

Specifically, the paper exhibits a **critical literature positioning gap** by completely omitting highly relevant, concurrent/recent gradient-flow-based merging methods (such as SWUDI and WUDI-Merging) which offer closed-form, data-free spectral solutions that are significantly faster. Additionally, there is a conceptual disconnect between the physical viscosity metaphor and the diagonal Fisher implementation, a lack of numerical stability analysis for the Euler solver, and a severe computational overhead that limits practical viability. 

To transition this paper to a solid accept, the authors must address these limitations in a revision.

---

## Constructive Feedback for the Authors

1. **Address the Related Work & Positioning Gap:**
   * Cite and discuss **WUDI-Merging (ICML 2025)** and **SWUDI (arXiv, June 2026)**.
   * Position FluidMerge relative to SWUDI: explain how viewing merging as a continuous-time advection-diffusion flow compares conceptually to viewing it as a noisy linear inverse problem solved by spectral gradient-flow filtering.
   * Discuss the trade-offs: while FluidMerge uses iterative data-driven updates to adapt both the encoder and head, SWUDI provides a data-free closed-form spectral solution with a 28–72$\times$ speedup.
2. **Clarify the Viscosity vs. Diagonal Fisher Disconnect:**
   * Be explicit about the fact that a diagonal Fisher matrix loses cross-parameter spatial coupling, and explain that "viscosity" here behaves mathematically as independent coordinate-wise restorative spring forces (harmonic anchoring).
   * Discuss how future work could restore true physical coupling or structured viscosity across layers using Kronecker-Factored Approximate Curvature (K-FAC).
3. **Provide Numerical Solver Stability Analysis:**
   * Add an empirical analysis or discussion regarding the discretization error and numerical stability of the Euler solver with $\Delta t = 0.1$.
   * Validate if smaller step sizes (e.g., $\Delta t = 0.01, N = 1000$) or adaptive solvers (Dormand-Prince) improve trajectory accuracy or final multi-task performance.
4. **Tone Down Domain Shift Claims:**
   * Refine statements claiming that FluidMerge "resolves" the domain shift barrier. Clearly clarify that the continuous-time trajectory serves as an *adaptive refinement* step on top of a viable initialization, and that the Expert-Weighted (Task Arithmetic) initialization is what actually bypasses the barrier.
