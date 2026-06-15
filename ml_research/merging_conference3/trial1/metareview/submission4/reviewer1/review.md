# Peer Review Report

---

## 1. Summary of the Paper
This paper introduces **FluidMerge** (Fluid-Dynamic Parameter Coalescence), a continuous-time framework for post-hoc model merging and test-time adaptation (TTA). The primary goal is to merge task-specific expert checkpoints—which have been fine-tuned from a shared pretrained base model—into a single multi-task model on unlabeled test-time data. 

Moving away from flat Euclidean merging operators (e.g., static coordinate averages, sign-matching masks), the authors model the parameter trajectory during test-time adaptation as a continuous physical fluid flow governed by a continuous-time advection-diffusion ordinary differential equation (ODE):

$$\frac{d\theta(t)}{dt} = \sum_{k=1}^K w_k(t) \mathbf{F}_k(\theta(t)) + \nu \mathbf{D}(\theta(t))$$

Where the advection force field $\mathbf{F}_k$ represents the negative gradient of a self-supervised teacher-student loss guided by domain-aligned expert teachers, and the diffusion operator $\mathbf{D}$ acts as a regularizing viscosity to prevent representation tearing. 

The paper identifies a severe **domain shift barrier** when combining pretrained base encoders with task-specific classification heads, leading to random-guessing performance (~6.23% accuracy) and calibration collapse (Expected Calibration Error > 90%) due to pseudo-label overfitting. To address this, the authors propose:
1.  **Expert-Weighted Initial Boundary Conditions**, initializing the optimization starting from the standard Task Arithmetic average weight-vector ($\theta_{\text{TA}}$).
2.  **Fisher-Information-based Viscosity**, a coordinate-wise regularizer based on the empirical diagonal Fisher Information Matrix. 

Under discretized Euler integration, the authors prove that this Fisher viscosity is mathematically isomorphic to running standard gradient descent under an Elastic Weight Consolidation (EWC) penalty anchoring the parameters to their initial Task Arithmetic state.

Using a Vision Transformer (ViT-B-32) across eight image classification datasets under a standardized "Synergy-Refinement Protocol," the authors demonstrate that FluidMerge (Fisher - Ours) achieves the highest multi-task accuracy (**59.34%**), outperforming static Task Arithmetic (**57.74%**), standard $L_2$ weight anchoring (**58.48%**), and competitive test-time adaptation baselines including AdaMerging (**58.04%**) and SyMerge (**58.42%**).

---

## 2. Strengths and Weaknesses

### Major Strengths
*   **Creative Physical Metaphor:** Framing high-dimensional parameter-space adaptation during model merging as a continuous viscous fluid flow is highly creative and provides an elegant conceptual perspective on test-time adaptation dynamics.
*   **De-escalation of Metaphorical Overselling:** The author maintains high scientific integrity by explicitly identifying and proving the exact mathematical equivalences between FluidMerge and standard deep learning techniques. Specifically, proving that under Euler discretization, the continuous advection-diffusion flow is mathematically isomorphic to gradient descent under an EWC penalty is a beautiful, unifying theoretical derivation.
*   **Outstanding Diagnostic Analysis:** Rather than presenting simple performance tables, the paper conducts deep architectural diagnoses. It explains the "domain shift barrier" and "calibration collapse" through standard, mathematically grounded machine learning concepts (non-linear representations, overparameterization, and pseudo-label overfitting), providing immense diagnostic clarity.
*   **Methodological Rigor:** The evaluation is highly rigorous, utilizing appropriate controls (including L2 Weight Anchoring and a Static TA + Head Tuning control) and robust statistical analyses (paired two-tailed t-tests across 8 datasets and low random seed variance).
*   **Exceptional Presentation:** The paper is beautifully structured, logically organized, and written with exemplary clarity and formal notation.

### Major Weaknesses
*   **Inaccurate Coordinate-Free Mathematical Claim:** The paper claims that the empirical diagonal Fisher-Information viscosity is a "coordinate-free" operator on the parameter manifold (Section 3.6). This is mathematically incorrect. A diagonal approximation of a tensor (like the Fisher Information Matrix) assumes coordinate independence and is highly dependent on the choice of basis (standard basis). It is only permutation-invariant under the discrete subgroup of tensor permutations, which is a very weak form of coordinate independence. Rotating the coordinate basis would alter the diagonal Fisher coordinates completely, violating general coordinate covariance.
*   **Conceptual Inconsistency in Physical Analogy:** The paper frames the diagonal Fisher regularizer as a parameter-space "diffusion" or "viscosity" operator. However, in physical mechanics, diffusion and viscosity are dissipative/transport phenomena that couple adjacent spatial coordinates (e.g., via a spatial Laplacian $\nabla^2 \theta$) to transfer momentum or concentration. The diagonal Fisher force:
    $$[\mathbf{D}_{\text{Fisher}}(\theta)]_i = - F_i^{(0)} \left( \theta_i - \theta_i(0) \right)$$
    is completely decoupled across coordinates (no cross-coordinate terms). Mathematically, this acts as a set of independent **point-wise restorative spring forces** (a conservative harmonic trap pulling each coordinate back to its initial position) rather than a physical viscous fluid damping force. Calling a decoupled harmonic potential "diffusion" or "viscosity" is a conceptual stretch of the physics analogy.
*   **Astronomical Computational Overhead vs. Marginal Gains:** The proposed continuous-time full-encoder adaptation is highly impractical for real-world deployments. Standard Task Arithmetic is completely free ($0$ seconds and $0$ GB memory). Kept completely frozen, standard Task Arithmetic with cheap **Head-Only Tuning** yields **58.12%** accuracy at virtually zero computational cost. In contrast, FluidMerge (Fisher - Ours) requires **20.5 minutes** of NVIDIA A100 GPU compute, **14.8 GB** of GPU memory, and full-encoder backpropagation through 86M parameters over 100 epochs, including the evaluation of 8 separate teacher models. The absolute accuracy gain over Head-Only Tuning is a meager **1.22%**, which does not justify the massive computational bottleneck.
*   **Limited LLM Evaluation:** The "LoRA-FluidMerge" extension in Appendix A is evaluated on a tiny model scale (OPT-125M) and only two distinct domains ($K=2$). The validation cross-entropy loss reduction over Task Arithmetic is extremely minor (**0.0201** average loss points), and the paper fails to evaluate whether this tiny reduction translates to any meaningful downstream functional performance (e.g., accuracy on medical QA or coding tasks).

---

## 3. Soundness
**Rating: Good**

### Justification:
The paper is technically solid, and the mathematical derivations proving the isomorphism between Euler discretization and the EWC update step are completely correct. The self-supervised routing of data and teacher-student guidance is appropriate. However, the soundness is prevented from being "Excellent" due to:
1.  The mathematically incorrect claim that a diagonal empirical Fisher Information Matrix is "coordinate-free."
2.  The conceptual stretching of the physical fluid-mechanics analogy, labeling a coordinate-wise restorative spring force as "viscosity" or "diffusion."
3.  The lack of formal convergence proofs or stability analyses for the hybrid continuous-discrete coupled dynamical system (Euler-integrated encoder coupled with Adam-optimized classification heads).

---

## 4. Presentation
**Rating: Excellent**

### Justification:
The paper is exceptionally clearly written, logical, and beautifully structured. The abstract perfectly summarizes the core contributions, the related work situates the paper within the literature, and the methodology is described with exemplary mathematical clarity. Tables 1 and 2 are highly informative, capturing both Top-1 Accuracy and Expected Calibration Error (ECE) across standardized setups. The authors are highly transparent about their computational footprints and the limitations of their continuous-time formulations.

---

## 5. Significance
**Rating: Fair**

### Justification:
*   **Theoretical Significance:** High. By establishing a bridge between continuous fluid mechanics, test-time adaptation, and Bayesian continual learning, the paper provides a novel, highly creative framework. It establishes a valuable maximum-capacity upper-bound for what is achievable when the entire parameter manifold is adapted post-hoc.
*   **Practical Significance:** Low. In real-world edge or low-latency environments, full-encoder backpropagation for 20.5 minutes is completely unviable. Since cheap Head-Only Tuning captures over 90% of the adaptation benefit, the practical utility of FluidMerge is highly limited. The low-rank LoRA-FluidMerge extension in the Appendix is a promising step to address this, but its empirical validation on LLMs is currently too small-scale to demonstrate major significance.

---

## 6. Originality
**Rating: Good**

### Justification:
Modeling the post-hoc merging trajectory of *neural weights* as continuous fluid advection-diffusion is highly original and represents a significant conceptual departure from standard flat algebraic operators. While the final discretized implementation reduces strictly to standard machine learning baselines (EWC and Task Arithmetic), the conceptual framing and the taxonomy developed are novel and intellectually stimulating.

---

## 7. Overall Recommendation
**Rating: 4: Weak Accept**

### Justification:
This is a technically solid, highly original, and beautifully written paper that advances the conceptual understanding of test-time model merging. The authors deserve significant credit for their scientific integrity in explicitly proving the mathematical equivalences of their methods to EWC and Task Arithmetic, rather than hiding behind physical metaphors. Additionally, their architectural diagnosis of domain shift barriers and calibration collapse represents an outstanding contribution to scientific transparency. 

However, the paper is held back by the astronomical computational overhead of full-encoder adaptation, the limited scale of its LLM evaluations, and minor mathematical misstatements regarding coordinate independence and physical viscosity. It is a technically solid paper that others are highly likely to build on, making a "Weak Accept" highly appropriate, provided the authors address the requested theoretical and empirical revisions.

---

## 8. Constructive Suggestions for Revisions
1.  **Correct the Mathematical Claim regarding Diagonal FIM:** The authors should correct the claim that the diagonal empirical Fisher is "coordinate-free." They should explicitly clarify that it is basis-dependent and only invariant under coordinate-wise permutations (a discrete subgroup), whereas only the full FIM defines a coordinate-free Riemannian metric.
2.  **Refine the Physical Analogy Terminology:** The authors should clarify the physical distinction in their formulation. Rather than framing the diagonal Fisher operator strictly as a "viscosity/diffusion" force (which physically requires cross-coordinate coupling), they should explicitly label it as a coordinate-wise **harmonic restorative spring force**. Correcting this terminology will align the physical analogy with the actual mathematical operations.
3.  **Scale the LLM and Low-Rank Evaluations:** To make the parameter-efficient "LoRA-FluidMerge" extension convincing, the authors should scale the OPT-125M evaluation to more than 2 tasks. Crucially, they should evaluate downstream functional performance (e.g., accuracy on medical QA or coding tasks) rather than strictly reporting validation cross-entropy losses, as a loss delta of 0.0201 is too minor to be statistically or functionally meaningful.
4.  **Discuss the Efficiency/Complexity Trade-offs:** The authors should bring the computational complexity and runtime analysis (Section 4.5 and Table 2) into the main text rather than keeping it buried in the Appendix (if space permits) to ensure that the practical limitations and the marginal 1.22% accuracy gain over cheap Head-Only Tuning are transparently presented from the outset.
