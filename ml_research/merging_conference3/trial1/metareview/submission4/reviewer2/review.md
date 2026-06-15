# Peer Review

## 1. Summary of the Paper
This paper addresses the challenge of multi-task model merging, which aims to fuse several task-specific expert checkpoints fine-tuned from a shared pretrained model into a single unified model without retraining. To go beyond flat, static Euclidean operations (such as Task Arithmetic or Ties-Merging), the paper proposes **FluidMerge** (Fluid-Dynamic Parameter Coalescence). This framework models the path of model parameters during post-hoc test-time adaptation (TTA) on unlabeled data as a continuous-time advection-diffusion fluid flow. 

The parameter trajectory is governed by two main physical forces:
- **Advection:** Self-supervised task-specific losses based on cross-entropy with soft pseudo-labels provided by stationary task-expert teachers.
- **Diffusion:** Localized regularization that acts as fluid viscosity. The authors introduce a function-sensitive, coordinate-free, and permutation-invariant **Fisher-Information-based Viscosity** based on the empirical diagonal Fisher Information Matrix.

To overcome representational domain shifts, the paper implements:
- **Expert-Weighted Initial Boundary Conditions:** Initializing the continuous-time parameter optimization from the Task Arithmetic weight-space average ($\theta_{\text{TA}}$) rather than raw pretrained base weights ($\theta_0$).
- **Dynamic Classification Head Optimization:** Jointly optimizing the linear classification heads in tandem with the shared encoder weights, resulting in a hybrid continuous-discrete dynamical system.

The method is evaluated using a Vision Transformer (ViT-B-32) backbone across eight image classification datasets. Under the Synergy-Refinement Protocol, FluidMerge with Fisher Viscosity achieves **59.34%** average top-1 accuracy, outperforming static Task Arithmetic (**57.74%**), head-only adaptation (**58.12%**), standard $L_2$ anchoring (**58.48%**), and competitive TTA baselines (AdaMerging: **58.04%**, Task Surgery: **58.23%**, and SyMerge: **58.42%**).

---

## 2. Strengths and Weaknesses

### Strengths
1. **Compelling and Intuitive Conceptual Metaphor:** Modeling parameter trajectories as continuous physical fluid flows is a creative perspective. The paper uses this analogy effectively to construct a joint advection-diffusion ODE in parameter space.
2. **High Level of Scientific Transparency:** The authors are exceptionally honest about de-escalating metaphorical overselling. They explicitly show the exact mathematical equivalences between their physical fluid mechanisms and standard deep learning components (e.g., Fisher viscosity under Euler integration is mathematically isomorphic to gradient descent under an Elastic Weight Consolidation (EWC) penalty). This makes the work highly rigorous and easy to interpret.
3. **Thorough Diagnostic Analysis:** The paper includes a systematic, rigorous diagnosis of the "domain shift barrier" and the overfitting failure mode (ECE explosion) of unaligned models under TTA. This is an extremely valuable practical insight for researchers and engineers working on post-hoc adaptation.
4. **Comprehensive Evaluation Protocol:** The Synergy-Refinement Protocol provides a fair, standardized comparison by initializing all methods under the same viable boundary conditions ($\theta_{\text{TA}}$).
5. **Practical Profiling and LoRA Validation:** Including detailed wall-clock times, GPU memory footprint (Table 3), and implementing a parameter-efficient LoRA variant show a strong commitment to practical utility, resource constraints, and scalability.

### Weaknesses
1. **High Computational Overhead for Marginal Practical Gains:**
   While FluidMerge achieves state-of-the-art results, the cost-benefit ratio is highly unfavorable for practical deployment. Full-encoder adaptation requires computing empirical Fisher coordinates and backpropagating gradients through the entire 86M parameter image encoder over 100 epochs, while running forward passes through $K=8$ separate teacher experts. This leads to a massive execution time of **20.5 minutes** on an A100 GPU and **14.8 GB** of memory overhead. 
   Crucially, this complex setup only outperforms the control ablation **Static TA + Head-Only Tuning** (which runs instantly at zero encoder compute) by **1.22%** absolute top-1 accuracy (59.34% vs. 58.12%). In real-world edge or production environments, such an extreme computational overhead is highly unlikely to be justified for a ~1.2% gain.
2. **Omission of LoRA-FluidMerge Performance Metrics:**
   To address the computational bottleneck, the authors implement a parameter-efficient LoRA-FluidMerge variant in Section 4.6 and demonstrate a 64.1$\times$ parameter reduction and a 1.32$\times$ speedup. However, they **do not report any classification accuracy or ECE results** for this configuration. Without confirming whether the low-rank constraint maintains the 59.34% multi-task performance or causes representation decay, this practical validation remains incomplete.
3. **Simplified Diagonal Fisher Metric:**
   The viscosity operator relies on a standard diagonal approximation of the Fisher Information Matrix, assuming complete independence across parameter coordinates. In high-dimensional networks, parameter updates are highly correlated across dimensions (channels, layers, heads). Consequently, the current operator acts as coordinate-wise restorative spring forces (harmonic anchoring) rather than a physical spatial viscosity.
4. **Numerical Instability of Euler Solver:**
   Discretizing the continuous ODE using a first-order Euler scheme with a large, fixed step size ($\Delta t = 0.1$) lacks numerical stability guarantees for non-convex loss surfaces. The integration could easily drift or diverge on "stiff" parameter trajectories without adaptive solvers.
5. **No Evaluation on Mixed/Noisy Streams:**
   The paper proposes a confidence-based entropy filter to mitigate OOD noise during adaptation but disables it ($\tau = \infty$) in all primary experiments by routing clean in-distribution data. In actual production deployment, test-time adaptation data is rarely pre-sorted, and evaluating this filter under noisy/mixed streams is critical to establish its robustness.

---

## 3. Soundness
**Rating: Good**

**Justification:**
The proposed methodology is technically solid and theoretically well-grounded. The mathematical proof of isomorphism between Fisher-Information viscosity and Elastic Weight Consolidation (EWC) under discretized Euler integration is flawless. The joint optimization of representation geometry (encoder) and decision boundaries (classification heads) is highly appropriate. However, the rating is capped at "Good" rather than "Excellent" due to:
- The coordinate-wise independence assumption of the diagonal Fisher approximation.
- The use of a simple, fixed-step first-order Euler solver without adaptive step-size error control.
- The lack of empirical evaluation for the confidence-based filtering mechanism.

---

## 4. Presentation
**Rating: Excellent**

**Justification:**
The presentation quality is outstanding. The paper is clearly written, logically structured, and exceptionally transparent. The related work is highly thorough, and the authors make an admirable effort to explain their physical metaphors in plain machine learning terms. The figures, equations, and tables are clean and comprehensive. The inclusion of Expected Calibration Error (ECE) alongside Top-1 Accuracy provides excellent multi-dimensional visibility into the adaptation dynamics.

---

## 5. Significance
**Rating: Good**

**Justification:**
The paper addresses an important problem (post-hoc model merging and TTA) and provides highly valuable diagnostic insights regarding representational mismatch and calibration collapse. The conceptual perspective of weight-space fluid dynamics is likely to influence future research in continuous-time merging operators. However, the significance for practical applications is somewhat limited by:
- The high computational budget (20.5 minutes on an A100) required to achieve a marginal ~1.2% improvement over a cheap head-tuning baseline.
- The scale limitations: running full-encoder backpropagation for multi-task TTA on standard 86M parameter ViT backbones is already expensive, and scaling this to modern LLMs/VLMs with billions of parameters is currently computationally prohibitive.

---

## 6. Originality
**Rating: Good**

**Justification:**
Reimagining weight-space trajectories as continuous fluid flows is highly creative, and the "permutation invariance paradox" and "Fisher-Information viscosity" are elegantly formulated. However, once the mathematical wrappers are peeled back, the actual operational elements represent a combination of existing, well-established deep learning techniques: Task Arithmetic initialization, teacher-student distillation, and EWC regularization. Therefore, the originality lies in the novel conceptual packaging and joint pipeline integration rather than entirely new optimization primitives.

---

## 7. Overall Recommendation
**Rating: 4: Weak Accept**

**Justification:**
The paper is technically solid, exceptionally written, and scientifically honest. It provides highly valuable diagnostic insights into test-time adaptation dynamics and successfully out-performs competitive baselines under a standardized protocol. However, from a practical and deployment perspective, its high computational complexity (20.5 minutes on an A100) and small accuracy gains over frozen-encoder head-only tuning limit its immediate industrial utility. The paper is a strong candidate for publication as it establishes a valuable capacity upper-bound and a compelling continuous-time framework that others are likely to build upon, but the authors must address the practical and scalability limitations to fully unlock its impact.

---

## 8. Constructive Questions & Suggestions for Authors

1. **Provide Empirical Results for LoRA-FluidMerge:**
   To complete the practical validation in Section 4.6, please report the average Top-1 Accuracy and ECE achieved by the LoRA-FluidMerge configuration. Does the 64.1$\times$ parameter constraint degrade or preserve the 59.34% multi-task accuracy?
2. **Discuss Edge/Production Deployment Constraints:**
   Given that Static TA + Head-Only Tuning achieves **58.12%** accuracy instantly at zero encoder cost, and FluidMerge achieves **59.34%** at the cost of **20.5 minutes** of full-encoder backpropagation on an A100 GPU:
   - In what realistic industrial or production scenarios do you envision full-encoder FluidMerge being deployed?
   - How can practitioners trade off this adaptation accuracy and compute budget dynamically?
3. **Evaluate under Mixed/Noisy Test-Time Streams:**
   To verify the utility of the confidence-based filtering threshold ($\tau$), please conduct a diagnostic experiment where the test-time adaptation stream is corrupted with out-of-distribution or noisy images, and demonstrate how varying $\tau$ affects trajectory stability and validation accuracy.
4. **Compare with Adaptive ODE Solvers:**
   For continuous-time rigor, have you explored adaptive step-size solvers (e.g., Runge-Kutta 4(5) or Dormand-Prince) via packages like `torchdiffeq`? Discuss whether adaptive step-sizing improves numerical stability or reduces the number of required integration steps below $N=100$.
5. **Address Scale Limitations:**
   Please add a brief discussion on how FluidMerge could scale to larger architectures (e.g., billions of parameters). Would block-wise approximations, Kronecker-Factored Fisher metrics (K-FAC), or selective layer-wise fluid flows be necessary to make the method computationally viable at scale?
