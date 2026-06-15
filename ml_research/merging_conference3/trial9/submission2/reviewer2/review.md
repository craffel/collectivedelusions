# Peer Review for Submission: Resource-Budgeted Top-M Expert Serving (RB-TopM)

## 1. Summary of the Paper
This paper presents **Resource-Budgeted Top-$M$ Expert Serving (RB-TopM)**, a training-free, hardware-aware dynamic model ensembling framework designed for resource-constrained edge serving. The framework addresses the computational and memory-bandwidth bottlenecks of multi-task ensembling on edge devices by introducing a system-level control loop. Governed by a real-time resource availability coefficient $C_{\text{budget}} \in [0, 1]$, the control loop dynamically scales:
1. **A Dynamic Top-$M$ Cap ($M(C_{\text{budget}})$):** A ceiling restricting the maximum number of active parallel expert pathways, forcing a smooth transition from soft-blending to hard single-expert routing.
2. **An Adaptive Gating Threshold ($\theta(C_{\text{budget}})$):** A dynamic threshold that filters out low-contribution adapter pathways whose routing coefficients fall below $\theta$.

Additionally, the authors integrate a Coordinate diagonal Gaussian Mixture Model (GMM) safety shield in the early representation space (Layer 3) to flag out-of-distribution (OOD) queries, setting all expert routing coefficients to zero for flagged queries and defaulting execution to the pre-trained base model. The method is validated on a synthetic 14-layer Analytical Coordinate Sandbox (ICS) and a TVM compiler-level simulation pilot running MobileNetV3-Large on DomainNet.

---

## 2. Strengths and Weaknesses

### Strengths
1. **Pragmatic and Systems-Aware Motivation:** The paper directly addresses a critical bottleneck in TinyML and edge serving—specifically, the off-chip DRAM memory bandwidth required to fetch specialized low-rank expert weights, which is a major driver of latency and thermal throttling.
2. **The "Activation Dilution" Insight:** The discovery and exploration of the non-monotonic accuracy-latency trade-off (where lower budget settings like $C_{\text{budget}} = 0.4$ outperform full ensembling at $C_{\text{budget}} = 1.0$) is a highly interesting and counter-intuitive phenomenon.
3. **Outstanding Presentation and Structural Clarity:** The paper is exceptionally well-structured. The narrative is easy to follow and is supported by a comprehensive physical/logical data flow diagram (Figure 2), a helpful notational glossary (Table 1), and complete pseudocode in the appendix.
4. **Exhaustive Appendix and Sensitivity Sweeps:** The paper provides highly thorough sensitivity sweeps across hyperparameters (softmax temperatures, number of GMM components, calibration split sizes) and scales the expert population up to $K=24$, ensuring complete academic transparency.

### Weaknesses
1. **Severe Mathematical Shortcuts and Lack of Theoretical Rigor:** The mathematical proofs and derivations in the appendix (Appendix A.1) rely on highly unrealistic simplifying assumptions, such as modeling complex, non-linear deep representation features as isotropic white Gaussian noise and assuming statistical independence where strong dependencies exist. This compromises the validity of the theoretical claims.
2. **Inconsistent Probability Support on Bounded Domains:** The Gaussian Mixture Model safety shield is fitted over cosine similarity coordinates that are strictly bounded in $[-1, 1]^K$. Fitting a standard multivariate Gaussian Mixture Model (which assumes infinite support $\mathbb{R}^K$) over a bounded domain is mathematically inconsistent.
3. **Over-Reliance on a Synthetic, Orthogonal Sandbox:** The core accuracy and OOD protection sweeps are evaluated on a synthetic "14-layer Analytical Coordinate Sandbox" (ICS) where task domains are artificially projected onto orthogonal subspaces. This simplified sandbox completely lacks the complex, non-linear activation dynamics and manifold overlaps of real deep neural networks, heavily inflating the early router's performance.
4. **Heuristic Nature of the Control Loop:** The dynamic scaling equations for $M(C_{\text{budget}})$ and $\theta(C_{\text{budget}})$ are simple linear interpolations and floor functions. The framework lacks any control-theoretic foundation (e.g., Lyapunov stability or optimal control theory) to prove that these linear relationships are optimal or stable.

---

## 3. Soundness

**Rating: Fair**

While the engineering design is logical, the paper falls short of the rigorous standards of mathematical and theoretical soundness. Specifically:

### 3.1. Phenomenological and Heuristic Modeling of Interference
In Appendix A.1, the authors attempt to mathematically prove the "activation dilution" phenomenon. They model the interference of a secondary expert adapter $k \neq k^*$ on the clean, task-aligned input $h_0^{(l-1)}$ as a zero-mean white Gaussian noise term:
$$e_k^{(l)} = h_0^{(l-1)} A_k^{(l)} B_k^{(l)} \sim \mathcal{N}(0, \sigma_{\text{inter}}^{(l)2} I_D)$$
Furthermore, they model the layer-wise environmental noise as isotropic white noise $\epsilon_l \sim \mathcal{N}(0, \sigma_{\text{env}}^2 I_D)$ at every layer. 
* **Critique:** In real-world deep neural networks, intermediate representations and interference signals are highly non-linear, non-Gaussian, and heavily biased due to deep non-linear activation functions (e.g., ReLU, GeLU, SiLU). Modeling these highly structured latent features as simple Gaussian white noise is mathematically naive and phenomenological rather than derived from fundamental principles.

### 3.2. Unjustified Independence Assumptions in Covariance Derivation
In the ensembled representation covariance derivation (Equation 19):
$$\text{Cov}\left( Y^{(l)} \right) \approx \alpha_{k^*}^2 \text{Cov}(y_{k^*}^{(l)}) + \sum_{k \neq k^*} \left( \bar{\alpha}_k^2 + \text{Var}(\delta_k) \right) \left[ \sigma_{\text{inter}}^{(l)2} I_D + \sigma_{\text{env}}^2 (A_k^{(l)} B_k^{(l)}) (A_k^{(l)} B_k^{(l)})^T \right]$$
The authors assume that the routing noise $\delta_k$, the interference $e_k^{(l)}$, and the environmental noise $\epsilon_l$ are mutually independent.
* **Critique:** The routing coefficients $\alpha_k$ (and thus the noise $\delta_k$) are computed directly from the early activations at Layer 3, which contain the exact same environmental noise propagated from the input. Therefore, the routing coefficients and the deeper activation representations are heavily dependent. Ignoring this dependency and omitting the cross-covariance terms in Equation 19 is a major mathematical shortcut that compromises the validity of the "activation dilution" proof.

### 3.3. Bounded Space support of the GMM
The Coordinate GMM safety shield is fitted over cosine similarity coordinates $u'_b \in [-1, 1]^K$.
* **Critique:** A Gaussian Mixture Model is defined on $\mathbb{R}^K$ and assumes infinite support. Fitting it over a bounded space $[-1, 1]^K$ is mathematically inconsistent. While the authors present directional alternatives (such as the von Mises-Fisher distribution), they reject them primarily due to computational complexity on microcontrollers. While pragmatically justified, this leaves the GMM safety shield theoretically ungrounded.

### 3.4. EM Variance Floor Regularization
The EM optimization algorithm is modified by applying a hard covariance floor during parameter estimation:
$$\sigma_{kj}^2 \gets \max\left( \sigma_{kj}^2, \epsilon \right)$$
* **Critique:** Artificially flooring variances during the M-step of the EM algorithm alters the optimization landscape. The paper provides no mathematical proof or guarantee that this constrained EM algorithm converges to a stable local maximum or represents a valid probability density.

---

## 4. Presentation

**Rating: Excellent**

The paper is exceptionally well-written, clearly structured, and easy to follow. The presentation is supported by a comprehensive physical and logical data flow diagram (Figure 2), a detailed notational glossary (Table 1), and step-by-step algorithms and pseudocode in the appendix. The authors are transparent about their assumptions, sandbox limitations, and systems-level details.

---

## 5. Significance

**Rating: Good**

The work addresses an important and highly relevant problem: multi-task serving on low-power edge systems. The proposed framework has high practical utility and could influence future TinyML research, particularly as a zero-overhead, plug-and-play addition on top of pre-trained LoRA adapters. However, its significance to core machine learning theory is limited by its heuristic nature and lack of formal convergence or optimality proofs.

---

## 6. Originality

**Rating: Good**

The core building blocks—Zero-Shot Centroid Alignment (ZCA), diagonal Gaussian Mixture Models, and hard-thresholding operators—are well-known in the literature. However, the unique combination of these primitives with a dynamic hardware-aware budget control loop is novel and represents a valuable contribution to systems-ML. The Hierarchical HMD-GMM architecture is also a highly original engineering solution to address covariance singularities and coordinate manifold overlaps in larger expert registries.

---

## 7. Overall Recommendation

**Recommendation: 3: Weak reject**

**Justification:** 
This paper possesses clear practical merits and addresses a highly significant engineering bottleneck in TinyML edge serving. However, the theoretical weaknesses and mathematical shortcuts overall outweigh these merits in its current form. 

Specifically:
1. The mathematical derivations in Appendix A.1 rely on highly simplifying, unrealistic assumptions (e.g., Gaussian white noise representations, mutal independence between routing weights and layer-wise activations) that compromise the validity of the "activation dilution" proof.
2. The core evaluation is conducted on a synthetic "14-layer Analytical Coordinate Sandbox" where task domains are artificially projected onto orthogonal subspaces. This simplified sandbox completely lacks the complex, non-linear activation dynamics and manifold overlaps of real deep neural networks, heavily inflating the early router's performance.
3. The Coordinate GMM safety shield is fitted over a bounded coordinate space, violating the mathematical support assumptions of multivariate Gaussian distributions.

To be suitable for publication, the authors must either:
* Provide a mathematically rigorous formulation of "activation dilution" that accounts for statistical dependencies and non-linear representations across deep layers.
* Replace the synthetic sandbox evaluations with an exhaustive empirical evaluation of the routing, pruning, and GMM safety shield on the actual, non-linear activation manifolds of a real-world deep neural network (e.g., Vision Transformers or ResNets running on multi-task benchmarks).
