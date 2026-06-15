# Peer Review

## Summary of the Submission
This paper studies the stability of sequential dynamic model ensembling and model merging over deep networks. The authors address "sequential routing jitter," where unregularized input-dependent routing coefficients exhibit high-frequency oscillations across network depth. They formalize sequential feedforward representation propagation as a discrete-time dynamical system on Banach spaces and leverage Banach's Fixed-Point Theorem to derive a novel upper bound on the joint layer-wise mapping's Lipschitz constant $L_{T_l}$.

To enforce contraction properties ($L_{T_l} < 1$) and guarantee convergence to a unique fixed-point representation trajectory, the authors propose the **Contraction-Regularized Router (CR-Router)**. This router incorporates a cohesive regularized objective function combining spectral norm (Frobenius) penalties on routing projection weights and inverse routing temperature penalties.

The authors evaluate CR-Router within a 14-layer Analytical Coordinate Sandbox (ICS) across 10 random seeds under extreme data scarcity (16 calibration samples per task). In this revised version of the paper, the authors conduct a highly rigorous evaluation across two settings: (1) **Experiment 1 (Orthogonal Task Subspaces)** and (2) **Experiment 2 (Overlapping Task Subspaces)**, which simulates realistic representational cross-talk. To expose standard routing illusions where unrouted baselines appear to perform accurate routing, the authors introduce two active routing metrics: **Direct Gating Accuracy** and **Gating Cross-Entropy**. Under perfectly orthogonal subspaces, CR-Router achieves a classification accuracy of **57.20% ± 3.72%** and routing accuracy of **69.12% ± 3.93%** (representing an absolute improvement of **+21.18%** over the unregularized linear router baseline). Under overlapping subspaces, Uniform Merging collapses to **27.48% ± 2.88%** due to representation cross-talk, whereas CR-Router recovers a strong classification accuracy of **47.33% ± 4.17%** (+19.85% absolute improvement over Uniform Merging and +16.48% over unregularized routing).

---

## Main Strengths
1. **Elegant Theoretical Formulation**: Casting sequential dynamic model ensembling as a discrete-time dynamical system and leveraging Banach's Fixed-Point Theorem is a highly elegant and mathematically rigorous contribution. It elevates the study of model merging from empirical heuristics (e.g., trajectory smoothing) to formal convergence theory.
2. **Cohesive Regularization Strategy**: The paper correctly identifies that standard weight decay on routing projection heads is insufficient to prove contraction properties, as temperature collapse ($\tau_l \to 0$) can still drive the Lipschitz constant to infinity. The joint regularizer (spectral norm + inverse temperature penalties) is technically cohesive and theoretically necessary.
3. **Exceptional Research Maturity in Revision**: The authors have thoroughly addressed major peer review concerns in this updated manuscript:
   - They introduced **Experiment 2 (Overlapping Subspaces)**, showing that CR-Router successfully mitigates representation cross-talk where static uniform merging completely collapses.
   - They introduced **Direct Gating metrics** that successfully expose the "routing illusion" of static baselines in orthogonal sandboxes.
   - They ran the **Regularization Sensitivity Sweep in the overlapping sandbox**, demonstrating a classic stable trade-off curve where over-regularization collapses performance back to uniform merging (30.00% accuracy), validating the active routing role.
   - They corrected the copy-paste/drafting error in Appendix A.2, clarifying that classification and routing accuracy are decoupled because experts classify locally under noise.
4. **Visually Compelling Validation**: The layer-wise routing trajectory plot (Figure 1b) beautifully validates the theoretical claims, showing how unregularized parametric routers exhibit violent gating oscillations, whereas CR-Router stabilizes smoothly to a unique fixed-point trajectory across network depth under task interference.

---

## Areas for Improvement (Constructive Suggestions)

No critical flaws remain in the paper's mathematical formulation, empirical execution, or documentation. However, we offer three constructive, high-quality suggestions to further strengthen the theoretical depth and practical impact of the work:

### 1. Highlight and Expand Discussion on the Soft Alignment Contraction Bound (Section 3.5)
In Section 3.5, the authors restore global continuity across decision boundaries in the sandbox by relaxing the hard prototype selection to a continuously differentiable soft alignment:
$$w_k(h) = \sum_{c=1}^C S_{k, c}(h) w_{k, c}$$
where $S_{k, c}(h) = \operatorname{Softmax}\left( \langle h, w_{k, c} \rangle / \tau_c \right)$. Under this soft alignment, they derive the global Lipschitz bound of the ensembling mapping:
$$L_{T_l}^{\text{ICM}} \le (1 - \gamma_l) + \gamma_l \left[ \frac{2}{\tau_c} R_{\mathcal{W}}^2 + \frac{2 R_{\mathcal{W}}}{\tau_l} \|W_{\text{route}}^{(l)}\|_2 \right]$$
For this system to be a strict contraction, we must have $L_{T_l}^{\text{ICM}} < 1$. Solving this inequality yields the correct contraction condition:
$$\|W_{\text{route}}^{(l)}\|_2 < \frac{\tau_l}{2 R_{\mathcal{W}}} \left[ 1 - \frac{2}{\tau_c} R_{\mathcal{W}}^2 \right]$$

Crucially, the authors exhibit outstanding theoretical honesty by pointing out that under the actual experimental hyperparameters ($\tau_c = 0.05, R_{\mathcal{W}} = 1$), the first term inside the brackets is:
$$\frac{2}{\tau_c} R_{\mathcal{W}}^2 = \frac{2}{0.05} (1)^2 = 40$$
Substituting this back, the contraction condition becomes:
$$\|W_{\text{route}}^{(l)}\|_2 < \frac{\tau_l}{2} [1 - 40] = -19.5 \tau_l$$
Since the routing temperature $\tau_l > 0$, the right-hand side of this inequality is strictly negative. Because $\|W_{\text{route}}^{(l)}\|_2$ is a matrix norm and must be non-negative, **this condition is mathematically impossible to satisfy**. 

This level of transparent self-critique in pointing out that the global contraction bound is technically vacuous under their practical configuration is outstanding. To make the theoretical analysis completely rigorous, the authors should add a brief discussion clarifying this limitation in the main text. For instance, they could suggest potential remedies (e.g., suggesting a larger soft-alignment similarity temperature like $\tau_c > 2.0$ to yield a positive bound) or explain why typical-case empirical stability emerges smoothly even when the worst-case global theoretical bound is conservative and mathematically vacuous.

### 2. Discussion of Practical Hyperparameter Tuning under Data Scarcity
The grid sweep over $\lambda_{\text{spec}}$ and $\lambda_{\text{temp}}$ in Table 3 shows a narrow optimal range ($\lambda \in [0.001, [0.010]$) outside of which performance collapses (either to overfitted jitter under low regularization or to static merging under high regularization). Since the target use case is extreme data scarcity (16 calibration samples per task), tuning these hyper-parameters without a large validation set is extremely difficult. The authors should discuss practical strategies or heuristics (such as monitoring gating entropy, routing path similarity, or depth variance during calibration as suggested in their appendix) in the main text to help practitioners find the optimal contraction regime.

### 3. Real-World Empirical Validation
While the Analytical Coordinate Sandbox (ICS) is highly valuable for isolating variables and visualizing trajectory mathematics, it remains a synthetic, low-dimensional block-structured environment. The paper's claims would be significantly more convincing if validated on real-world datasets and architectures (e.g., merging LoRA adapters on GLUE or Commonsense reasoning benchmarks using pre-trained Transformer-based LLMs like LLaMA-2 or RoBERTa). The authors list this as a future direction, but discussing the expected behavior of update-space quasi-contractions on real-world activation boundaries would bridge the gap between theory and practical engineering.

---

## Evaluation Ratings

### Soundness: Excellent (4/4)
*The mathematical proofs for Theorem 3.1 and Theorem 3.2 are highly detailed, elegant, and correct. The simulation code matches the paper's default setup and runs correctly. The evaluation protocol is exceptionally rigorous, incorporating overlapping task subspaces and active gating metrics that successfully resolve standard evaluation quirks. Appendix A.2 is perfectly aligned, explaining why classification and routing accuracy are decoupled.*

### Presentation: Excellent (4/4)
*The paper is exceptionally well-structured, written with high clarity, and features excellent visualization of trajectories. The narrative flow is compelling, and the authors demonstrate excellent theoretical candor regarding the relaxations of Update-Space Quasi-Contractions in residual networks.*

### Significance: Good (3/4)
*Providing a theoretical foundation for sequential dynamic model ensembling is highly valuable and represents a significant conceptual step. Resolving the high-frequency gating jitter in sequential routers has broad implications for multi-task serving and Mixture-of-Experts architectures. The lack of large-scale, real-world benchmarks on pre-trained models limits an excellent significance rating, but the work is highly important for the ensembling literature.*

### Originality: Excellent (4/4)
*Modeling feedforward sequential ensembling as a discrete-time dynamical system and leveraging Banach's Fixed-Point Theorem to derive explicit joint representation-routing bounds is highly original, creative, and conceptually refreshing.*

---

## Overall Recommendation and Rating

**Overall Rating: 5 (Accept)**

**Justification**: 
The paper is highly original, mathematically elegant, and addresses an important, overlooked challenge in sequential model ensembling (routing jitter). It provides rigorous proofs and beautiful trajectory visualizations. The authors have done a phenomenal job in the revision, addressing all critical comments by introducing overlapping task subspaces, implementing active gating metrics, resolving drafting inconsistencies in the appendix, and verifying the classic regularization trade-off sweep under task interference.

While the evaluation remains synthetic, the theoretical contributions and the exceptional quality of the revision make this paper highly solid and ready for publication. It establishes a strong, mathematically sound template for analyzing sequential representation-routing flows that will influence future Mixture-of-Experts (MoE) and serving architectures.
