# Peer Review of Conference Submission

## Summary of the Paper
The submission studies sequential dynamic model ensembling and model merging over deep networks. The paper specifically addresses "sequential routing jitter"—where layer-wise dynamic gating coefficients oscillate violently across network depth, leading to representational instability, degraded joint classification accuracy, and high sensitivity to random seeds under extreme data scarcity (e.g., 16 calibration samples). To mitigate this, the paper frames deep ensembling feature propagation as a discrete-time dynamical system and uses Banach's Fixed-Point Theorem to derive parameter constraints that guarantee convergence to a unique, stable trajectory under depth. 

The authors propose the **Contraction-Regularized Router (CR-Router)**, which bounds the routing function's Lipschitz constant by regularizing both the spectral norm of the routing projection ($W_{\text{route}}$) and the routing Softmax inverse temperature ($1/\tau_l$) during calibration. They also introduce **Update-Space Quasi-Contraction** to adapt to residual connections, **Centroid-Based Routing Warm-Starting** to mitigate initialization sensitivity, and **Adaptive Test-Time Temperature Annealing** to sharp-route inference samples. The method is evaluated on a synthetic 14-layer sandbox and PCA-reduced computer vision datasets (MNIST, Fashion-MNIST, USPS) mapped to 192 dimensions.

---

## Strengths and Weaknesses

### Strengths
1. **Interesting Formulation:** Conceptualizing sequential model ensembling as a discrete-time dynamical system and attempting to control representation trajectory stability via Banach's Fixed-Point Theorem is an elegant and academically interesting angle.
2. **Joint Regularization Concept:** The idea of jointly regularizing the weight norms and the temperature parameters of the routing heads is a direct, logical translation of the derived Lipschitz bounds.
3. **Writing Quality:** The paper is well-written, mathematically dense, and organized logically. The theoretical proofs are well-structured in the appendix.

### Weaknesses
1. **Broken and Vacuous Global Contraction Bounds:** In Section 3.4, the derived condition for global contraction under Soft Coordinate Alignment is:
   $$\|W_{\text{route}}^{(l)}\|_2 < \frac{\tau_l}{2 R_{\mathcal{W}}} \left[ 1 - \frac{2}{\tau_c} \kappa R_{\mathcal{W}}^2 \right]$$
   Under the evaluated empirical sandbox hyperparameters ($\tau_c = 0.05$, $R_{\mathcal{W}} = 1$, and $\kappa = 1$), the bracketed term is $1 - 40 = -39$, which simplifies the contractive condition to $\|W_{\text{route}}^{(l)}\|_2 < -19.5 \tau_l$. Since the spectral norm is non-negative and $\tau_l > 0$, **this condition is mathematically impossible to satisfy**. The global Lipschitz bound in their actual sandbox ranges from 4.9 to 40.0, completely violating the contractive assumption and rendering the global theory vacuous for their own experiments.
2. **Incompatibility with Residual Architectures:** Modern deep networks rely on residual identity paths where $F_{\text{base}}^{(l)}(h) = h$, forcing the base Lipschitz constant to be $L_{\text{base}} = 1$. A strict contraction ($L_{T_l} < 1$) is thus mathematically impossible. The proposed "Update-Space Quasi-Contraction" ($L_{U_l} < \epsilon$) results in a full-layer Lipschitz constant of $1 + \epsilon \ge 1$. Bounding the update space does not yield a unique fixed-point, nor does it guarantee convergence. The main theoretical selling point of the paper is bypassed in practical neural networks.
3. **Fundamental Contradiction in Test-Time Temperature Annealing:** The authors use strict Lipschitz regularization during training to prevent jitter, but then apply post-hoc temperature annealing at test time (scaling down temperatures by a factor of 10 or 100 to make the Softmax near-hard argmax). Scaling down $\tau_l \to 0$ explodes the Softmax Lipschitz constant ($\mathcal{O}(1/\tau_l)$) to infinity. Therefore, **the stable contraction properties are completely discarded during inference**. The paper lacks any empirical gating trajectory plots or Gating Depth-Variance (GDV) measurements of the annealed test-time models to prove they are still stable.
4. **Toy-Scale Evaluation and Lack of Real-World Validation:** Despite claiming relevance to modern large-scale language model serving (such as routing LoRA adapters on frozen LLMs), the paper does **not** evaluate on a single real transformer, language model, or large-scale dataset. All experiments are conducted on a synthetic 14-layer sandbox or PCA-reduced toy computer vision embeddings (MNIST, Fashion-MNIST, USPS) projected to 192 dimensions. There is no proof that the proposed method functions or scales in realistic deep learning workloads.
5. **Massive Accuracy Drops Compared to Simple Baselines:** In Experiment 3 (Table 6), CR-Router ($\gamma_{\text{scale}} = 1.0$) achieves **53.70%** accuracy, representing a **16.90% absolute drop** compared to SABLE (70.60%), a non-parametric centroid-based baseline. Even with test-time annealing ($\gamma_{\text{scale}} = 0.10$, Table 8), CR-Router only reaches **62.45%**, remaining **8.15% lower** than SABLE. SABLE is simple, has zero training overhead, and requires no calibration optimization. A 17% absolute performance drop is unacceptable for real-world deployments, making the proposed router highly impractical.
6. **Ad-Hoc Centroid Initialization:** The necessity of "Centroid-Based Routing Warm-Starting" to prevent the optimization from getting trapped in suboptimal basins suggests that the regularized landscape is highly non-convex and sensitive. This undermines the practical benefit of the contraction mapping theory, which supposedly guarantees robust convergence.

---

## Soundness
**Rating: Poor**

**Justification:**
The technical soundness is severely compromised. First, the global contraction bound is mathematically impossible to satisfy under the actual experimental hyperparameters of the paper, meaning the global theory does not apply to their evaluated models. Second, the "Update-Space Quasi-Contraction" does not possess the convergence properties of a strict contraction, meaning the core claim of depth-wise convergence is lost in standard residual networks. Third, the use of test-time temperature annealing completely violates the Lipschitz bounds enforced during training, representing a fundamental logical contradiction that is never empirically analyzed or justified with trajectory plots. Finally, the evaluation relies entirely on toy-scale simulations and PCA projections rather than actual deep learning models.

---

## Presentation
**Rating: Fair**

**Justification:**
While the paper is well-structured and uses sophisticated terminology, it suffers from significant conceptual obfuscation. Important theoretical failures (such as the impossible global bound) are buried in brief, passive paragraphs instead of being openly discussed. Section 4.6 is framed as a "Case Study" of routing LoRA in Transformers, which strongly misleads the reader into thinking a transformer was evaluated, when it is in fact a purely theoretical section with no code or experimental data. The extensive formalism serves primarily to mask the extreme simplicity and toy-like nature of the actual experiments.

---

## Significance
**Rating: Poor**

**Justification:**
The practical significance of this work is extremely low. The proposed method is substantially outperformed by SABLE, a simple, non-parametric, zero-training centroid-distance baseline, by a massive margin of up to 16.9% absolute. A marginal speedup of 15ms is highly unlikely to justify a massive loss in task accuracy. Furthermore, because the theory is mathematically incompatible with standard residual connections and requires unverified test-time temperature sharpening, it has no proven value or utility for modern ML practitioners serving large models (like LLMs).

---

## Originality
**Rating: Fair**

**Justification:**
The application of Lipschitz bounds to sequential model ensembling is neat, but the underlying mathematical techniques are standard and highly predictable (e.g., spectral normalization of projection matrices and bounding Softmax Lipschitz constants). The paper also completely fails to contextualize this problem against the vast literature on Mixture-of-Experts (MoE) router stabilization (auxiliary load-balancing losses, Z-losses, etc.), which deals with the exact same sequential routing jitter. Treating this as a unique problem of "model merging" overstates the methodological novelty.

---

## Detailed Comments and Questions for the Authors

1. **Regarding the Global Contraction Bound:** In Section 3.4, your derived global bound requires $\|W_{\text{route}}^{(l)}\|_2 < -19.5 \tau_l$ under your empirical hyperparameters ($\tau_c = 0.05, R_{\mathcal{W}} = 1, \kappa = 1$). Since this is mathematically impossible, can you provide a valid set of hyperparameters under which the global contraction condition is actually satisfied? What is the classification accuracy of CR-Router when restricted to those mathematically valid hyperparameters?
2. **Regarding the Test-Time Temperature Annealing Contradiction:** When you scale down the temperature during inference ($\gamma_{\text{scale}} = 0.10$ or $0.01$), the Softmax becomes highly non-smooth, and the Lipschitz constant explodes to infinity. Can you provide Gating Depth-Variance (GDV) plots and Running Lipschitz Bounds (RLB) *at test time* for these annealed models? Do they still maintain the stable, jitter-free trajectories you claim to solve, or do they revert to the chaotic, oscillating trajectories of the unregularized linear router?
3. **Regarding the "Update-Space Quasi-Contraction":** Since $L_{\text{base}} = 1$ in modern residual networks, your "quasi-contraction" $L_{T_l} \le 1 + \epsilon$ does not guarantee a unique fixed point or depth-wise convergence. Given this, does the core mathematical argument of the paper (Banach's Fixed-Point Theorem) actually have any mathematical relevance to standard residual networks?
4. **Regarding the Baseline Accuracy Gap:** In Table 6, SABLE outperforms CR-Router by 16.9% absolute, and ChemMerge outperforms it by 15.2% absolute. Given such a massive drop in representation quality, why should a practitioner prefer CR-Router over SABLE or ChemMerge? Is a minor CPU latency speedup of 15ms worth a 17% absolute reduction in accuracy?
5. **Real-World Validation:** Why did you not evaluate your method on actual deep networks (e.g., a standard ResNet or Vision Transformer on CIFAR-100 or ImageNet) or actual language models (e.g., routing 4-8 specialized LoRA adapters on a frozen GPT-2 or LLaMA-3.2-1B on GLUE)? 

---

## Overall Recommendation

**Rating: 2 (Reject)**

**Justification:**
The submission presents an elegant dynamical perspective on sequential model ensembling, but it falls far short of the rigorous standards required for a top-tier machine learning conference. 

The paper's core theoretical guarantees are broken in practice: the global contraction bound is mathematically impossible to satisfy under the evaluated hyperparameters, and the "Update-Space Quasi-Contraction" lacks any of the unique fixed-point convergence properties that motivate the work. Furthermore, the introduction of test-time temperature annealing creates a fundamental logical contradiction that invalidates the Lipschitz bounds during inference. 

Empirically, the paper relies entirely on toy-scale synthetic coordinate simulations and PCA-reduced computer vision embeddings, completely failing to validate the method on actual deep architectures or language models despite heavily advertising such capabilities. Finally, the proposed method is heavily outperformed (by up to 17% absolute accuracy) by simple, non-parametric, zero-training centroid baselines (SABLE), which completely undermines its practical significance and utility. For these reasons, the paper in its current form must be rejected.
