# Peer Review: Unitary Geodesic Routing (UGR)

## 1. Paper Summary
The paper presents **Unitary Geodesic Routing (UGR)**, a novel geometric framework for dynamic test-time model ensembling across heterogeneous, non-stationary task streams. While prior stateful ensembling routers (e.g., Momentum-Merge, ChemMerge) perform linear blending in unconstrained flat Euclidean spaces ($\mathbb{R}^K$) and project back to the probability simplex post-hoc via Softmax normalization, UGR models ensembling states directly on the curved $(K-1)$-dimensional unit hypersphere $\mathbb{S}^{K-1} \subset \mathbb{R}^K$.

UGR leverages the square-root map (Born's rule mapping) from classical Information Geometry to project coordinates to the probability simplex natively in a Softmax-free, scale-preserving manner. It models temporal updates as closed-form geodesic rotations (Spherical Linear Interpolation / Slerp) along the shortest great-circle path of the hypersphere, completely bypassing virtual-time numerical ODE solvers. To balance stability and plasticity, UGR utilizes **Torque-Driven Adaptive Agility** (scaling the step size dynamically based on angular representational torque) and **Spatial-Temporal Geodesic Coupling** (recurrently propagating states across consecutive queries).

The authors evaluate UGR in a 14-layer synthetic Analytical Coordinate Sandbox (ICS) and a real-world multi-task text classification Mixture-of-Experts pipeline on the `20newsgroups` dataset. The empirical results demonstrate that UGR achieves state-of-the-art joint classification accuracy while slashing layer-to-layer routing jitter (oscillations) and running training-free with negligible serving latency.

---

## 2. Overall Recommendation
**Rating: 5 (Accept)**

**Justification:**  
This is an exceptionally solid, mathematically elegant, and phenomenally well-evaluated paper. The authors present a highly original paradigm shift that rejects the unconstrained flat-space assumptions of previous temporal routers in favor of curved manifolds. From an empirical standpoint, the scientific hygiene is outstanding: evaluations are run across 10 independent random seeds for the synthetic sandbox and 5 seeds for the real-world text classification, with complete means and standard deviations reported for all major tables. Furthermore, the authors conducted a rigorous code-level audit of prior baselines to identify and correct a critical initialization bug, ensuring an absolutely fair, hygienic, and synchronized comparison. The depth of the ablation suite—evaluating mathematically exact Born target mappings, Softmax-free ReLU target alternatives, and hybrid/continuous reset strategies—is exemplary. While the scale of the real-world NLP evaluation is currently limited to TF-IDF features and MLP experts, the thoroughness of the empirical validation, combined with the detailed mathematical derivations and PyTorch backpropagation gradients in the Appendix, makes this work highly deserving of publication at a top-tier machine learning venue.

---

## 3. Rating Table

| Dimension | Rating | Justification |
| :--- | :--- | :--- |
| **Soundness** | **Excellent** | The mathematical foundations are robust and isometric to Fisher-Rao geodesic flows on the probability simplex. The first-order torque adaptation represents a control-theoretically sound dynamical system that natively avoids overshoot or oscillation. Edge cases like collinearity and high-dimensional measure concentration are thoroughly resolved via local sub-manifold projection. |
| **Presentation** | **Excellent** | The paper is written with outstanding clarity and precision. Pseudocode is clean and self-contained, and the figures (especially the wiggling off the simplex comparison and the Accuracy-Stability Pareto frontier) are highly polished, professional, and informative. |
| **Significance** | **Good** | Dynamic test-time model ensembling is a highly relevant, high-impact frontier given the dominance of PEFT/LoRA and Mixture-of-Experts. The geometric connection opens up rich new research directions. However, the current real-world evaluation is limited to a smaller-scale TF-IDF MLP setup, slightly dampening immediate industry significance. |
| **Originality** | **Excellent** | The creative combination of Information Geometry's square-root homeomorphism (Born's rule) with Rodrigues-like geodesic Slerp updates to construct a stateful test-time router is highly original and represent a substantial shift from unconstrained Euclidean blending. |

---

## 4. Strengths

1. **Exemplary Empirical Rigor and Statistical Confidence:**  
   Unlike many machine learning papers that rely on single-run evaluations, this paper evaluates all methods across **10 independent random seeds** on the synthetic sandbox and **5 independent seeds** on the real text classification dataset. Reporting precise means and standard deviations for all metrics (joint accuracy, overall jitter, and intra-query jitter) gives immense confidence in the statistical significance of the results.
2. **Scientific Hygiene in Baselining:**  
   The authors are commended for auditing prior baselines at the code level. Finding and correcting the initialization bug in the prior *Momentum-Merge (Advanced)* baseline—where the boundary prior was being overwritten with the target vector, artificially suppressing its jitter—shows commendable scientific integrity. Enforcing a strict, synchronized, and uniform initialization for all uncoupled models ensures an absolutely fair and transparent comparison.
3. **Rigorous and Deep Suite of Ablations:**  
   The ablation studies are exceptionally thorough, evaluating multiple critical dimensions:
   * **UGR (Born Target):** Evaluates the exact square-root target mapping to eliminate quadratic sharpening distortion, proving it acts as a trajectory-smoothness maximizer (reducing NLP routing jitter by 2.3$\times$ to $1.60 \times 10^{-4}$).
   * **UGR (Softmax-Free Target):** Replaces target Softmax with ReLU and $L_1$-normalization, achieving a pristine $1.50 \times 10^{-4}$ routing jitter (a 4.0$\times$ reduction over Coupled Momentum-Merge).
   * **UGR (Hybrid & Continuous Reset):** Mitigates the single-layer boundary transition shock caused by spatial-temporal coupling under sudden switches, reducing boundary shock by over 2.5$\times$.
   * **Centroid Sample Efficiency:** Sweeps calibration samples from 4 to 128, proving high sample efficiency.
   * **Differentiable Backpropagation Validation:** PyTorch-based training prototype proving numerical stability.
4. **First-Order Control-Theoretic Advantage:**  
   Modeling Torque-Driven Agility as a first-order non-linear dynamical system with non-linear damping represents a major control-theoretic benefit over second-order inertial or momentum-based methods. Because velocity scales directly with angular mismatch and has no acceleration terms, UGR's trajectories are mathematically guaranteed to completely avoid overshoot, oscillation, or hysteresis.
5. **Practical Serving Efficiency:**  
   The CPU-based wall-clock timing benchmark is highly practical. Showing that UGR adds less than **0.07 ms** of latency per query over the stateless baseline and achieves **2052.7 QPS** (and **2295.3 QPS** for the Softmax-Free target variant) proves that closed-form geodesic updates successfully bypass the severe latency bottlenecks of virtual-time ODE solvers.

---

## 5. Weaknesses

1. **Scale Gap in Real-World Evaluation:**  
   The primary weakness of the paper lies in the scale of the real-world evaluation. While the statistical hygiene of the evaluation is outstanding, the experiments are conducted on the classic `20newsgroups` dataset using simple TF-IDF representations (max features $D=1024$) and 2-layer MLP classifiers. The authors present a highly promising "Real-World Serving Blueprint" (Appendix Section A.3) and a "Real-World Validation Roadmap" (Section 5) outlining how UGR can be applied to token-level LoRA ensembling in modern Large Language Models (e.g., LLaMA-3, Mistral), but they do not empirically execute experiments at this scale. Evaluating UGR on actual pre-trained transformer backbones (e.g., RoBERTa, LLaMA) with actual Parameter-Efficient Fine-Tuning (PEFT) expert adapters on standard multi-task benchmarks (MMLU, GLUE) would significantly strengthen the paper's significance.
2. **Hyperparameter Sensitivity to Workload Characteristics:**  
   The optimal geodesic step-size (inertia) parameter $\eta$ shifts heavily depending on the switching frequency of the stream ($\eta=0.80$ for the high-frequency synthetic sandbox, $\eta=0.10$ for the stable NLP block-structured stream). In real-world multi-tenant production serving, the task-switching frequency is rarely known a priori. If a mismatch occurs, UGR risks either carrying over stale temporal priors (if $\eta$ is too small) or introducing high-frequency noise jitter (if $\eta$ is too large), meaning practitioners must carefully hand-tune this hyperparameter.
3. **Centroid Quality and Static Assumption:**  
   The main text evaluations assume pre-computed, static centroids. While the authors derive and validate an online centroid update rule starting from random initialization in Appendix A.5, evaluating UGR's performance under active semantic drift on real text representations (where task boundaries are highly overlapping) would strengthen the empirical claims of robustness.

---

## 6. Actionable Feedback & Suggestions for Improvement

* **Scale up the Real-World Evaluation:**  
  We strongly recommend the authors scale up their real-world evaluation to modern deep neural networks. Even a moderate-scale experiment using pre-trained transformer representations (e.g., RoBERTa or BERT) with LoRA adapters on a multi-task sequential text sequence (e.g., GLUE subtasks or MMLU subjects) would completely bridge the scale gap and silence any criticisms regarding the simplicity of TF-IDF on the `20newsgroups` dataset.
* **Formulate a Workload-Adaptive Step-Size Mechanism:**  
  To address the hyperparameter sensitivity of $\eta$, the authors could explore a simple adaptive step-size mechanism. For instance, the step size could be dynamically scaled by a running average of the representational torque:
  \begin{equation}
  \bar{\phi}_t = (1 - \lambda)\bar{\phi}_{t-1} + \lambda \phi_t
  \end{equation}
  where a high running torque indicates a highly non-stationary environment (favoring a larger $\eta$), and a low running torque indicates a stable task sequence (favoring a smaller $\eta$ for maximum smoothing). This would make UGR a robust, "plug-and-play" serving router.
* **Empirical Validation of Online Drift on Real Text:**  
  While the random-initialization latent expert discovery simulation in Appendix A.5 is mathematically compelling, running a small-scale real-world drift experiment (e.g., where domain distributions gradually shift over the 800-query text stream) and showing that UGR with online centroid updates maintains high accuracy would dramatically elevate the empirical depth of the work.
