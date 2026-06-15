# Peer Review of Conference Submission: PAC-Kinetics

## 1. Summary of the Paper
The paper addresses the challenge of **test-time dynamic model ensembling/routing** under sequential, heterogeneous query streams. To deploy multiple task-specific adapters (such as LoRA) under a single shared backbone, existing dynamic routers either use "stateless" designs that suffer from high-frequency routing oscillations due to query noise ("routing jitter paradox"), or heuristic "stateful" designs (such as ChemMerge) that reduce jitter but lack mathematical stability or generalization guarantees, leading to severe accuracy collapse under rapid task switches.

To resolve these limitations, the paper proposes **PAC-Kinetics**, a stateful ensembling framework that:
1. Formulates representation dynamics as concentrations in a continuous-time chemical kinetics reactor, resulting in a stable discrete-time linear recurrence.
2. Proves global asymptotic stability (GAS) and input-to-state stability (ISS) under Lyapunov control theory.
3. Derives a Catoni-type PAC-Bayesian generalization bound for stationary $\beta$-mixing stochastic processes, resolved via the **Even/Odd Block Splitting** technique to prevent the TV penalty explosion.
4. Integrates **Adaptive Online Kinetics** to dynamically scale down state-retention during rapid task transitions, preserving responsiveness.
5. Formally analyzes the "deterministic-randomized" serving gap via trajectory sensitivity bounds.

---

## 2. Main Strengths
* **High Mathematical Rigor:** The paper successfully integrates control theory (Lyapunov stability, contractive operators), biochemistry-inspired kinetics, and PAC-Bayesian theory to establish a solid mathematical foundation for dynamic ensembling.
* **Rigorous Handling of Mixing Processes:** Deriving a Catoni-type PAC-Bayesian generalization bound under stationary $\beta$-mixing processes is highly non-trivial. The use of Even/Odd Block Splitting to avoid the exploding TV penalty of dependent-data concentration inequalities is a major theoretical contribution.
* **Adaptive Online Kinetics:** The differentiable similarity-based self-modulating retention mechanism is simple, elegant, and successfully balances representation smoothing with transition responsiveness.
* **Trajectory Discrepancy Bounds:** The formal sensitivity analysis (Section 7.2) showing that trajectory discrepancy scales quadratically as $(1-\rho)^{-2}$ under parameter perturbations provides a thorough and necessary explanation for the deterministic-randomized serving gap.
* **Systems Efficiency:** The state-tracking update is extremely lightweight, requiring $<0.31$ KB of parameter memory and running in flatly $\approx 10.4$ microseconds on CPU and $<3.5$ microseconds on GPU, making it highly viable for production frameworks like S-LoRA or Punica.

---

## 3. Main Weaknesses (Empirical Scrutiny)

### A. Glaring Empirical Inconsistencies and Contradictions
There are multiple severe discrepancies and contradictions across the tables and text of the paper. These inconsistencies must be resolved to ensure the reliability of the empirical findings:
1. **Orthogonal Heterogeneous Stream Discrepancy:** In Table 1 (Main Results), **PAC-Kinetics (Ours)** on the Orthogonal Heterogeneous Stream is reported to achieve a Joint Accuracy of **$92.35\% \pm 0.48\%$**. However, in Section 4.5 (page 13), the text explicitly states: *"In contrast, PAC-Kinetics achieves 86.53% \pm 4.10% and 82.21% \pm 1.92%, representing drops of 8.28% and 12.53% respectively."* Furthermore, Table 3 (Prior Variance sweep) and Table 4 (Calibration Length sweep) both report the default PAC-Kinetics accuracy for this exact configuration as **$86.53\% \pm 4.10\%$**. Why is there a $6\%$ absolute discrepancy between the main tables and the sweeps/discussion?
2. **Overlapping Heterogeneous Stream Discrepancy:** Table 2 reports that **PAC-Kinetics (Ours)** on the Overlapping Heterogeneous Stream achieves **$92.90\% \pm 0.71\%$** Joint Accuracy. But Section 4.5 states that it achieves **$82.21\% \pm 1.92\%$** (representing a $12.53\%$ drop relative to PAC-ZCA's $94.74\%$). Moreover, Table 9 (Ablation Study) reports the default "Unconstrained (Default)" model on this exact same stream achieves **$67.69\% \pm 11.25\%$** Representation Alignment Accuracy. Why are there three completely different performance figures ($92.90\%$, $82.21\%$, and $67.69\%$) reported for what appears to be the same default model?

### B. "Simulation-to-Physical" Evaluation Gap
* The primary evaluation is conducted in a closed-form "Coordinates Sandbox" (Analytical Coordinate Sandbox / ICS) where activation trajectories are simulated inside a vector space. 
* The physical evaluation (Section 7.1.6) is limited to MNIST and Fashion-MNIST using a shallow, 3-layer MLP. 
* This is a massive gap relative to the paper's systems-level motivations (serving multi-tenant Large Language Models and deep Vision Transformers under high-throughput GPU frameworks like S-LoRA and Punica). Shallow 3-layer MLPs do not suffer from the "cascading representation collapse" that deep networks possess. Therefore, the practical effectiveness of PAC-Kinetics in actual, deep cascading production backbones remains unproven.

### C. Overlapping Performance and Lack of Significance Tests
* In the physical validation (Table 7), PAC-Kinetics achieves $76.40\% \pm 5.50\%$ classification accuracy under homogeneous streams, while stateless PAC-ZCA achieves $71.20\% \pm 4.02\%$. The performance intervals heavily overlap ($70.90\% - 81.90\%$ vs. $67.18\% - 75.22\%$). Without a formal statistical significance test (such as a paired t-test), it is impossible to verify if the $5.20\%$ improvement is statistically meaningful.
* Furthermore, on physical heterogeneous streams, stateless PAC-ZCA ($71.20\% \pm 4.02\%$) outperforms PAC-Kinetics ($66.30\% \pm 7.79\%$), showing that the proposed Adaptive Online Kinetics mechanism does not fully overcome the routing lag liability on physical datasets.

### D. Missing Static EMA Baseline
* In Section 3.3, the authors mention that under a diagonal retention matrix and specific injection matrix, their linear recurrence is mathematically identical to a standard multi-dimensional Exponential Moving Average (EMA) filter.
* To justify the complexity of learning task-specific retention rates $a_k$ and a full cross-task coupling matrix $W \in \mathbb{R}^{K \times K}$ via PAC-Bayesian optimization, the authors should provide an empirical baseline of a simple, tuned static EMA filter in the main experiments.

### E. Theoretical-to-Empirical Mismatch (Deterministic-Randomized Gap)
* Theorem 3.1 mathematically guarantees generalization strictly for the **randomized** Gibbs posterior $Q$. Yet, Table 1 and Table 2 reveal that evaluating this randomized router directly (**PAC-Kinetics (Rand)**) results in a complete accuracy collapse down to near-uniform levels ($\approx 31\% - 33\%$).
* This indicates that the model with the mathematical guarantee is practically unusable. The served model relies entirely on the **deterministic surrogate** approximation (the mean parameters $\Theta_{\text{opt}}$), which lacks direct PAC-Bayesian generalization bounds in this work. This is a significant limitation of applying PAC-Bayesian theory to stateful dynamical systems.

---

## 4. Questions and Actionable Suggestions for the Authors
1. **Clarify Empirical Discrepancies:** Please carefully explain the contradictions in the reported heterogeneous stream accuracies (e.g., $92.35\%$ vs. $86.53\%$ on orthogonal, and $92.90\%$ vs. $82.21\%$ vs. $67.69\%$ on overlapping). Which configurations and parameter scales were actually used for each?
2. **Run Significance Tests:** Please provide paired t-tests or Wilcoxon signed-rank tests for the physical evaluation results in Table 7 to confirm that the homogeneous accuracy improvement over PAC-ZCA is statistically significant.
3. **Add static EMA Baseline:** Please include a static, grid-searched multi-dimensional EMA filter baseline in Tables 1, 2, and 7 to demonstrate the exact empirical value-add of your learned coupling matrix $W$.
4. **Evaluate on a Real Deep Backbone:** If possible, please provide a small-scale evaluation on a real, multi-layer deep network (e.g., a 6-layer ViT or a small BERT-style transformer with 2-3 LoRA adapters) to demonstrate that the method actually prevents cascading representation collapse in deep cascading networks.

---

## 5. Ratings and Overall Recommendation

### Ratings
* **Soundness:** **Fair** (The mathematical proofs and control-theoretic bounds are excellent and technically sound. However, the severe discrepancies in the reported empirical results and the lack of statistical significance tests under physical validation degrade the overall soundness rating.)
* **Presentation:** **Excellent** (The paper is beautifully written, logically structured, and provides highly intuitive conceptual figures.)
* **Significance:** **Good** (If the empirical inconsistencies are resolved and the simulation-to-physical gap is bridged, this work provides a highly significant and novel contribution to dynamic PEFT serving on the edge.)
* **Originality:** **Excellent** (The integration of control-theoretic stability, chemical kinetics, and PAC-Bayesian mixing process bounds is highly original and creative.)

### Overall Recommendation
* **Recommendation Score:** **3: Weak Reject**
* **Justification:** The paper possesses exceptional theoretical merits, providing elegant stability proofs, a novel mixing-process PAC-Bayesian bound, and detailed trajectory sensitivity analyses. However, from an empirical perspective, the paper has clear weaknesses that outweigh these merits in its current form:
  1. Multiple severe discrepancies and contradictions in the reported accuracy figures across tables and text.
  2. A massive simulation-to-physical evaluation gap (relying heavily on simulated sandboxes and toy 3-layer MLPs on MNIST/Fashion-MNIST while motivating large-scale LLMs and deep ViTs).
  3. Overlapping standard deviations and a lack of statistical significance tests on physical datasets, where stateless baselines actually outperform the proposed method on heterogeneous streams.
  4. Complete performance collapse of the randomized router that holds the theoretical generalization guarantee.
  
  The paper requires a thorough empirical audit, the addition of a static EMA baseline, significance testing, and a small-scale evaluation on a real deep transformer architecture before it can be accepted. I strongly encourage the authors to resolve these empirical issues, as the core theoretical contribution is exceptionally strong.
