# 4. Experimental Check and Empirical Scrutiny

As a reviewer focused on empirical rigor, I have carefully scrutinized the experimental setup, choice of baselines, datasets, statistical validity, and the consistency of the reported results. While the paper contains a comprehensive set of sensitivity analyses, several severe concerns must be addressed regarding empirical contradictions and statistical soundness.

---

## 1. Major Empirical Inconsistencies and Contradictions
There are multiple glaring discrepancies between the results reported in the main tables, the text of the paper, and the sensitivity/ablation sweeps in the appendix. These contradictions must be clarified:

### Discrepancy A: Orthogonal Heterogeneous Stream Results
* **Table 1 (Main Results):** Reports that **PAC-Kinetics (Ours)** on the Orthogonal Heterogeneous Stream achieves a Joint Accuracy of **$92.35\% \pm 0.48\%$**.
* **Section 4.5 (Main Text, page 13):** Explicitly states: *"In contrast, PAC-Kinetics achieves 86.53% \pm 4.10% and 82.21% \pm 1.92%, representing drops of 8.28% and 12.53% respectively."*
* **Table 3 (Prior Variance sweep):** Reports the default PAC-Kinetics ($\sigma_0^2 = 5.0$) on this exact same stream achieves **$86.53\% \pm 4.10\%$**.
* **Table 4 (Calibration Length sweep):** Reports the default PAC-Kinetics ($T=32$) on this exact same stream achieves **$86.53\% \pm 4.10\%$**.
* **Critique:** Why is there a $6\%$ discrepancy for the exact same default configuration of PAC-Kinetics on the orthogonal heterogeneous stream? If the default model achieves $92.35\%$, why do Table 3, Table 4, and the text in Section 4.5 report $86.53\%$? This suggests a severe mix-up of experimental runs or parameters.

### Discrepancy B: Overlapping Heterogeneous Stream Results
* **Table 2 (Main Results):** Reports that **PAC-Kinetics (Ours)** on the Overlapping Heterogeneous Stream achieves a Joint Accuracy of **$92.90\% \pm 0.71\%$**.
* **Section 4.5 (Main Text, page 13):** States that PAC-Kinetics achieves **$82.21\% \pm 1.92\%$** (which represents a $12.53\%$ drop relative to PAC-ZCA's $94.74\%$).
* **Table 9 (Non-Negative Ablation, page 18):** Reports that the **Unconstrained (Default)** model (which is PAC-Kinetics) on the Overlapping Heterogeneous Stream achieves a Representation Alignment Accuracy of **$67.69\% \pm 11.25\%$**.
* **Critique:** Why are there three completely different performance figures ($92.90\%$, $82.21\%$, and $67.69\%$) reported for the default configuration on the overlapping heterogeneous stream? If representation alignment is a different metric, why is the metric in Table 2 labeled "Accuracy (\%)" while Section 4.1 states that "Accuracy" in the sandbox is defined exactly as "Representation Alignment Accuracy"?

These discrepancies raise serious concerns about the reliability, consistency, and controlled nature of the empirical evaluations.

---

## 2. Statistical Significance Concerns
While the authors report standard deviations across 5 random seeds, the overlapping standard deviations under more realistic conditions suggest that some reported benefits are not statistically significant:
* **Physical Homogeneous Stream (Table 7):** PAC-Kinetics achieves a classification accuracy of $76.40\% \pm 5.50\%$, and stateless PAC-ZCA achieves $71.20\% \pm 4.02\%$. Given the standard deviations, their performance intervals heavily overlap ($70.9\% - 81.9\%$ for PAC-Kinetics vs. $67.18\% - 75.22\%$ for PAC-ZCA). Without a formal t-test or Wilcoxon signed-rank test, it is impossible to verify if the $5.20\%$ improvement is statistically meaningful or simply a byproduct of seed-to-seed variance on a small test sequence of 200 samples.
* **Physical Heterogeneous Stream (Table 7):** On the heterogeneous stream, stateless PAC-ZCA ($71.20\% \pm 4.02\%$) outperforms PAC-Kinetics ($66.30\% \pm 7.79\%$). This proves that even with the "Adaptive Online Kinetics" mechanism, the stateful model's routing lag (inertial drag) remains a major liability on physical datasets, failing to match the performance of simple stateless routers under rapid task switches.

---

## 3. Baselines Evaluation
* The inclusion of standard sequence models like GRUs and LSTMs in Appendix 7.1.5 is highly appreciated and strengthens the paper's comparison against alternative sequential learning architectures.
* However, there is no comparison against a simple, tuned **Exponential Moving Average (EMA)** filter baseline in the main results. In Section 3.3, the authors state that under a diagonal retention matrix $A = a \cdot I$ and injection matrix $W = (1-a) \cdot I$, their linear recurrence collapses to a standard multi-dimensional EMA. To justify the complexity of learning task-specific $a_k$ and a full cross-task coupling matrix $W \in \mathbb{R}^{K \times K}$ via PAC-Bayesian optimization, the authors *must* provide an empirical comparison against a standard, grid-searched static EMA filter to show the exact "delta" and benefit of their learned kinetics.

---

## 4. The "Deterministic-Randomized" Generalization Gap
* **The Critique:** Theorem 3.1 mathematically guarantees generalization for the **randomized** Gibbs posterior $Q = \mathcal{N}(\Theta_{\text{opt}}, \sigma_0^2 I)$. Yet, Table 1 and Table 2 reveal that evaluating this randomized router directly (**PAC-Kinetics (Rand)**) results in a complete accuracy collapse down to near-uniform levels ($\approx 31\% - 33\%$).
* This means that the model for which generalization is mathematically guaranteed is practically unusable in production. The actual model served is the **deterministic surrogate** (the mean parameters $\Theta_{\text{opt}}$), which lacks direct PAC-Bayesian generalization bounds in this work. While the authors try to bridge this gap in Section 7.2 using trajectory discrepancy bounds, the severe performance collapse of the randomized router highlights a fundamental theoretical-to-empirical mismatch. It indicates that the optimized posterior distribution is dominated by unstable parameter regions, and the served model relies entirely on the deterministic surrogate approximation, which is a significant limitation of applying PAC-Bayes to stateful dynamical systems.
