# 4. Experiment and Evaluation Check

This document evaluates the paper's experimental setup, baseline selection, hyperparameter choices, and the validity of the reported metrics.

---

## 1. Structural Bias Against Baselines in Batched Environments
While the paper evaluates against a respectable array of baselines (Static Uniform, Unregularized/Regularized Global Linear, QWS-Merge, BSigmoid-Router, L3-Router), the comparison under batch task heterogeneity (Sweep 3) is fundamentally undermined by an evaluation discrepancy:
- The baseline routers (like `BSigmoidRouter`) are restricted to outputting a single, batch-averaged routing coefficient of shape `[K]`, which is the physically correct way to execute batched inference in weight-space fusions on a GPU.
- Our proposed `CAMRouter` is evaluated with `return_sample_alphas=True`, returning sample-specific coefficients of shape `[B, K]`. This completely bypasses the physical batch-merging constraint.
- Comparing a baseline constrained to batch-level average coefficients with a method that gets to bypass the constraint via sample-specific coefficients is a structural, unfair bias.
- Furthermore, the authors do not evaluate their proposed "Decoupled Historical Gating (DHG)" in Sweep 3, leaving its claimed benefits completely unproven under realistic batch-merging constraints.

---

## 2. Artificial "Sandbox" Limits Generalizability
The entire experimental section is conducted within a synthetic, 300-line Python simulator (`run_experiments_new.py`) rather than on actual model weights of pre-trained Vision Transformers. 
- In the simulator, the relationship between weight routing coefficients ($\alpha_k$) and task performance is represented by a simple, hand-crafted algebraic formula:
  $$\text{prob\_correct} = 0.1317 + (\text{ceiling} - 0.1317) \times \text{norm\_score}$$
- This closed-form proxy cannot capture the complex, highly non-linear, and chaotic parameter-space dynamics of real deep neural networks.
- In a real weight-space fusion, minor deviations in routing coefficients can lead to catastrophic representation conflicts and sudden, non-linear performance collapse. By simplifying this relationship into a smooth, monotonic sigmoid-based function, the simulator trivializes the optimization problem and makes the findings highly non-generalizable to real-world model merging tasks.

---

## 3. High Statistical Noise and Over-Interpretation of Results
The simulator evaluates models using a test set of only **100 samples** per task.
- Results are averaged over only **3 seeds** (as seen in `run_experiments_new.py`: `seeds = [42, 43, 44]`).
- Evaluating a model on 100 samples means each correct sample represents a full $1.0\%$ change in accuracy. A binomial distribution with $p = 0.5307$ and $N=100$ has a standard deviation of $\approx 5.0\%$.
- Averaging over only 3 seeds means the standard error is extremely high ($\approx 2.9\%$).
- The authors make detailed claims and sweeps based on small fluctuations of $2\%$ to $4\%$ in accuracy (for instance, Sweep 1 states $h=1$ achieves 56.73% and $h=4$ achieves 53.07%, and Sweep 5 shows weight decay variations between 47.40% and 53.07%). Given the standard error of the evaluation, these fluctuations are well within standard statistical noise and do not represent robust architectural or optimization discoveries.

---

## 4. Discrepancy in Hyperparameter Reporting
There is a direct contradiction in the default hyperparameters reported in the paper:
- In Table 1, CAM-Router's Joint Mean Accuracy is reported as **53.07%**.
- According to Sweep 5, the accuracy of **53.07%** corresponds to a weight decay penalty of $\lambda_{wd} = 0.0$ or $\lambda_{wd} = 10^{-4}$.
- However, Sweep 5 marks $\lambda_{wd} = 10^{-3}$ as the **Default** (which achieves **47.40%**).
- Similarly, Sweep 1 shows that $h=1$ attention heads achieves **56.73%**, outperforming the default $h=4$ configuration at **53.07%**.
- There is no justification provided for why the authors chose to report a sub-optimal configuration as the main baseline in Table 1, or why they claim a weight decay of $10^{-3}$ is the "default" when the main results in Table 1 utilize a weight decay of $0.0$ or $10^{-4}$.
