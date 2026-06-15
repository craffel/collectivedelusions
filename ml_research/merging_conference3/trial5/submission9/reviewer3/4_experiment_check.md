# Experimental Evaluation Check

## Baselines
The paper compares GSC-Merge against a comprehensive set of baselines:
- **Uniform Merging** (no tuning)
- **Task Arithmetic (TA)** (globally swept scaling factor)
- **Sparse Task Arithmetic (STA)** (grid-swept magnitude pruning threshold)
- **TIES-Merging** (grid-swept pruning threshold + sign election)
- **OFS-Tune (Unconstrained)** (direct coefficient optimization)

The baselines are well-chosen and represent the standard state-of-the-art in weight-space merging. The authors perform grid sweeps over the hyperparameter spaces of STA and TIES-Merging to avoid under-tuning bias, which represents a fair and rigorous comparison.

However, a notable concern is the extremely poor performance of STA (11.99%) and TIES-Merging (12.91%) under the task-conditional swapping setup, which is barely above random guessing (10.0%). These coordinate-wise pruning methods typically perform much better in the literature on more homogeneous task suites (e.g., multiple natural language processing tasks or similar visual domains). Applying them to highly conflicting datasets (such as MNIST mixed with CIFAR-10) seems to cause complete representation collapse, indicating that coordinate-wise pruning without adaptive layer-wise scaling is highly fragile in heterogeneous settings.

## Statistical Soundness of Evaluation
The authors utilize **5 independent random validation calibration splits** to calculate both the mean and standard deviation of accuracy across all methods. This is an excellent practice for establishing statistical confidence in low-data regimes (16 samples per task).

However, a close examination of the reported standard deviations (SD) reveals that the paper's primary empirical claim is **not fully supported by the reported data**:
- **Claim:** GSC-Merge "dramatically reduces split-sensitivity variance across validation splits... serving as a robust spectral regularizer."
- **Contradiction in Reported Results (Table 1):**
  - For **SVHN** under Task-Conditional Swapping: Unconstrained OFS-Tune exhibits an SD of **$\pm 19.70\%$**. GSC-Merge with $\gamma=0.5$ has an SD of **$\pm 19.60\%$**, and with $\gamma=0.3$ has an SD of **$\pm 19.09\%$**. These variances are statistically indistinguishable, showing no "dramatic reduction" in split-sensitivity.
  - For **MNIST** under Task-Conditional Swapping: Unconstrained OFS-Tune exhibits an SD of **$\pm 12.54\%$**. GSC-Merge with $\gamma=0.5$ actually exhibits a **higher** SD of **$\pm 12.89\%$**, which directly contradicts the regularizing claim. GSC-Merge with $\gamma=0.3$ reduces the SD to $\pm 8.92\%$, but at a severe cost to accuracy (dropping from $54.37\%$ to $48.70\%$, a $5.67\%$ reduction).
  - For **CIFAR-10** under Task-Conditional Swapping: Unconstrained OFS-Tune SD is **$\pm 8.87\%$**. GSC-Merge with $\gamma=0.5$ has an SD of **$\pm 8.93\%$** (higher than unconstrained), and with $\gamma=0.3$ has an SD of **$\pm 8.81\%$** (virtually identical).
  - Under **Truly Task-Agnostic Settings (Table 2)**:
    - Unconstrained joint mean is $20.86 \pm 4.81\%$.
    - GSC-Merge ($\gamma=0.5$) joint mean is $20.61 \pm 4.80\%$.
    - GSC-Merge ($\gamma=0.3$) joint mean is $19.08 \pm 4.85\%$.
    - Here, the standard deviations are again nearly identical, and for $\gamma=0.3$, the variance is actually *higher* than unconstrained while the mean performance is lower.

**Empirical Conclusion:** The claim of spectral regularization dramatically reducing split-sensitivity variance is unsupported. The variances are statistically equivalent across GSC-Merge and unconstrained tuning in almost all configurations. 

## Inconsistencies and Typographical Errors
There is a direct contradiction between the text in Section 4.4 and the results presented in Table 2:
- **Text (Section 4.4, "Addressing the Remaining Performance Gap"):** *"Under the truly task-agnostic setting, this gap becomes even wider (17.19% vs 74.96%)."*
- **Table 2 (Truly Task-Agnostic Table):** The joint mean accuracies for GSC-Merge are:
  - $\gamma = 0.1$: $16.77 \pm 3.25\%$
  - $\gamma = 0.2$: $18.91 \pm 4.29\%$
  - $\gamma = 0.3$: $19.08 \pm 4.85\%$
  - $\gamma = 0.5$: $20.61 \pm 4.80\%$
- **Discrepancy:** The figure 17.19% does not appear in Table 2 for any of the GSC-Merge ranks, indicating an inconsistency (likely a leftover from an earlier draft or specific intermediate seed).

## General Performance Concerns
A massive performance gap remains between GSC-Merge (42.13% under task-conditional swapping, 19.08% under task-agnostic) and the task-specific expert ceiling (74.96%). In the truly task-agnostic setting, getting ~19-20% accuracy on a 4-task suite is extremely low (barely above random guessing in some contexts), indicating that weight-space merging on highly disparate tasks remains highly impractical for real-world deployments. The authors should tone down their claims regarding the "success" of the merging and focus more on this persistent, fundamental performance gap.
