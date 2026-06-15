# Mock Review: RegCalMerge (Calibrated & Regularized Test-Time Model Merging)

**Overall Recommendation:** **5: Accept** (Technically solid paper, with high impact on test-time model adaptation and merging, excellent presentation, high conceptual originality, and strong scientific transparency, with a few addressable implementational and empirical weaknesses.)

---

## 1. Summary of the Submission
This paper deconstructs the current test-time, entropy-based model merging landscape (such as **AdaMerging**) and identifies two critical failure modes: 
1. **The Overfitting-Optimizer Paradox (Transductive Overfitting):** Layer-wise continuous merging coefficients overfit to the local statistics of small, test-time calibration batches rather than capturing stable, localized layer interactions.
2. **Sacrificial Task Bias:** Standard uncalibrated joint entropy objectives prioritize simpler, low-entropy tasks (e.g., MNIST) and severely degrade complex, high-entropy tasks (e.g., SVHN).

To resolve these failures, the paper proposes **RegCalMerge**, which introduces:
- **CalMerge (Calibration Engine):** Combining **Class-Capacity Normalization (CCN)** and **Scale-Normalized Entropy Weighting (SNEW)** to balance multi-task gradients.
- **Elastic Spatial Regularization (ESR):** An optional dual-penalty regularizer utilizing a **Proximity Penalty** ($\beta$) and a **Spatial Deviation Penalty** ($\gamma$) to smooth coefficients across layers and prevent transductive parameter drift.

The paper presents a comprehensive, multi-seed empirical analysis on CLIP ViT-B/32 across four diverse visual domains, demonstrating that **CalMerge** achieves state-of-the-art Joint Mean accuracy (**61.82%**), while **ESR** offers a controllable dial to trade off peak local accuracy for global parameter-space stability. They also present a specialized, label-restricted simulation to validate their calibration mechanisms under heterogeneous class capacities.

---

## 2. Key Strengths
- **Conceptual Novelty of Diagnostics:** The identification of the **Overfitting-Optimizer Paradox** is highly original and significant. The introduction of the **"spatial shuffling diagnostic"**—which shows that randomly shuffling optimized layer-wise coefficients across layers preserves ~95% of performance gains—elegantly and empirically deconstructs standard adaptive merging, proving that the optimizer acts primarily as a transductive drift mechanism rather than discovering genuine layer-wise representational mixtures.
- **Scientific Integrity and Transparency:** The authors exhibit an exceptionally high standard of scientific honesty. They explicitly address and analyze their own experimental constraints, such as the deterministic $\pm0.00\%$ standard deviation of gradient descent across seeds, the homogeneous nature of standard visual benchmarks (and designing a custom label-restricted setup to resolve it), and small test splits.
- **Robust and Controllable Framework:** **Elastic Spatial Regularization (ESR)** successfully bridges the gap between fully parameterized layer-wise merging and completely collapsed scalar-per-task merging, establishing a stable, predictable, and mathematically sound optimization surface.
- **Clear and Engaging Exposition:** The paper is beautifully written, with clear mathematical formulations, memorable and descriptive terminology (e.g., *Hierarchical Representational Conflict*), and excellent flow.

---

## 3. Key Weaknesses & Areas for Improvement (Up to 3 Critical Flaws)

### Flaw 1: Implementational Discrepancy in Elastic Spatial Regularization (ESR)
There is a major discrepancy between the mathematical formulation of ESR in the paper and its actual implementation in the codebase, which harms scientific reproducibility:
- **Mathematical Formula (Equation 10):** Normalizes the proximity and spatial deviation penalties by dividing by the total number of layer-task coordinates $K L$:
  $$\mathcal{R}_{\text{spatial}}(\Lambda) = \frac{\beta}{K L} \sum_{k=1}^K \sum_{l=1}^L (\lambda^l_k - \lambda_{\text{init}})^2 + \frac{\gamma}{K L} \sum_{k=1}^K \sum_{l=1}^L (\lambda^l_k - \bar{\lambda}_k)^2$$
- **Code Implementation (`run_regcalmerge.py`):** Calculates a raw sum over the coordinates without dividing by $KL$ (or $(K-1)L$ for optimized task coefficients):
  ```python
  proximity_penalty = torch.sum((lambdas_raw - 0.3) ** 2)
  spatial_dev_penalty = torch.sum((lambdas_raw - mean_lambdas) ** 2)
  total_loss = losses + beta * proximity_penalty + gamma * spatial_dev_penalty
  ```
- **The Impact:** In their benchmark, there are $L=13$ layer groups and $K-1=3$ optimized task coefficients, making $(K-1)L = 39$. Because the implementation does not divide by this factor, the actual effective regularization applied in the experiments is **$39\times$ stronger** than what the normalized formula in the paper describes.
- **Actionable Suggestion:** The authors must correct this discrepancy. If they wish to keep the mathematically elegant normalized formulation, they should update their code to divide the penalties by $(K-1)L$ and rescale the reported optimal hyperparameters in Table 3. Alternatively, they should update the text to state that the reported optimal parameters refer to the unnormalized formulation, and provide the exact unnormalized formula.

### Flaw 2: Evaluation Split Size and Data-Sampling Robustness
The quantitative evaluations are conducted on extremely restricted test splits (256 images total per dataset):
- **The Issue:** Since MNIST, FashionMNIST, and CIFAR-10 have test sets of 10,000 images, evaluating on only 256 images (2 batches of size 128) represents just 1% to 2.5% of the standard test sets, making accuracy estimates highly susceptible to sample noise.
- **The Seed Trap:** Because the 3 random seeds are evaluated on the exact same cached in-memory calibration batch, first-order gradient descent (Adam GD) is completely deterministic across seeds ($\pm0.00\%$ standard deviation). This means the multi-seed evaluation does not actually capture **data-sampling robustness**—i.e., how sensitive the method is to *which* 16 calibration images are selected.
- **Actionable Suggestion:** To make the results fully publication-grade, the authors should:
  1. Evaluate the final frozen merged models on the **entire** standard test splits of each dataset to eliminate sample-split variance.
  2. Sample **different random calibration batches** across different seeds. This would produce realistic non-zero standard deviations for Adam GD and prove that RegCalMerge is robust to the specific test-time calibration samples.

### Flaw 3: Implementation Omission of the "Calibrated Spatial Mean (Cal-Mean)" Baseline
The paper introduces and heavily discusses Method 9, **Calibrated Spatial Mean (Cal-Mean)** (yielding 61.13% Joint Mean in Table 1), as a crucial baseline to prove that layer-wise degrees of freedom are indeed necessary:
- **The Issue:** While Cal-Mean is implemented in `run_heterogeneous_experiment.py` for the heterogeneous label space, the main pipeline script `run_regcalmerge.py` (which generates the metrics for the homogeneous Table 1) **does not actually execute** or support the Cal-Mean baseline.
- **Actionable Suggestion:** To ensure absolute codebase-to-paper consistency, the authors should integrate the Cal-Mean evaluation block directly into the main `run_regcalmerge.py` script.

---

## 4. Detailed Feedback, Questions, and Suggestions for the Authors
- **SNEW Division-by-Zero Safeguard:** Mathematically, SNEW ($w_k = 1 / \bar{\mathcal{H}}_k$) can trigger division-by-zero if a task expert achieves zero entropy on the calibration batch. While the implementation handles this safely using `max(ent, 1e-5)`, this practical numerical safeguard is omitted from the methodology text. We suggest adding a brief sentence in Section 3.4 specifying this numerical stabilizer.
- **CCN Bounded Domain:** For mathematical completeness, please specify in Section 3.3 that CCN is formally defined on classification tasks with $C_k \ge 2$, avoiding the boundary limit of $\log C_k = \log 1 = 0$.
- **Calibration Stream Volume Sweep:** Since the core contribution of the paper is resolving transductive overfitting on tiny calibration splits, a sweep over different calibration batch sizes (e.g., $N \in \{8, 16, 32, 64, 128\}$) would be highly valuable. Adding such a sweep would illustrate how the overfitting paradox and the benefits of ESR change as more test-time data becomes available, greatly enhancing the empirical depth of the paper.

---

## 5. Final Recommendation
This is an exceptionally strong, well-written, and intellectually engaging submission that makes a high-impact contribution to the model merging literature. The "spatial shuffling diagnostic" represents a brilliant paradigm shift for how we evaluate test-time adaptation. Once the implementational discrepancy (Flaw 1), evaluation split scale/seed issues (Flaw 2), and baseline execution gaps (Flaw 3) are resolved, this paper will be a highly solid and outstanding contribution. I strongly recommend its acceptance.
