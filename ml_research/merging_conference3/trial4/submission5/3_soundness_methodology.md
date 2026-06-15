# Reviewer Report: 3_soundness_methodology.md

## 3. Soundness and Methodology Check

### Mathematical Correctness of Formulas
The mathematical formulation of Sparsity-Guided Task Arithmetic (SG-TA) and its variants is highly rigorous, correct, and elegantly presented. Specifically:

1. **Task Vector Definition (Eq. 1):**
   $$\tau_i = \theta_{\text{ft}, i} - \theta_{\text{pre}}$$
   This captures the standard formulation of weight offsets established in the literature.

2. **Global and Layer-wise Thresholds (Eq. 2 & 4):**
   * Global threshold $t_i = \text{Quantile}(|\tau_i|, 1 - k_i)$ and Layer threshold $t_i^{(l)} = \text{Quantile}(|\tau_i^{(l)}|, 1 - k_i^{(l)})$ are mathematically sound and properly define the quantiles of the absolute parameter shifts.
   * The binary indicator function $M_i = \mathbb{I}(|\tau_i| > t_i)$ correctly maps the coordinates exceeding the thresholds to $1$, and others to $0$.

3. **Weight Fusion (Eq. 7):**
   $$\theta_{\text{merged}} = \theta_{\text{pre}} + \sum_{i=1}^T \alpha_i (M_i \odot \tau_i)$$
   This correctly overlays the masked, scaled task vectors onto the pre-trained weights.

4. **Task Vector Magnitude Normalization (Eq. 8, 9, 10):**
   $$\mu_i = \frac{1}{D} \sum_{d=1}^D |\tau_{i, d}|$$
   $$\hat{\tau}_i = \frac{\tau_i}{\mu_i}$$
   This division scales the task vectors to have a mean absolute magnitude of $1.0$, which is a mathematically correct and elegant way to resolve magnitude imbalance across tasks.

5. **Sigmoid-Gated Soft Masking (Eq. 11):**
   $$\tau_{i, \text{soft}, d} = \sigma \left( \beta \left( \frac{|\tau_{i, d}|}{\theta_i} - 1.0 \right) \right) \cdot \tau_{i, d}$$
   This formulation is mathematically precise:
   * When $|\tau_{i, d}| > \theta_i$, the term in the parentheses is positive, and as $\beta \rightarrow \infty$, the sigmoid output approaches $1.0$.
   * When $|\tau_{i, d}| < \theta_i$, the term is negative, and as $\beta \rightarrow \infty$, the sigmoid output approaches $0.0$.
   * This continuous gate smoothly interpolates between $0$ and $1$, successfully recovering the hard binary mask in the limit, establishing rigorous mathematical consistency.

6. **Offline Few-Shot Validation Tuning (Eq. 12):**
   The maximization objective over validation splits $\mathcal{D}_{\text{val}}^{(i)}$ is correctly formalized.

### Conceptual Soundness and Assumptions
The methodology relies on several key conceptual pillars, all of which are thoroughly justified:

* **The Spatial Regularization Hypothesis:** The authors hypothesize that low-magnitude updates act as uncorrelated background noise, whereas task specialization is localized to high-magnitude updates. They support this in the discussion with direct diagnostic evidence:
  * Pairs of task vectors are highly orthogonal (similarity $0.0152$ to $0.0331$).
  * Small updates are even more orthogonal ($0.0099$ to $0.0169$), proving they represent independent noise.
  * Binary masks generated under GQ at $k=0.3$ have an overlap of $10.17\%$ to $11.06\%$, which is extremely close to the random baseline ($9\%$), proving that task-specific updates do not cluster in the same parameters and can be consolidated without high spatial overlap.
  This provides excellent empirical grounding for the mathematical assumptions of the paper.

* **Classification Heads and Routing Oracle:**
  * In multi-task model merging, it is common to assume separate classification heads and route test samples to their respective task heads. The authors explicitly state this standard convention and discuss practical zero-shot routing alternatives (e.g., predictive entropy minimization, training a compact shared multi-task head). This ensures transparency.

* **Few-Shot Validation Noise and Mitigation:**
  * When normalizing vectors, small calibration sets ($N_{\text{val}}=10$) introduce standard deviation ($\pm 4.56\%$). The authors explain this as localized validation noise and validate this hypothesis by conducting a physical size sweep over $N_{\text{val}} \in [10, 20, 50, 100]$. They show that doubling $N_{\text{val}}$ to $20$ immediately slashes the variance by 4x (to $\pm 1.10\%$) and boosts performance to $63.73\%$, which is a highly actionable and scientifically robust validation.

* **Optimization Sequence Dependency:**
  * Coordinate descent (CS) sequentially optimizes task coordinates and is theoretically sensitive to order. The authors address this by running a control sweep in reverse order, which converges to nearly identical performance ($58.28\% \pm 2.45\%$ vs. $58.40\% \pm 2.32\%$), validating the optimization stability.

### Overall Soundness Rating
**Excellent.** There are no logical inconsistencies, mathematical gaps, or unaddressed assumptions. The authors have systematically anticipated potential criticisms and addressed them through elegant mathematical formulations and rigorous physical control sweeps.
