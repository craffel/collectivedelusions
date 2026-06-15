# 3. Soundness of Methodology

We evaluate the mathematical formulations, logical reasoning, and soundness of the methodology proposed in the paper.

## 1. Mathematical Formulation & Implementation Discrepancy (Critical Weakness)
We identified a critical discrepancy between the mathematical formulas in the paper and the actual code implementation for **Elastic Spatial Regularization (ESR)**:
- **In the paper (Equation 10):**
  $$\mathcal{R}_{\text{spatial}}(\Lambda) = \frac{\beta}{K L} \sum_{k=1}^K \sum_{l=1}^L (\lambda^l_k - \lambda_{\text{init}})^2 + \frac{\gamma}{K L} \sum_{k=1}^K \sum_{l=1}^L (\lambda^l_k - \bar{\lambda}_k)^2$$
  Here, the regularization term is explicitly normalized by dividing by the total number of layer-task coordinates $K L$. The text notes: *"By dividing the sum by the total number of layer-task coordinates $K L$, we normalize the magnitude of the penalty to be scale-invariant ($O(1)$)..."*
- **In the actual code implementation (`run_regcalmerge.py`):**
  ```python
  proximity_penalty = torch.sum((lambdas_raw - 0.3) ** 2)
  spatial_dev_penalty = torch.sum((lambdas_raw - mean_lambdas) ** 2)
  total_loss = losses + beta * proximity_penalty + gamma * spatial_dev_penalty
  ```
  The code performs a raw sum over the parameters and does **not** divide by $K L$ (or $(K-1)L$, since only the task-expert coefficients are optimized).
- **The Impact:**
  In their concrete setup, there are $L = 13$ layer groups and $K-1 = 3$ optimized task vectors (as the pretrain model coefficient is fixed at 1.0 and not optimized), yielding a total of $(K-1)L = 39$ coordinates.
  Because the code does not divide by this factor, the actual effective regularization applied in the experiments is **$39\times$ stronger** than what the normalized formula in Equation 10 describes!
  Although the authors include a footnote/remark stating that *"the division by $K L$ is absorbed directly into the scaled hyperparameter values..."*, this is highly problematic for reproducibility:
  1. If a practitioner implements Equation 10 exactly as written and uses the optimal hyperparameters ($\beta=1.0, \gamma=1.0$) reported in Table 3, their regularization will be $39\times$ weaker than what the authors actually used.
  2. The paper's claim of presenting a "normalized formulation" that ensures "hyperparameters remain stable and transferable across larger architectures" is undercut by the fact that the actual experiments used the unnormalized version.
  **Actionable Recommendation:** The authors must either update the mathematical formula in Equation 10 to match the unnormalized sum, or correct the code implementation to divide by $KL$ and rescale the reported grid sweep hyperparameters to their actual normalized equivalents.

---

## 2. Omission of SNEW Division-by-Zero Safeguard
The Scale-Normalized Entropy Weighting (SNEW) is mathematically defined as:
$$w_k = \frac{1}{\bar{\mathcal{H}}_k(\Lambda_{\text{init}})}$$
- **The Issue:**
  If any task expert is extremely confident and achieves perfect prediction confidence on the tiny calibration batch, its entropy $\bar{\mathcal{H}}_k(\Lambda_{\text{init}})$ will be $0.0$, leading to division by zero and a complete collapse of the optimizer.
- **The Code Solution:**
  In the implementation, the authors successfully employ a safeguard:
  `scale_weights[dataset_name] = 1.0 / max(ent, 1e-5)`
- **The Critique:**
  While the implementation is sound, this practical detail and safeguard are completely omitted from the text. For mathematical rigor and completeness, the paper should explicitly state the use of this numerical stability safeguard in the methodology section.

---

## 3. Class-Capacity Normalization (CCN) Bounded Domain
Equation 11 defines the capacity-normalized entropy by dividing by $\log C_k$.
- If $C_k = 1$ (a single-class dataset, which is degenerate but theoretically possible), $\log C_k = 0$, leading to division by zero.
- The authors should formally specify that CCN is defined on classification tasks with $C_k \ge 2$, establishing a bounded domain for the normalization factor.

---

## 4. Logical Consistency and Theoretical Framing
On a conceptual level, the paper's reasoning is highly sound and logically consistent:
- **Hierarchical Representational Conflict:** The authors provide an elegant, representation-theoretic explanation for why ESR smoothing ($\gamma > 0$) causes a slight drop in accuracy. In deep neural networks, early layers learn generic features, while deep layers learn abstract, task-specific features. Thus, forcing the merging coefficients to be homogeneous across layers directly restricts the network's capacity to adjust early vs deep layers independently. This is a very insightful explanation that bridges parameter-space regularization with representation learning theory.
- **Benchmark Honesty:** The authors are exceptionally honest about the homogeneous nature of the standard benchmark (where $C_k = 10$ for all datasets, making CCN a global scalar scaling), and use a clever label-restricted simulation to validate the behavior of CCN under true heterogeneous conditions. This shows high scientific integrity.
