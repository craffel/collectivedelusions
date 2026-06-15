# 3. Soundness and Methodology Evaluation

This evaluation focuses on the clarity, appropriateness, potential technical flaws, and reproducibility of the proposed ThermoMerge methodology.

## Rating: Fair

While the paper is written with high technical and rhetorical polish, a closer examination of its mathematical and experimental formulations reveals several critical technical flaws, hyperparameter contradictions, and logical inconsistencies.

---

## 1. Critical Technical Flaws

### A. Trivial Global Minimization in Temperature Optimization
In Section 3.5, the authors introduce trainable task-specific thermal capacities $\tau_k > 0$ that scale the temperature during test-time adaptation: $T_k(t) = \tau_k \cdot T(t)$. These parameters are optimized alongside the merging coefficients $\boldsymbol{\Lambda}$ using the F-Min objective:
$$\mathcal{L}(\boldsymbol{\Lambda}, \boldsymbol{\tau}) = \sum_{k=1}^K \mathbb{E} [ T_k(t) \cdot \mathcal{D}_{KL}(p^{(k)}(x; T_k(t)) \parallel p^{(MTL, k)}(x; T_k(t))) ]$$

Mathematically, this objective has a **trivial global minimum** as $T_k \to \infty$ (and thus as $\tau_k \to \infty$):
* As the temperature $T_k$ increases, the Boltzmann probability distributions $p^{(k)}$ and $p^{(MTL, k)}$ both flatten and approach the uniform distribution.
* The KL divergence between two uniform distributions is exactly zero: $\lim_{T \to \infty} \mathcal{D}_{KL}(p(T) \parallel q(T)) = 0$.
* Even when scaled by $T_k(t)$, the term $T_k \cdot \mathcal{D}_{KL}$ vanishes as $\mathcal{O}(1/T_k)$ at high temperatures.

Consequently, any gradient-descent optimizer minimizing the F-Min loss with respect to $\tau_k$ will trivially push $\tau_k$ toward its maximum allowed value to drive the loss to zero. To prevent numerical overflow, the authors clamp $\tau_k \in [0.2, 5.0]$. However, because of the vanishing KL divergence, the optimizer is pathologically driven to push $\tau_k$ to the upper bound of **5.0**. 
The claim that "task-wise local temperatures $T_k(t)$ adaptively converge to their task-specific equilibria" is theoretically flawed; $\tau_k$ does not find a physical equilibrium, but rather hits an arbitrary clamping wall because of a structural defect in the objective function.

### B. Invariance of $\tau_k$ during Inference
As the authors correctly note in Section 3.5:
"Since the crystallized temperature $T_k = \tau_k$ is strictly positive ($\tau_k \in [0.2, 5.0]$), dividing the logits by a constant positive scalar does not alter their rank-ordering. Consequently, the final classification accuracy on any individual task is mathematically invariant to the value of $\tau_k$ during evaluation."

This means that the learned "crystallized" task-wise temperatures are entirely discarded during final inference. Their only role is to scale gradients during the TTA phase. Given the trivial minimization flaw described above, the optimization of $\tau_k$ is essentially a complex, bounded gradient-scaling heuristic rather than a meaningful thermodynamic adaptation of the model's representations.

---

## 2. Mathematical and Hyperparameter Inconsistencies

There are glaring contradictions between the hyperparameters reported in the main text, the abstract, and the appendix, which severely undermines the paper's scientific rigor and reproducibility:

1. **Starting Temperature ($T_{start}$):**
   * The **Abstract** and **Section 3.4** state that the Thermodynamic Annealing Schedule (TAS) starts at $T_{start} = 5.0$.
   * **Appendix Table 3** and **Appendix Section 4.5.1** state that $T_{start} = 2.0$.
2. **Cooling Rate ($\beta$):**
   * **Section 3.4** states that the cooling rate is $\beta = 0.05$.
   * **Appendix Table 3** and **Appendix Section 4.5.2** state that the cooling rate is $\beta = 0.40$.
3. **Optimization Steps:**
   * **Section 4.1.4** states that the adaptation is run for **100 steps** for all methods.
   * **Appendix Table 3** states that the adaptation is run for **100 steps (50 steps for ThermoMerge)**.

These discrepancies leave the reader guessing which configuration was actually used to produce the results in Table 1, and make direct replication impossible.

---

## 3. Appropriateness of the Experimental Baseline and Setup

### A. Evaluation on a Disconnect representation space
The absolute accuracy levels across all evaluated methods are exceptionally poor:
* **MNIST accuracy:** 20.00% (ThermoMerge) vs. 21.40% (Task Arithmetic). Standard ResNet-18 fine-tuning on MNIST should easily yield >99% accuracy. An accuracy of 20% on a 10-class dataset is barely above random guessing.
* **FashionMNIST accuracy:** 32.60% (ThermoMerge) vs. 35.40% (Task Arithmetic).
* **Average multi-task accuracy:** ~29% for the best-performing method.

An average accuracy of 29% across these simple datasets (MNIST, FashionMNIST, SVHN, CIFAR-10) indicates that the merged models have suffered a near-total representation collapse. A "SOTA" improvement of 1.8% over Task Arithmetic is of little practical significance when the absolute performance of the model remains completely non-functional. This suggests that either the task-specific classification heads are interfering severely, the experts were trained in a way that destroyed mode connectivity, or the test-time adaptation learning rate was set too high, causing catastrophic representation drift.

### B. The SimpleCNN "Strawman" Baseline
The comparison of pre-trained ResNet-18 with a from-scratch SimpleCNN backbone is presented as a major empirical discovery showing that "ancestral connectivity resolves the Gray-to-Color collapse." 
However, it is a well-established and foundational fact in the model merging literature that models *must* share a pre-trained ancestor to reside in a common basin and exhibit linear mode connectivity. Merging models trained from scratch is known to fail catastrophically due to permutation symmetries and representation misalignment. Setting up a from-scratch baseline to prove that model merging fails without pre-training is a strawman comparison that contributes nothing new to our understanding of the field.
