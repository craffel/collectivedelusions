# 3. Soundness and Methodology Check

This document provides a rigorous critique of the mathematical formulations, physical analogies, and methodological assumptions presented in **ThermoMerge**.

## 3.1. Validity of the Physical Analogies
The mapping of neural network classification outputs to physical canonical ensembles is highly elegant, but we must scrutinize the assumptions of this analogy:
1. **Energy Mapping ($E_c \equiv -f_c(x; \theta)$):** This is a standard and physically sound mapping. Classification logits are unbounded real numbers, making them suitable to represent energy states. Applying the Softmax function with temperature scaling ($T$) corresponds exactly to the canonical Boltzmann distribution where the partition function $Z$ normalizes the probabilities.
2. **Variational vs. Equilibrium Free Energy:** Section 3.3 presents a highly rigorous physical interpretation of the F-Min objective. The authors clarify a subtle thermodynamic point: the temperature-scaled KL divergence does not simply represent a difference between the equilibrium free energies of the expert and the merged model ($F_k - F_{MTL, k}$). Rather, it represents the gap between the *variational free energy* of the expert distribution evaluated under the merged model's energy states and the true *equilibrium free energy* of the merged model. This is technically precise and correct, representing a major strength in the paper's physical grounding.
3. **The Concept of System Frustration:** The paper uses the term *system frustration* from spin glass theory to describe representation interference when merging experts. While intuitive, in spin glasses, frustration refers to the impossibility of simultaneously satisfying competing microscopic coupling constraints. In model merging, the "frustration" is at the parameter-level where the optimal parameters for one task directly oppose those of another. The physical analogy is highly appropriate and serves as an excellent conceptual framework.

---

## 3.2. Methodological Assumptions and Constraints
We identify several key methodological assumptions that warrant critique:

### 3.2.1. Task-wise Thermal Capacity Clamping
In Section 3.5, the authors introduce trainable local task coupling coefficients $\tau_k > 0$, defining local temperatures $T_k(t) = \tau_k \cdot T(t)$. To prevent extreme logit scaling, they clamp $\tau_k \in [0.2, 5.0]$.
- **The Strengths:** The paper provides a clear, honest explanation of the range $[0.2, 5.0]$: temperatures below $0.2$ trigger numerical overflow and NaN gradients during backpropagation, while temperatures above $5.0$ flatten prediction probabilities into uniform high-entropy noise, destroying the gradient signal. This is a very robust physical and numerical justification that ensures perfect empirical reproducibility.

### 3.2.2. Logit Temperature Invariance at Inference
Section 3.5 includes an important clarification: because the final classification decision is obtained via an $\arg\max$ operator over the logits, and because the crystallized temperature $T_k = \tau_k$ is strictly positive, dividing the logits by $\tau_k$ does not alter their rank-ordering. Thus, the final accuracy is mathematically invariant to $\tau_k$ during evaluation.
- **The Strengths:** The authors properly identify that $\tau_k$ plays a crucial dynamic role *strictly during the optimization/adaptation phase* by shaping the gradients of the Free Energy loss and controlling parameter convergence. This is an excellent, mathematically honest disclosure.

### 3.2.3. Non-Equilibrium Adaptation Dynamics
The Thermodynamic Annealing Schedule assumes a slow, quasi-equilibrium cooling process where the system has sufficient time to reach a thermal state at each temperature step. However, test-time adaptation is limited to only 50 or 100 gradient steps.
- **The Critique:** Due to the rapid exponential decay of $T(t)$, the optimization is a fast, non-equilibrium process where the parameters may not reach true local thermal equilibria, causing sub-optimal crystallization on certain tasks (e.g., FashionMNIST). The paper acknowledges this trade-off (Section 4.3.5) and discusses non-equilibrium adaptive cooling rates, which demonstrates solid scientific integrity, but the limitation remains a core methodological challenge.

### 3.2.4. Documentation Discrepancies in Appendix C (Table 4)
We identify a few minor, yet noticeable inconsistencies between the reported configurations in Appendix C (Table 4) and the actual implementation described in the main text and executed in `experiment.py`:
- **Optimization Steps:** Table 4 lists 100 optimization steps for Test-Time Adaptation, but the code in `experiment.py` uses `TTA_STEPS_TM = 50`.
- **Annealing Schedule parameters:** Table 4 lists $T_{start}=5.0$ and $\beta=0.05$ as the adaptation hyperparameters, but the main results in Table 1 are evaluated using the optimal quenched configuration ($T_{start}=2.0, \beta=0.40$).
- **The Impact:** These documentation mismatches should be corrected in Table 4 to avoid confusing readers who attempt to replicate the exact results in Table 1 using the listed hyperparameter configurations.

---

## 3.3. Soundness Rating
- **Rating: Excellent.** The mathematical derivations are correct, and the physical interpretations are highly rigorous and theoretically sound. The authors have resolved previous major concerns, and have been highly transparent about the role of $\tau_k$ and non-equilibrium dynamics. The only remaining issue is the minor hyperparameter documentation discrepancies in Table 4.
