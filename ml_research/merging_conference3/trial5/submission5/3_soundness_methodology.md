# Technical Soundness and Methodology Review

## 1. Overall Soundness Rating
**Rating**: **Excellent**

The technical soundness of this paper is outstanding. The authors do not merely present an empirical comparison; they conduct a systematic, multi-tiered scientific inquiry that isolates variables with exceptional rigor. Every claim is validated through multiple independent methodologies, including mathematical proof, controlled simulation (sandbox), real-scale vision-language verification, optimization sweeps, multi-seed statistical audits, and task-correlation sweeps.

---

## 2. Rigor of Methodological Controls

The authors employ several commendable methodological controls to ensure their findings are robust and free of confounding factors:

### A. The Isolating Coordinate Sandbox
The sandbox is a key methodological strength. By separating the total error of a merged model into:
$$\text{Error}_{total} = \text{Error}_{routing} + \text{Error}_{alignment}$$
and setting $\text{Error}_{alignment} \approx 0$ via fixed, orthogonal feature coordinates, the authors isolate the behavior of dynamic routing equations in a controlled coordinate field. This prevents high-dimensional coordinate misalignment in the backbone from masking the systematic failures of the routing math itself.

### B. Scientific Fairness Controls
In Appendix B (Optimization Sensitivity Audit), the authors apply strict controls to ensure that the catastrophic collapse of QWS-Merge is not an artifact of bad hyperparameters or sub-optimal learning rates. They sweep learning rates across five scales:
$$\eta \in \{10^{-2}, 5 \cdot 10^{-3}, 10^{-3}, 5 \cdot 10^{-4}, 10^{-4}\}$$
and confirm that peak QWS-Merge performance occurs at $\eta = 10^{-2}$, while lower learning rates lead to total collapse due to the highly non-convex, rugged cosine landscape. 

Furthermore, they implement exact initialization and gradient norm clipping (threshold of 1.0) matching the original QWS-Merge codebase. This rules out optimization bias and confirms that the failure of QWS-Merge is a fundamental structural limitation of wave-based, non-monotonic routing equations.

### C. Multi-Seed Robustness Audit
In Appendix D, the authors execute their entire pipeline across five independent random seeds:
$$\text{Seeds} \in \{10, 11, 12, 13, 14\}$$
regenerating the dataset splits and training all models from scratch. The statistical results verify that the superiority of classical routing and the collapse of QWS-Merge are consistent, robust behaviors:
* **Linear Router (Unreg)**: 69.68% $\pm$ 1.11%
* **L3-Linear (L2 Reg)**: 60.12% $\pm$ 2.99%
* **QWS-Merge SOTA**: 33.34% $\pm$ 9.51% (extremely high variance and consistent collapse)

### D. Task Correlation and Overlap Sweep
In Appendix E, the authors sweep a task-correlation parameter:
$$\rho \in \{0.0, 0.25, 0.50, 0.75\}$$
to ensure that orthogonal task boundaries did not artificially favor linear routing. Even when tasks overlap by 75%, classical linear routing systematically dominates wave-based routing.

### E. Deep Layer-by-Layer Weight Merging (No Averaging)
To address the concern that the sandbox classification-head setup collapsed the layer dimension, the authors design a 14-layer sequential expert merging framework in Appendix F. Under this advanced setup where routing parameters do not collapse, QWS-Merge still collapses catastrophically (Joint Mean: **10.60%**), while the global Linear Router achieves the peak Joint Mean (**35.50%**). The authors explain this through gradient backpropagation dynamics: under extreme data scarcity, backpropagating gradients through a 14-layer chain of product terms:
$$\frac{\partial \mathcal{L}}{\partial \alpha_{k, b}(l)} = \left( \frac{\partial \mathcal{L}}{\partial H^{(L)}_b} \prod_{j=l+1}^L \frac{\partial H^{(j)}_b}{\partial H^{(j-1)}_b} \right) \frac{\partial H^{(l)}_b}{\partial \alpha_{k, b}(l)}$$
introduces massive optimization noise and co-adaptation, causing unconstrained multi-layer routing to experience uncontrolled scaling and numerical collapse.

---

## 3. Analysis of Mathematical Proofs

The closed-form proof of **Layer-Averaging Collapse** is mathematically sound and holds universal significance for any dynamic model-merging scheme that merges a shared, global parameter group (like a classification head, an embedding layer, or a language model projection head). 

### Proof Validation:
1. Dynamic merged weight matrix at layer $l$:
   $$W_{merged}^{(l)}(x) = W_{base}^{(l)} + \sum_{k=1}^K \bar{\alpha}_k(l) V_k^{(l)}$$
2. In single-head classifiers, the final classification head is merged via:
   $$\bar{\alpha}_k = \frac{1}{L} \sum_{l=1}^L \alpha_{l, k}$$
3. Substituting layer-wise linear coefficients:
   $$\alpha_{l, k} = \langle \psi(x)_b, W_{l, k} \rangle + B_{l, k}$$
4. Averaged result:
   $$\bar{\alpha}_k = \frac{1}{L} \sum_{l=1}^L \left( \langle \psi(x)_b, W_{l, k} \rangle + B_{l, k} \right) = \langle \psi(x)_b, \frac{1}{L} \sum_{l=1}^L W_{l, k} \rangle + \frac{1}{L} \sum_{l=1}^L B_{l, k} = \langle \psi(x)_b, W_{eff, k} \rangle + B_{eff, k}$$
This algebraic step is absolutely correct and mathematically elegant. It proves that the 14-layer specialized routing parameter space is functionally isomorphic to a single-layer routing space, but with 14 times more parameters—introducing massive parameter redundancy and optimization noise under low-data splits (64 samples). This is a vital architectural insight for future network designs.

---

## 4. Assessment of Scientific Honesty and Refutation of Confounders

A major strength of this paper is its transparent, scientifically honest, and self-critical approach:
* The authors do not hide the limitations of their sandbox; they acknowledge the scale gap and provide a concrete deployment roadmap (Appendix A) and a real-scale CLIP pilot (Section 4.5).
* They deconstruct their own proposed model **L3-Softmax**, showing that its apparent robustness is an illusion (Section 5, Appendix C).
* They address the absolute parameter footprint savings (Section 3.4, Appendix G.2), clarifying that while their L3-Router saves 16.7% parameters over QWS-Merge, this absolute saving (56 parameters) is practically negligible compared to backbone model scales. They frame this savings as having **theoretical and structural interest** (proving classical linear channels achieve superior dynamic capacity without redundant auxiliary wave variables) rather than practical hardware storage benefits.

This exemplary scientific hygiene sets a high bar for machine learning publications.
