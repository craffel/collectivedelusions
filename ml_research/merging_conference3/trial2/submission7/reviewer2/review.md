# Peer Review of Submission 7

## Overall Recommendation
**Score: 2: Reject**
*Justification:* While the paper presents an interesting conceptual bridge between statistical physics and neural network model merging, the submission is severely compromised by fatal flaws in the experimental setup, a fundamentally degenerate optimization formulation, critical contradictions in hyperparameter reporting, a major writing error including an active persona leak, and serious double-blind review violations in the bibliography. Given these issues, the paper falls far below the standard for acceptance.

---

## Detailed Evaluation

### 1. Soundness (Rating: Poor)
The submission suffers from major technical and methodological flaws that undermine its core claims:

*   **Degenerate Temperature Optimization:** The authors introduce trainable task-wise thermal capacities $\tau_k$ that scale the task-specific temperatures $T_k(t) = \tau_k \cdot T(t)$, optimized jointly with the merging coefficients by minimizing the Free Energy Discrepancy (Equation 24). However, optimizing KL divergence with respect to the temperature parameter is fundamentally degenerate. As $T_k \to \infty$, the Boltzmann distributions $p^{(k)}$ and $p^{(MTL, k)}$ both become completely uniform, driving the KL divergence to exactly zero. The optimizer thus has a trivial, degenerate incentive to push $\tau_k$ to its maximum allowed value, rather than finding a meaningful "local thermal equilibrium." While the authors use a hard clamp of $\tau_k \in [0.2, 5.0]$ to contain this, they do not provide any analysis of the optimized values of $\tau_k$. It is highly probable that all $\tau_k$ simply hit the upper bound of $5.0$.
*   **Irreconcilable Hyperparameter Contradictions:** There is a direct contradiction regarding the core hyperparameters of the Thermodynamic Annealing Schedule (TAS). Section 3.4 states that $T_{start} = 5.0$ and $\beta = 0.05$. However, Table 3 (Appendix), Section 4.5, and Section 5.1 state that $T_{start} = 2.0$ and $\beta = 0.40$ is the used/optimal configuration. This discrepancy makes reproducibility impossible and raises serious questions about which parameters were actually used to generate the main empirical results.
*   **Unrealistic Data Scale for Test-Time Adaptation:** The authors use a "test-time adaptation" stream of 128 images per dataset for 100 steps, which totals 12,800 images per task (51,200 unlabeled images overall). Standard test-time adaptation (TTA) is designed to be a lightweight, online adaptation phase. Using 12,800 samples per task is closer to a full-blown transductive training phase. More importantly, three of the evaluated datasets (MNIST, FashionMNIST, CIFAR-10) have test sets of exactly 10,000 images. Streaming 12,800 "fresh, unlabeled test images" from a 10,000-image dataset is physically impossible without looping or drawing samples with replacement, which violates the claim of a realistic, non-looping streaming TTA setting.
*   **Omission of Expert Baselines:** The paper completely omits the baseline classification accuracies of the individual task experts before merging. This lack of transparency is highly misleading, as we cannot evaluate the actual recovery rate or decay caused by the merging process.

---

### 2. Presentation (Rating: Poor)
While the paper is well-structured and uses sophisticated formatting, it contains severe presentation and quality-control lapses:

*   **Embarrassing Persona Leak:** In Section 5 (Conclusion and Future Horizons) on Page 4, the text literally states:
    > *"As **The Visionary**, we believe that the successful marriage of thermodynamics and model merging is not merely a specialized solution for parameter fusion..."*
    This is a severe, highly unprofessional writing error. It is a clear persona leak showing that the authors (or an automated writing agent) forgot to remove prompt-assigned persona labels from the final manuscript.
*   **Severe Double-Blind Violations in Bibliography:** The bibliography contains anonymous self-citations to concurrent submissions under review at ICML 2026. Shockingly, these citations include internal pipeline subdirectory tracking strings directly in the journal fields:
    - `journal={Under Review at ICML 2026 (Preprint from trial1\_submission7)}`
    - `journal={Under Review at ICML 2026 (Preprint from trial1\_submission2)}`
    - `journal={Under Review at ICML 2026 (Preprint from trial1\_submission10)}`
    This is a severe breach of the double-blind review process. It exposes internal pipeline logs and directory structures from the authors' workspace.
*   **Overhyped Physical Framing:** The paper relies excessively on high-flown thermodynamic metaphors ("parameter-space system frustration," "crystallization," "quenched thermodynamic equilibrium") to describe very simple, standard machine learning mechanisms (such as parameter conflict, optimizer convergence, and logit temperature scaling). This excessive hype obscures the simple reality that ThermoMerge is mathematically equivalent to standard temperature-scaled KL Knowledge Distillation.

---

### 3. Significance (Rating: Poor)
The practical utility and significance of this submission are exceptionally low:

*   **Catastrophically Poor Absolute Performance:** Under ResNet-18, ThermoMerge achieves an average accuracy of only **29.05%** across the four tasks. Most notably, its MNIST accuracy is a mere **20.00%** (barely above the 10% random chance baseline). For a pre-trained ResNet-18, individual expert models fine-tuned on MNIST and FashionMNIST easily achieve accuracies of **99%+ and 92%+** respectively. A merged model that gets 20% accuracy on MNIST is a catastrophic failure. 
*   **Fatal Resolution Mismatch:** The catastrophically poor absolute performance is caused by a severe structural mismatch in the experimental setup: resizing MNIST, F-MNIST, CIFAR-10, and SVHN to $32 \times 32$ pixels while passing them through the frozen early layers of an ImageNet-pre-trained ResNet-18 (which expects $224 \times 224$ images). This fatal resolution mismatch degrades features so severely that the backbone is rendered non-functional, invalidating any scientific conclusions drawn from the experiments.
*   **High Computational Overhead with Insignificant Gains:** ThermoMerge requires running $K$ forward passes through all frozen expert models at each adaptation step, leading to a highly prohibitive $\mathcal{O}(K)$ scaling overhead during inference. Despite this heavy cost, the average improvement over the cold alignment baseline (SyMerge) is a mere **+1.15%**, and on MNIST and FashionMNIST, ThermoMerge actually performs *worse* than standard static Task Arithmetic. 

---

### 4. Originality (Rating: Fair)
The conceptual idea of mapping logits to a Boltzmann distribution and optimizing a temperature-scaled KL divergence during model merging is moderately interesting. However, when stripped of its physical analogies, the algorithmic components are highly standard:
- Minimizing the "Helmholtz Free Energy Discrepancy" is mathematically identical to standard temperature-scaled KL Knowledge Distillation (Hinton et al., 2015).
- The "Thermodynamic Annealing Schedule" is a standard decaying temperature schedule.
- "Task-wise Thermal Coupling" is standard logit temperature scaling (Guo et al., 2017).
Combining these standard techniques is an incremental contribution, and the novelty is predominantly stylistic rather than algorithmic.

---

## Detailed Strengths and Weaknesses

### Strengths:
1.  **Correct Mathematical Derivations:** The step-by-step mathematical expansion in Appendix A linking the temperature-scaled KL divergence of Boltzmann distributions to Helmholtz free energy differences is correct and highly detailed.
2.  **Comprehensive Appendix Analyses:** The authors provide thorough sensitivity analyses for the starting temperature $T_{start}$ and thermal cooling rate $\beta$ in Appendix C.
3.  **Detailed Scaling Roadmap:** Appendix B provides a comprehensive scaling roadmap for deploying ThermoMerge on massive foundation models.

### Weaknesses (Required Actions for Revision):
1.  **Remove the Persona Leak:** Correct the highly unprofessional "As The Visionary" text in Section 5.
2.  **Resolve Hyperparameter Contradictions:** Clarify whether the TAS parameters were $T_{start} = 5.0, \beta = 0.05$ or $T_{start} = 2.0, \beta = 0.40$, and update both the Method and Experiment sections to be completely consistent.
3.  **Sanitize the Bibliography:** Remove the internal pipeline tracking strings (`trial1_submissionX`) from the bibliography.
4.  **Fix the Experimental Mismatch:** Re-run the experiments using a model architecture and image resolutions that match properly (e.g., fine-tuning on native $224 \times 224$ resolutions or using a backbone pre-trained on $32 \times 32$ inputs).
5.  **Address Trainable Temperature Degeneracy:** Analyze and report the optimized trajectories of $\tau_k$, showing whether they simply hit the upper bound clamp of $5.0$.
6.  **Disclose Baseline Expert Accuracies:** Report the original, unmerged accuracies of the individual task experts in Table 1 to provide proper scientific context.
