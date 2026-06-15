# Evaluation Phase 5: Presentation, Strengths, and Impact Evaluation

## Major Strengths
1. **Rigorous Appendix Derivations:** The step-by-step mathematical derivation in Appendix A connecting the temperature-scaled KL divergence of Boltzmann distributions to the expected negative energy differences and Helmholtz free energy differences is correct and highly detailed.
2. **Detailed Scaling Roadmap:** Section 6 (Appendix) provides a comprehensive, highly concrete scaling roadmap for deploying ThermoMerge on massive foundation models, including PEFT (ThermoLoRA) and a multimodal logit-to-energy formulation.
3. **Hyperparameter Sensitivity Analyses:** The empirical sensitivity analyses of the starting temperature $T_{start}$ and thermal cooling rate $\beta$ (Section 8, Appendix) are highly thorough and provide interesting physical insights (e.g., the "annealing versus quenching" trade-off).
4. **Strong Comparison with Adaptive Baselines:** The authors compare their method against a good suite of standard test-time adaptive baselines (AdaMerging and SyMerge).

## Areas for Improvement (Major Weaknesses and Writing Errors)

### 1. Embarrassing Persona Leak in the Body of the Paper
In Section 5 (Conclusion and Future Horizons) on Page 4, the text literally states:
> *"As **The Visionary**, we believe that the successful marriage of thermodynamics and model merging is not merely a specialized solution for parameter fusion, but the first step..."*
This is a severe and unprofessional writing error. It is a clear **persona leak** showing that the authors (or an automated writing agent) forgot to remove prompt-assigned persona labels from the final LaTeX manuscript. This heavily degrades the scientific presentation and quality control of the submission.

### 2. Contradictory Hyperparameter Disclosures
As noted in Phase 3, the paper contains an explicit contradiction regarding the hyperparameters used for the Thermodynamic Annealing Schedule (TAS):
- Section 3.4 lists $T_{start} = 5.0$ and $\beta = 0.05$.
- Table 3, Section 4.5, and Section 5.1 list $T_{start} = 2.0$ and $\beta = 0.40$ as the configurations.
This discrepancy makes reproducibility impossible and must be resolved.

### 3. Serious Double-Blind Violations in Bibliography
The bibliography in `references.bib` contains several anonymous self-citations to concurrent submissions under review at ICML 2026. Crucially, the authors have included internal pipeline-specific subdirectory tracking strings directly in the journal fields:
- `journal={Under Review at ICML 2026 (Preprint from trial1\_submission7)}`
- `journal={Under Review at ICML 2026 (Preprint from trial1\_submission2)}`
- `journal={Under Review at ICML 2026 (Preprint from trial1\_submission10)}`
This is a severe breach of the double-blind review process. It exposes internal pipeline logs and directory structures from the authors' workspace, representing extremely poor academic practice.

### 4. Overhyped Physical Terminology
The paper relies excessively on high-flown thermodynamic metaphors ("parameter-space system frustration," "replica symmetry breaking," "crystallization," "quenched thermodynamic equilibrium") to describe very simple, standard machine learning mechanisms (e.g., parameter conflict, optimizer convergence, and logit temperature scaling). This excessive hype obscures the simple reality that ThermoMerge is mathematically equivalent to standard temperature-scaled KL Knowledge Distillation.

### 5. Choice of Non-Functional Experimental Setting
Resizing MNIST, F-MNIST, CIFAR-10, and SVHN to $32 \times 32$ pixels while passing them through frozen early layers of an ImageNet-trained ResNet-18 is a severe structural mismatch. This fatal mismatch degrades the features so severely that the merged models perform catastrophically poorly (e.g., **20.00% accuracy on MNIST**, which is near-random chance). An experimental setting where all methods hover around 20%-30% absolute accuracy is not a reliable testbed for validating model merging.

## Overall Presentation Quality
The overall presentation quality is **poor to fair**. While the LaTeX layout, mathematical formatting, and multi-panel figures are well-crafted, the presence of an active persona leak ("As The Visionary"), severe double-blind violations in the bibliography, direct hyperparameter contradictions, and overhyped terminology significantly damage the academic rigor of the paper.

## Potential Impact and Significance
The practical significance of this paper is **very low**:
- **High Computational Overhead:** ThermoMerge requires running $K$ forward passes through all frozen expert networks at each adaptation step, scaling as $\mathcal{O}(K)$. For realistic multi-task settings where $K \ge 20$ or $50$, this is computationally and memory prohibitive.
- **Insignificant Accuracy Gains:** The margin of improvement is extremely small (only **+1.15%** average accuracy over SyMerge on ResNet-18, and actually performing *worse* than static baselines on MNIST and FashionMNIST).
- **Impracticality:** Practitioners are highly unlikely to adopt an online test-time adaptation framework with $\mathcal{O}(K)$ overhead and highly sensitive hyperparameter schedules (temperature decay, scaling constraints) for such minor and inconsistent improvements.
