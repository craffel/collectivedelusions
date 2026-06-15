# Soundness and Methodology Evaluation

## Clarity of the Description
The methodology is exceptionally well-described, mathematically precise, and easy to follow. The paper clearly defines:
* The formal setup of model merging and task vectors (Section 3.1).
* The standard multi-task vision model merging evaluation protocol (unmerged disjoint heads, oracle routing, merged backbone).
* The unsupervised test-time prediction entropy optimization problem (Section 3.2).
* The mathematical formulation of the two diagnostic treatments: *Intra-Task Layer Shuffling* and *Spatial Averaging* (Section 3.3).
* The mathematical formalization of the *Spatial Averaging Paradox* through multi-task gradient imbalance and logit-scaling dynamics (Section 3.4).
* The proposed *Calibrated Prediction Entropy* remedy (Section 3.5).

The distinction between different formulations (task-wise vs. layer-wise) and the description of why prediction entropy is uncalibrated across task difficulties are particularly clear and insightful.

---

## Appropriateness of Methods
The methods used are highly appropriate for a deconstructive study:
* **Diagnostic Control Treatments**: Layer shuffling and spatial averaging are elegant, simple, and direct ways to test the structural specialization and overfitting of coefficients. They allow the authors to isolate and evaluate specific variables without introducing convoluted new pipelines.
* **Theoretical Grounding**: Explaining the Spatial Averaging Paradox through logit-scaling and entropy uncalibration provides a solid mathematical foundation. It connects the observed optimization pathologies to known properties of softmax and cross-entropy-based objectives.
* **Representational Similarity (Linear CKA)**: Linear CKA is the standard tool in deep learning for comparing internal representation manifolds. Using it across all 12 blocks provides an empirical, visual confirmation of the hierarchical routing hypothesis (how early layers remain aligned while late layers specialize).

---

## Potential Technical Flaws
A rigorous inspection reveals no major technical flaws in the methodology or reasoning. The authors have meticulously addressed potential criticisms that plagued earlier versions of the draft:
1. **The Overfitting vs. Generalization Contradiction**: The authors resolved this by reframing the "Overfitting-Optimizer Paradox" to be more nuanced. They acknowledge that layer-wise coefficients are not merely uncoordinated noise; indeed, they represent a functional routing specialized to the network's hierarchy (explaining why shuffling collapses performance and why unconstrained model outperforms spatial averaging on unseen data). However, they are simultaneously subject to transductive overfitting to the small calibration split, which post-hoc spatial averaging regularizes.
2. **Oracle Routing and Boundary Conditions**: The authors are completely transparent about the "Oracle Routing" assumption (keeping classification heads disjoint and using task identity at test-time). They explicitly state that this protocol is crucial for isolating the visual encoder's representation quality from classification head conflicts.
3. **Hyperparameter/Optimization Artifact**: The authors verified that the failure of direct task-wise optimization is a fundamental structural issue rather than an optimization artifact by conducting extensive learning rate sweeps ($10^{-4}$ to $10^{-1}$) and mutation noise scale sweeps, showing that the optimizer consistently converges to pathological regions regardless of hyperparameters.

---

## Reproducibility
The reproducibility of this work is **excellent**:
* **Complete Hyperparameter Disclosures**: Appendix A provides exhaustive details on expert head fine-tuning (optimizer, learning rate, weight decay, batch size, epochs), test-time adaptation (steps, initial coefficients, mutation noise scale, selection rules), and architectures (CLIP ViT-B/32, patch size, pixel dimensions).
* **Code and Script Structure**: The project directory contains fully functional, highly optimized, and modular scripts (`run_experiments.py` and `run_layer_cka.py`) that run on CPU or GPU and produce all metrics and figures in minutes.
* **Rigorous Seed-Control**: Every experiment is run across three independent seeds ($\mathcal{S} \in \{42, 100, 2026\}$) with standard deviations reported.
