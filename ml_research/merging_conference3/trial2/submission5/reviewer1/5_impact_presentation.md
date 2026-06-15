# 5. Presentation Quality and Practical Impact

## Major Strengths

1. **Practical Simplicity (Occam's Razor)**:
   The greatest strength of NETA is its extreme simplicity. It is an analytical, closed-form, 3-line weight-space transform. It requires **zero** optimization parameters, **zero** calibration data, and **zero** test-time training epochs, learning rates, or backpropagation passes. It is computationally instantaneous, making it highly attractive for resource-constrained edge deployments.

2. **Highly Honest and Self-Critical Narrative**:
   The authors are exceptionally honest about the limitations of their method. They do not hide the performance drop on SVHN or the fact that NETA's average accuracy is slightly lower than standard Task Arithmetic. They explicitly frame this as a "classic peak performance vs. representation fairness trade-off" and provide rigorous scale and directional alignment analyses (Section 3.4) to explain the geometric mechanics of this behavior. This level of scientific honesty is highly commendable.

3. **Exposing Crucial TTA Vulnerabilities**:
   Exposing the **Overfitting-Optimizer Paradox** is a highly valuable contribution to the community. It acts as a warning against the blind application of unsupervised test-time adaptation (prediction entropy minimization) in multi-task scenarios with high task-difficulty imbalances.

4. **Excellent Ablation Studies and Extensions**:
   The introduction of the continuous $\alpha$-relaxation, the soft-thresholding noise stabilizer $\beta$, and the analytical $\gamma^l$ scale compensation factor are highly practical additions. In particular, the $\gamma^l$ compensation factor is a powerful closed-form tool that recovers SVHN performance and raises average accuracy to 87.28% entirely zero-shot, without manual hyperparameter sweeps.

## Areas for Improvement (Practitioner's Perspective)

1. **Evaluation Scope must be Scaled Up**:
   The primary weakness of the paper is the toy-like scale of its empirical validation. To make a compelling case to practitioners, the authors must scale their evaluation to:
   - **Modern Large-Scale Benchmarks**: Evaluate on the standard 8-dataset visual classification suite (which includes ImageNet-1K, Stanford Cars, RESISC45, etc.) where domain shifts are far more realistic and challenging.
   - **Diverse Architectures**: Demonstrate NETA's generalizability on modern Large Language Models (LLMs) or Vision-Language Models (VLMs) merging, as these are the primary models being deployed in industry today.

2. **Automating the Composite Layer Grouping**:
   The composite Group 0 grouping is currently an ad-hoc, manual heuristic specific to CLIP ViT-B/32. The authors should propose a systematic, automated method for detecting and grouping structurally distinct or low-dimensional layers (e.g., based on parameter dimensionality or gradient magnitude) to make NETA truly model-agnostic and plug-and-play.

3. **Mitigating the Performance Deficit on Hard Tasks**:
   Although $\alpha$-relaxation and $\gamma^l$ compensation help recover some performance, NETA's peak performance on the hardest task (SVHN) still lags behind standard Task Arithmetic. Future work should investigate anisotropic scaling formulations (as noted in the limitations) where early layers enforce strict isotropic magnitude balancing to preserve visual stream consistency, but deeper layers allow selective task dominance based on task difficulty.

## Overall Presentation Quality
The overall presentation quality is **excellent**. The writing is clear, logical, and highly structured. The terminology is precise, the mathematical derivations are sound, and the visual representation (Table 1, Table 2, Table 3, Figure 1) is clear and easy to follow. The inclusion of the "Omitted Baselines" section shows thoroughness, and the boundary convergence analysis is incredibly insightful.

## Potential Impact and Significance
Currently, the practical impact is **moderate**. While NETA is an incredibly elegant and training-free method, its performance deficit on the hardest dataset (SVHN), lower average performance compared to Task Arithmetic, and evaluation on small-scale/toy datasets will make practitioners hesitant to adopt it over standard Task Arithmetic, DARE, or Layer-Wise AdaMerging (if calibration data is available). 

However, if the authors can successfully scale NETA to larger benchmarks (such as LLM merging or the 8-dataset visual suite) and automate the layer grouping, NETA's zero-overhead, zero-data, and closed-form design has the potential to become a standard baseline in model merging.
