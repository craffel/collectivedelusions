# Soundness and Methodology Evaluation

## Clarity of Description
The methodology of the paper is exceptionally clear and mathematically rigorous. The equations for weight-space task vectors (Equation 1), Task Arithmetic (Equation 2), layer-wise AdaMerging (Equation 3), task-wise AdaMerging (Equation 4), and joint Shannon prediction entropy (Equation 5) are standard and formally correct. The definitions of the proposed diagnostic treatments (Equations 6, 7, and 8) are mathematically precise and easy to comprehend.

## Appropriateness of Methods
- **Diagnostic Controls**: The use of *Intra-Task Layer Shuffling* and *Spatial Averaging* as control treatments is highly creative, clean, and appropriate for isolating the effects of architectural hierarchy vs. global scaling.
- **Linear CKA Similarity**: Using Linear CKA similarity to measure the representational alignment with the target task expert is standard and highly appropriate for weight-space merging research.
- **Multi-Seed Protocol**: Evaluating over three independent seeds ($\mathcal{S} \in \{42, 100, 2026\}$) with standard deviations is methodologically sound and crucial given the high variance observed on SVHN ($\pm 5\%$ to $\pm 8\%$).

## Reproducibility
The reproducibility of the empirical results appears very high. The authors provide:
- Exact dataset partition details (512 head training images, 64 calibration images, full test splits for evaluation).
- Exact optimizer settings (AdamW, learning rate $10^{-3}$, 5 epochs for head fine-tuning; Adam GD, learning rate $10^{-2}$, 200 steps, or 1+1 ES, mutation noise scale $\sigma = 0.01$, 500 steps for test-time adaptation).
- The exact model architecture (CLIP ViT-B/32 visual encoder with 12 Transformer block layers).

## Potential Technical Flaws & Conceptual Critiques

### 1. Conceptual Contradiction: Framing "Overfitting" as a Pathology
The authors frame the high-dimensional optimization of layer-wise AdaMerging as suffering from the **Overfitting-Optimizer Paradox**. They claim that the 1,000-parameter space is "prone to test-time overfitting."
- However, the empirical results show that AdaMerging (SOTA - Adam GD) achieves **88.05%** average accuracy on the full standard test splits (totaling 56,032 images) after adapting on just 64 unlabeled images per task!
- Spatially Averaging these parameters post-hoc (reducing degrees of freedom to 4) causes a performance **drop** of $3.09\%$, down to **84.96%**.
- If a model optimized on a tiny batch generalizes *better* to a massive, disjoint test split, it cannot be said to be suffering from a harmful "overfitting pathology" in a practical sense. To a practitioner, the layer-wise adaptation is simply capturing highly beneficial local representational routing that improves multi-task capability.
- The authors themselves admit in Section 4.2.4 that the $3.09\%$ drop represents "sacrificing the beneficial layer-specific routing" which is "structurally specialized." Thus, framing this as a "paradox of overfitting" is a substantial conceptual stretch. The unconstrained high-dimensional optimization is functionally superior for generalization, and the post-hoc spatial average is a degraded, less-expressive model, rather than a "better-regularized" one.

### 2. Failure of the Proposed Algorithmic Remedy (Calibrated Prediction Entropy)
The authors formulate **Calibrated Prediction Entropy** (Section 3.5) to resolve the multi-task gradient imbalance in direct task-wise optimization. 
- In Table 1, Calibrated Task-wise AdaMerging (Adam GD) achieves only **80.59%** accuracy, which is worse than uncalibrated Task-wise AdaMerging ($81.19\%$) and significantly worse than Task Arithmetic ($84.64\%$).
- The fact that the proposed remedy fails to improve performance indicates a flaw in the paper's constructive contribution. The authors state that this is because a global, low-dimensional bottleneck cannot resolve joint weight-space interference regardless of calibration. While this explains the failure, from a practical standpoint, it leaves the practitioner with no actionable, functional new algorithm to stabilize direct low-dimensional test-time adaptation.

### 3. Oracle Routing Assumption
The paper relies on the standard model merging assumption of **Oracle Routing** at evaluation time (Section 3.1). While this isolates visual representation quality, it represents a substantial gap in practical deployability. In real-world industrial environments, a multi-task system must route unlabeled test inputs to their correct heads autonomously. Relying on an oracle classifier to route inputs means this technique cannot be deployed "as-is" in fully unlabeled settings, a significant limitation for practitioners.
