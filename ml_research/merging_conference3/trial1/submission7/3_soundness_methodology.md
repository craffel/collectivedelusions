# Intermediate Review Report 3: Soundness and Methodology

## 1. Clarity and Notation of the Description
The methodology of the paper is exceptionally clear, logical, and scholarly. The notation is precise:
*   Pre-trained weights ($\theta_{\text{pre}}$) and task experts ($\theta_k$).
*   Extraction of task vectors ($\tau^l_k = \theta^l_k - \theta^l_{\text{pre}}$).
*   The layer-wise merging formulation ($\theta^l_{\text{merged}} = \theta^l_{\text{pre}} + \sum_k \lambda^l_k \tau^l_k$).
*   The exact mathematical formulations for the three diagnostic treatments (Intra-Task Layer Shuffling, Task-Wise Spatial Averaging, and Norm-Bounded Perturbation) and the linear Centered Kernel Alignment (CKA) representational analysis.

The addition of the Joint Entropy Minimization Task-Bias section is mathematically elegant, articulating how tasks of varying difficulties (and hence different baseline entropy scales) introduce a multi-task optimization bias.

## 2. Appropriateness of Methods
The proposed diagnostic treatments are highly appropriate and creative tools to stress-test the assumptions of layer-wise merging:
*   **Shuffling** is an elegant control. If layer-specificity is a physical reality, shuffling should break activation flows and degrade performance.
*   **Averaging** is the perfect baseline to test if the high-parameter layer-wise optimization is redundant.
*   **Noise perturbations** are standard for characterizing the optimization landscape's flatness.
*   **Linear CKA** is an excellent, state-of-the-art representational similarity index to measure feature-level alignment in activation space rather than relying purely on top-1 test accuracies.
*   **Optimizer Comparison:** Implementing standard backpropagation-based Adam GD alongside derivative-free 1+1 ES is a highly appropriate method to isolate optimizer confounding.
*   **Empirical Validation Sweeps (Appendix B & C):** Running a regularization sweep ($\beta$) and a calibration sample size sweep ($N_{\text{cal}}$) is the most mathematically rigorous way to prove and resolve the transductive overfitting hypotheses.

## 3. Methodological Strengths & Insights
The paper stands out because of several profound methodological insights that are rigorously formulated and integrated into the text:
1.  **Framing of the Overfitting-Optimizer Paradox:** The authors provide a highly rigorous, unified explanation of why layer-specificity behaves differently under zero-order and first-order optimization. By recognizing that both suffer from transductive overfitting to the small calibration split, they avoid the simplistic conclusion that one optimizer is "better." Rather, they show how optimizer capabilities shape the form that overfitting takes (random walk noise vs. delicate overfit configurations).
2.  **Uncovering the SVHN Rescue vs. CIFAR-10 Collapse Trade-off:** The authors demonstrate excellent academic maturity by looking beyond average accuracies. They point out that spatial averaging rescues SVHN by breaking the joint entropy task-bias (which naturally sacrifices SVHN), but collapses CIFAR-10 because layer-specificity is physically critical for highly complex, non-linear domains.
3.  **Acknowledging and Explaining CKA-Accuracy Decoupling:** Instead of overinterpreting a small CKA difference, the authors identify a critical scientific discrepancy: spatial averaging leads to slightly higher CKA representational similarity but a 10.35% drop in downstream classification performance. They provide a sound mathematical and representational explanation for this decoupling, serving as a valuable warning for future researchers.
4.  **Mathematically Formulating and Validating Coefficient Regularization:** Rather than just identifying the problem, the authors formulate a concrete, regularized loss function incorporating a task-proximity penalty to stabilize test-time model merging and validate its success empirically.

## 4. Minor Constructive Suggestions for Methodology

To further elevate the methodological depth of the paper for future work, the following minor constructive suggestions are offered:

*   **Study of Optimizer Regularization (Weight Decay):** While the authors successfully introduce and evaluate the proximity penalty, standard optimizer regularization (e.g., AdamW weight decay) on the merging coefficients represents a valuable comparison. Exploring whether weight decay alone can naturally mitigate transductive overfitting would be a useful addition.
*   **Weighted/Temperature-Scaled Entropy:** In Section 4.5, the authors identify the joint entropy minimization task-bias, where the optimizer sacrifices the complex, high-entropy task (SVHN) to minimize total loss. They recommend using weighted or temperature-scaled entropy formulations. Proposing or testing a simple temperature-scaled entropy formula (e.g., scaling each task's entropy by its baseline validation entropy) would be an excellent methodological contribution.
*   **Expansion to Autoregressive Language Models:** While the paper focuses on CLIP vision backbones, extending the proposed diagnostic treatments to large-scale autoregressive language models (e.g., 7B+ parameters) would represent a landmark achievement, confirming whether these transductive overfitting and spatial-averaging dynamics generalize to text generation and instruction-tuning tasks.
