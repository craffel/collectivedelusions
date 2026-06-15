# Soundness and Methodology Evaluation

## Mathematical and Conceptual Clarity
The mathematical formulation of PG-Merge is exceptionally clear, rigorous, and easy to follow:
- **Equations 1 & 2** clearly establish the task-vector paradigm and how layer-wise merging coefficients parameterize the merged weights.
- **Equation 3** defines the standard Shannon prediction entropy objective used for unsupervised test-time adaptation (TTA).
- **Equations 6 to 12** step-by-step detail the sorting, masking, and projection operations of PG-Merge, culminating in a mathematically self-contained algorithm.
- The distinction between raw gradients and masked/pruned gradients is precise, and the post-update projection (Equation 12) is well-justified as a mechanism to counter Adam's momentum leakage.

---

## Appropriateness of Methods
- **Entropy Minimization:** This is the standard, well-validated objective for unsupervised TTA (Tent, AdaMerging). It is highly appropriate for online settings where ground-truth labels are completely unavailable.
- **Sparsity as Regularization:** Limiting optimization degrees of freedom is a classic, theoretically sound way to restrict model capacity and prevent overfitting. In the low-sample TTA regime (where the model only has $64$ unlabeled calibration images), optimizing all $56$ layer-wise coefficients is highly prone to fitting local noise. Constraining updates to a dynamic, sparse subset (e.g., $5\%$ or $15\%$) of high-sensitivity coordinates is highly appropriate and acts as a dynamic low-pass filter on weight adjustments.

---

## Critical Technical Flaws and Empirical Gaps

While the methodology is sound, there are several key empirical omissions and architectural ambiguities that must be addressed to ensure complete scientific rigor:

### 1. The "SGD Compatibility" Claim lacks Empirical Support (Appendix A)
In Appendix A, the authors write a compelling theoretical argument advocating for pairing PG-Merge with standard Stochastic Gradient Descent (SGD) without momentum. They state that SGD completely eliminates the need for the post-update parameter projection (Equation 12) and resolves the "optimizer state mismatch" (momentum decay of inactive parameters in Adam). 

**The Flaw:** Despite devoting an entire appendix section to praising SGD as the "mathematically self-consistent" and "minimalist" ideal optimizer for PG-Merge, **the paper contains absolutely zero empirical results comparing SGD against Adam.**
All experiments in Section 4.1 use the Adam optimizer. To support the strong claims made in Appendix A, the authors *must* provide parallel experiments demonstrating:
- Whether PG-Merge + SGD achieves comparable or superior performance to PG-Merge + Adam.
- Whether SGD actually stabilizes the adaptation trajectory further or prevents representation decay as claimed.
Without this empirical verification, Appendix A remains purely speculative and weakens the paper's overall soundness.

### 2. Missing Initialization Details for Merging Coefficients ($\alpha$)
For any active test-time optimization method, the initialization of the parameters being optimized is of paramount importance. 
- In Section 4.1, the authors state: *"Uniform Merging (Task Arithmetic): Standard static parameter addition with uniform coefficients ($\alpha = 0.3$)."*
- However, they **never explicitly state the initial value of the merging coefficients $\alpha_{k, l}$** for AdaMerging, RegCalMerge, PolyMerge, and PG-Merge before test-time adaptation begins.
Are they initialized to uniform values (e.g., all $0.3$ or $1/K = 0.25$)? Or are they initialized to zero? Because test-time adaptation only runs for $100$ steps on a very small set of samples, the initial position in the parameter space dictates the optimization trajectory and final convergence. This is a critical omission for reproducibility and clarity.

### 3. Ambiguity in Classification Head Handling for Diverse Tasks
The paper evaluates model merging on a single `vit_tiny` backbone across four very different classification datasets: MNIST (10 digit classes), FashionMNIST (10 clothing classes), CIFAR-10 (10 object classes), and SVHN (10 digit classes). 
- Because these tasks have completely different output spaces, a single shared classification head is architecturally impossible unless the categories are mapped to a joint vocabulary, which is not mentioned.
- Standard practice in multi-task model merging is to have **task-specific classification heads** that are kept separate and routed dynamically based on a known task ID, while merging only the shared backbone parameters (the ViT encoder layers).
- The authors must explicitly clarify how the classification heads are handled. Are they kept separate? If so, are they adapted, or are they kept frozen while only the backbone merging coefficients $\alpha$ are optimized? This is a major architectural detail that is currently omitted.

---

## Reproducibility Assessment
The paper rates **Good** for reproducibility:
- **Strengths:** It specifies the exact backbone model (`vit_tiny_patch16_224`), the sample sizes (1,024 for expert fine-tuning, 64 for TTA calibration, 512 for evaluation), the optimization steps (100 steps for TTA), the learning rates ($10^{-3}$), and all key mathematical operations.
- **Weaknesses:** Reproducibility is hindered by the missing details on (1) coefficient initialization and (2) classification head routing/management. Clarifying these two aspects would elevate reproducibility to "Excellent."
