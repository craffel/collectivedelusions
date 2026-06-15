# Experimental and Baseline Evaluation of FluidMerge

## 1. Experimental Setup and Baselines
*   **Evaluation Model:** Vision Transformer backbone (**ViT-B-32** with ~86M parameters).
*   **Datasets:** Eight diverse image classification datasets (SUN397, Stanford Cars, RESISC45, EuroSAT, SVHN, GTSRB, MNIST, and DTD). This represents a standard and sufficiently diverse benchmark suite for multi-task model merging and test-time adaptation.
*   **Baselines:** The paper evaluates against appropriate, modern, state-of-the-art baselines:
    *   **AdaMerging:** Optimizes layer-wise scaling coefficients via entropy minimization.
    *   **SyMerge:** Adapts single task-specific layers using teacher-guided self-labeling.
    *   **Task Surgery:** Employs L1-based feature alignment.
    *   **Ties-Merging & OrthoMerge:** Highly competitive static model merging methods.
    *   **L2 Weight Anchoring:** Full-encoder tuning with Euclidean weight decay.
    *   **Static TA + Head Tuning:** Keeping the encoder frozen and updating only classification heads.
*   **Standardization:** All methods are evaluated under identical test-time adaptation (TTA) batch settings ($1000$ unlabeled images, mini-batch size of $32$).

---

## 2. Integrity and Support of Claims
The empirical results in Table 1 strongly support the paper's primary claims:
*   **Boundary Condition Impact:** Starting from the raw pretrained base weights ($\theta_0$) leads to a near-complete collapse (~6.23% accuracy), validating the "domain shift barrier." Starting from the Task Arithmetic boundary condition ($\theta(0) = \theta_{\text{TA}}$) completely resolves this collapse, placing all methods within high-performing basins (>57% accuracy).
*   **FluidMerge Superiority:** FluidMerge with Fisher-Information-based Viscosity achieves the highest multi-task accuracy (**59.34%**), outperforming static Task Arithmetic (**57.74%**), AdaMerging (**58.04%**), Task Surgery (**58.23%**), SyMerge (**58.42%**), and standard L2 Anchoring (**58.48%**).
*   **Calibration Preservation:** Fisher Viscosity successfully stabilizes the Expected Calibration Error, keeping it at **7.18%**, whereas unregularized self-training causes calibration collapse (ECE > 90%).
*   **Statistical Significance:** The paired two-tailed t-tests show that FluidMerge's improvements are highly robust and statistically significant ($p < 0.0001$ vs. static TA, $p < 0.001$ vs. L2 Anchoring).

---

## 3. Critical Critique of Experimental Claims (Theorist Perspective)
While the results are statistically valid and well-analyzed, a deeper evaluation reveals significant practical and conceptual limitations.

### A. Modest Accuracy Gains vs. Extreme Computational Costs
The paper's primary contribution is a continuous-time adaptation method that backpropagates gradients through the entire 86M parameter encoder. However, when we evaluate the cost-to-benefit ratio, the practical utility of the method becomes highly questionable:
1.  **Static Task Arithmetic (Static TA):** Yields **57.74%** accuracy at **zero** computational cost, zero GPU memory, and executes instantly ($0$ seconds).
2.  **Static TA + Head-Only Tuning:** Kept completely frozen at $\theta_{\text{TA}}$, optimizing *only* the linear heads. This yields **58.12%** accuracy, requiring almost zero computational overhead, since the classification heads contain a negligible number of parameters (e.g., 512 parameters per head) and converge rapidly.
3.  **FluidMerge (Fisher - Ours):** Yields **59.34%** accuracy, but requires **20.5 minutes** of premium NVIDIA A100 GPU compute, **14.8 GB** of GPU memory overhead, and full-encoder backpropagation through 86M parameters over 100 epochs, including the evaluation of 8 separate teacher expert models to compute soft labels.
4.  **The Marginal Delta:** The absolute accuracy gain of FluidMerge over static Task Arithmetic is only **1.60%**, and its gain over the highly efficient Head-Only Tuning control is a meager **1.22%**. 

From an engineering perspective, a 1.22% accuracy improvement at the cost of full-encoder backpropagation and a 20.5-minute delay is a massive computational bottleneck that is highly unlikely to be justified in real-world edge, mobile, or low-latency environments. The paper's method is highly inefficient and relies on massive overparameterized full-encoder optimization to squeeze out a tiny margin.

### B. The "Boundary Stress-Test" is a Strawman Evaluation
Section 4.3 presents a "Boundary Stress-Test" starting from the raw, unadapted pretrained base encoder weights ($\theta_0$).
1.  **Guaranteed Failure:** Baselines like AdaMerging and SyMerge are structurally and mathematically designed *specifically* to combine and adapt already fine-tuned task-expert representations. Forcing them to initialize at the raw, unadapted base model $\theta_0$ is completely outside their native design space.
2.  **Strawman Nature:** Forcing these baselines into a setup where they are guaranteed to fail (~4.37% and ~5.56% accuracy), and presenting this as a "rigorous diagnostic stress-test," is a strawman comparison. While it serves to highlight the importance of the Task Arithmetic initial boundary condition, it does not provide a realistic or fair comparative evaluation of prior merging techniques.

### C. Low-Rank Parameter Fluids (LoRA-FluidMerge) on LLMs is Under-Evaluated
In Appendix A, the paper introduces a "LoRA-FluidMerge" extension to address the computational budget of full-encoder tuning, presenting results on OPT-125M (Table 2):
1.  **Tiny Scale:** The experiment is conducted on a very small autoregressive model (OPT-125M) and evaluated on only two highly distinct domains (Medical and Python, $K=2$).
2.  **Insignificant Performance Gap:** LoRA-FluidMerge achieves a validation cross-entropy loss of **3.0140**, whereas static Task Arithmetic achieves **3.0341**. This represents an absolute average loss reduction of only **0.0201**.
3.  **Lack of Downstream Evaluation:** The paper fails to evaluate whether this tiny reduction in validation loss translates to any meaningful difference in downstream text generation quality (e.g., accuracy on medical QA or code generation benchmarks). Evaluating on only two tasks is insufficient to draw broad conclusions about the viability of weight-fluid simulations on large language models.
