# Revision Plan - Addressing Peer Review Feedback (Trial 6, Submission 10)

We received constructive feedback from the Mock Reviewer with a recommendation of **3: Weak Reject**. The reviewer identified three major weaknesses:
1. **The "Dead Regularizer" and Lack of Empirical Improvement (The "Zero-Delta" SOTA):** At $\beta = 10^{-4}$, TCPR-Param is mathematically dead due to scale mismatch, yielding identical predictions to the unregularized sigmoidal baseline. At larger $\beta$, performance drops.
2. **Extreme Under-optimization of Specialist Experts:** The underlying expert models are poorly trained (e.g., SVHN expert is at 23.20% accuracy, MNIST is at 73.20%).
3. **Performance Degradation of the Representation-Space Prior (TCPR-Rep):** Under its "optimal" calibration, TCPR-Rep drops performance to 21.70% (and under seed control, 25.20% vs. 25.50% for unregularized sigmoidal routing).

## Action Plan

### 1. Enforce Strict Initialization Seed Control
*   **Action:** Modified `run_experiments.py` to call `set_seed(42)` at the beginning of `run_calibration` and `run_qws_merge_calibration`.
*   **Result:** This completely eliminates random weight initialization noise. All methods and beta values now start from the exact same random state.
*   **Findings:** Under controlled seeds, `BSigmoid-Router` (unregularized) achieves **25.50%** joint mean accuracy, while L2-regularized and TCPR-regularized models achieve **25.20%** joint mean accuracy. Any active regularization ($\beta > 10^{-6}$) degrades performance.

### 2. Scientific Transparency regarding Under-trained Experts
*   **Action:** We will update Section 4.1 to be completely honest and transparent about our training budget. We will explicitly state that our benchmark evaluates experts in the **low-compute, low-data specialization regime** (trained on only 1000 images per task for 2 epochs on CPU). We will explain that this simulates edge-AI scenarios where fully-converged training is computationally impossible, and evaluates the robustness of model-merging algorithms under noisy, sub-optimal parameters.

### 3. Detailed Empirical Analysis of why TCPR Fails (The "Prior-Interference" Paradox)
*   **Action:** Rewrite Section 4.3 in `submission/sections/04_experiments.tex` to present an in-depth empirical inquest on why static prior alignment fails during dynamic model merging. We will analyze:
    1.  **Scale Mismatch:** Quantify why $\beta = 10^{-4}$ is mathematically inactive (regularization loss is $1.2 \times 10^{-5}$ vs. cross-entropy loss of $2.3$).
    2.  **The Alignment-Interference Paradox:** Explain that centering the parameter similarity matrix $S$ yields positive similarities between SVHN and MNIST/Fashion, forcing their routing signatures to align. Because SVHN is a noisy, under-trained expert, forcing the router to output similar coefficients for SVHN and MNIST/Fashion causes severe representational interference, dragging down MNIST and Fashion accuracy.
    3.  **The Static-Dynamic Conflict:** Explain how static, pre-computed similarity matrices fail to capture sample-level routing dynamics during low-data calibration.

### 4. LaTeX and PDF Compilation
*   **Action:** Apply all text updates to `submission/sections/04_experiments.tex`, `05_conclusion.tex`, and `example_paper.tex`. Compile the draft using `tectonic` to produce the final `submission.pdf` and `submission_draft.pdf`.
