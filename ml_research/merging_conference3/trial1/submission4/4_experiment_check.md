# Experiment Check: FluidMerge

## 1. Experimental Strengths
The empirical evaluation of FluidMerge is exceptionally thorough, transparent, and rigorous, with several key highlights:
- **Standardized Synergy-Refinement Protocol:** Initializing all test-time adaptation baselines starting from the Task Arithmetic weight-space average ($\theta_{\text{TA}}$) ensures a mathematically fair comparison under identical, high-performing conditions.
- **Robust Control Ablations:**
  1. **Static TA + Head-Only Tuning:** By showing that freezing the encoder and only tuning the classification heads yields **58.12%** average accuracy (within 1.22% of the full-encoder FluidMerge's **59.34%**), the authors isolate the precise benefit of full-encoder parameter-space advection-diffusion.
  2. **L2 Weight Anchoring (at TA):** By comparing to a standard $L_2$ weight anchoring baseline (**58.48%**), they demonstrate the functional benefit of coordinate-wise, function-sensitive Fisher viscosity (**59.34%**) over simple Euclidean weight-decay.
- **Exceptional Statistical Rigor:** Performing paired two-tailed t-tests and reporting exact, highly significant p-values (e.g., $p = 8.0 \times 10^{-6}$ vs. static TA, $p = 1.0 \times 10^{-4}$ vs. $L_2$ anchoring) is exemplary and rare in this literature.
- **Computational Complexity Transparency:** Table 3 provides complete wall-clock and GPU memory footprint profiling. The authors are commendably honest, explicitly identifying that full-encoder tuning is highly compute-intensive (20.5 minutes vs 0 seconds for Task Arithmetic) and stating that the method acts primarily as an upper-bound research tool rather than a low-latency edge deployment solution.
- **Diverse Evaluation:** Using a standard `ViT-B-32` backbone across 8 diverse image classification tasks (SUN397, Stanford Cars, RESISC45, EuroSAT, SVHN, GTSRB, MNIST, DTD) provides a comprehensive benchmark.

---

## 2. Experimental Weaknesses and Missing Baselines

### A. Missing Baselines and Static Merging Context
- **Task Surgery Baseline:** The authors successfully updated Table 1 to include the *Task Surgery (at TA)* baseline (yielding **58.23%** average accuracy and **8.85%** ECE). This completes the Synergy-Refinement Protocol comparison.
- **Static Merging Baselines:** The authors also discuss *Ties-Merging* (**57.12%** accuracy) and *OrthoMerge* (**57.45%** accuracy) in Section 4.2. Adding these baselines to Table 1 would make the comparison visually unified and even more robust.

### B. OOD Teacher Guidance Lack of Empirical Validation
In Section 3.2, the authors propose a confidence-based entropy thresholding mechanism to filter out noisy predictions from OOD teachers:
$$\tilde{P}_k^{\text{ft}} = P_k^{\text{ft}} \cdot \mathbb{I}\left(\mathcal{H}(P_k^{\text{ft}}) \le \tau\right)$$
- However, they **do not specify any value** for the hyperparameter $\tau$ in Section 4 (Experimental Setup).
- There is **no ablation study** showing whether this filtering was active, what percentage of teacher predictions were filtered, or how varying $\tau$ affects the final accuracy and calibration.
- If no filtering was active, the student was trained under severe OOD teacher noise (as discussed in the Soundness report), which represents a significant gap between the theoretical methodology and experimental validation.

### C. Evaluation Scope of OPT-125M (Appendix A)
The authors present a fully realized empirical evaluation of LoRA-FluidMerge on the `OPT-125M` language model in Appendix A, fine-tuned on Medical and Python corpora. 
- While this is a valuable addition, the experiment is highly limited in scope, evaluating only **two distinct tasks / experts**.
- Given that the main ViT evaluation scales to $K=8$ tasks, evaluating LoRA-FluidMerge on $K \ge 5$ LLM tasks would provide a much more robust verification of its continuous-time parameter trajectory in low-rank subspaces.
- Additionally, they only compare to a single static Task Arithmetic baseline in Table 4, omitting competitive LLM merging methods like Ties-Merging, DARE, or AdaMerging.
