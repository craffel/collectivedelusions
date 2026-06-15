# Evaluation Part 4: Experimental Evaluation and Check

## Critical Evaluation of the Experimental Setup
The experimental evaluation is carried out on a standard 8-task image classification benchmark using CLIP ViT-B/32. The choice of datasets is standard and diverse.

## Baselines and Comparisons
The paper compares against:
- Static baselines: Task Arithmetic, TIES-Merging.
- Adaptive baselines: AdaMerging, SyMerge, FoldMerge.
- Ablations: Unconstrained scaling, BPAM-Restricted, split-test calibration with/without proximity penalty, extreme low-data (5 samples per class) calibration, and hyperparameter sensitivity of $\beta$.

The comparisons are scientifically honest; the authors do not hide that their method is significantly outperformed by SOTA adaptive methods (e.g., FoldMerge's 83.56% vs BPAM-Static's 69.21%) and even by static TIES-Merging (72.90%). 

## Strengths of the Experimental Setup
1. **Symmetric Evaluation (Part A vs Part B):** Evaluating both frozen classification heads and active head adaptation is highly rigorous. It isolates whether the performance gains in test-time merging come from weight-space blending or decision-boundary adaptation.
2. **OOD Generalization (Split-Test):** Running a split-test (20% calibration, 80% unseen test) is a strong methodology to check for transductive overfitting.
3. **Extreme Low-Data Stress Test:** Evaluating the model under 5 samples per class is a valuable contribution, demonstrating a scenario where the proposed proximity penalty actually becomes functional and beneficial.
4. **Centered Kernel Alignment (CKA) Analysis:** Providing CKA similarity scores to resolve the "0-weight performance mystery" is highly empirical and convincing.

## Major Weaknesses from an Empiricist Perspective
1. **Lack of Statistical Significance and Variance Reporting:**
   - There are absolutely no error bars, standard deviations, confidence intervals, or indications of multiple random seeds in Table 1, Table 3, or Table 4. 
   - Test-time adaptation is notoriously sensitive to the specific samples in the calibration stream, particularly in the extreme low-data regime (5 samples per class in Table 4). Without reporting the mean and variance across multiple random seeds/splits, it is impossible to know if small differences (such as BPAM-Static beating Task Arithmetic by +0.11% in Table 1, or the minor differences between $\beta$ values in Table 4) are statistically significant or merely noise.
2. **Incomplete Argument on Activation Scale Distortion:**
   - The authors claim that unconstrained scaling leads to "scale distortion or activation collapse" under extreme conditions, which motivates their convex simplex constraint. However, they do not provide any empirical measurement (e.g., weight norms, activation scales, or attention map distributions) to substantiate this claim. They should show plots or statistics comparing the internal activations/weight norms of unconstrained scaling vs. BPAM-Static to prove that scale distortion actually occurs and is prevented by their method.
3. **Baselines in Part B:**
   - In Table 1 Part B, the authors compare BPAM-Full to "TIES-Merging + Head Tuning" and "AdaMerging + Head Tuning". However, they do not compare against "SyMerge + Head Tuning" or "FoldMerge + Head Tuning" in terms of training parameter efficiency and runtime. They list "SyMerge + Head Tuning" and "FoldMerge + Head Tuning" but do not provide the exact parameter footprints for SyMerge's adapters, which is left as "High". For completeness, they should specify the exact parameter counts.
