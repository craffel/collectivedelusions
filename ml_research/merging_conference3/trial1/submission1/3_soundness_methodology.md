# 3. Soundness and Methodology

A rigorous review of the mathematical formulation, implementation, and baseline design of **QP-Merge** reveals that the authors have successfully resolved several critical flaws present in earlier drafts, and the paper is technically very solid. However, a few remaining caveats regarding mathematical equivalence, generalization, and a critical reporting script bug should be noted.

## 1. Resolved Issues from Previous Drafts (Acknowledge Improvement)
The authors are commended for making significant improvements to the technical soundness and presentation of the paper:
- **Consistent Formulation (Equation 11):** The calibration objective is now correctly defined as an end-to-end Representation-Level Alignment loss:
  $$\mathcal{L} = \mathbb{E}_{X} \left[ \| f_{\text{FP32}}(X) - f_{\text{hybrid}}(X; D, \lambda) \|_2^2 \right]$$
  This perfectly matches the implementation in `task_vectors/qp_merge.py`, which computes the loss on the final output embeddings.
- **Valid Matrix Notation (Equation 12):** The column-wise scaling matrix $D_l \in \mathbb{R}^{d_{\text{in}} \times d_{\text{in}}}$ is now correctly multiplied on the *right* side of the dense task sum:
  $$W_{l, \text{hybrid}}(D_l, \lambda) = Q_b\left(W_{l, \text{base}} + \left( \sum_{t=1}^T \lambda_t \Delta W_{t, l, \text{dense}} \right) D_l \right) + \sum_{t=1}^T \lambda_t \Delta W_{t, l, \text{outlier}}$$
  This resolves the previous dimension clash and aligns with column-wise broadcasting in PyTorch.
- **Eliminated Confounding Variable:** The authors have added the crucial **FP32 Merged Bound (Optimized)** baseline (95.12% accuracy). By comparing QP-Merge INT8 (95.14%) to this baseline instead of the uniform FP32 baseline, they correctly show that the performance gain is due to the task coefficient optimization on the target domain, rather than an artificial regularization effect from low-bit quantization.

---

## 2. Technical Gaps and Soundness Limitations

### A. Lack of Mathematical Equivalence in Weight Scaling
Traditional activation-weight co-scaling methods (such as **SmoothQuant**) maintain mathematical equivalence by applying an inverse scaling factor to activations at runtime:
$$Y = (X D^{-1}) (D W) = X W$$
This guarantees that before quantization is applied, the network's output is mathematically identical to the original floating-point model. 

In contrast, QP-Merge applies $D_l$ only to the dense task vector updates *without* applying any inverse scaling to the input activations $X$ during inference:
$$W_{\text{scaled\_base}} = W_{\text{base}} + \left(\sum \lambda_t \Delta W_{t, \text{dense}}\right) D_l$$
Because the scaling is applied permanently to the weights without a corresponding inverse scale on the activations, QP-Merge does not maintain mathematical equivalence. While the optimization loop successfully minimizes the embedding discrepancy on the 128-sample calibration set, this lack of equivalence poses a significant risk of **representation drift** and overfitting.
The authors have added a dedicated paragraph "Justification for Non-Equivalence in Weight Scaling" in Section 3.3 to explain why this is practically necessary (different tasks have conflicting activation scales, making unified equivalent scaling extremely difficult). This is an excellent addition, but the lack of mathematical equivalence remains an approximation that should be carefully validated.

### B. Risk of Overfitting to Tiny Calibration Set & Lack of OOD Evaluation
The calibration set is extremely small, consisting of only $M=128$ unlabeled samples (64 from MNIST, 64 from SVHN). 
Because QP-Merge permanently alters the network's mathematical mapping without maintaining equivalence, there is a risk of overfitting the weights to this microscopic calibration set. 
While Table 4 shows a sensitivity sweep over $M$ that demonstrates reasonable stability (average accuracy remains around 93.8% - 94.5% across sizes), the authors do not evaluate the model's performance on slightly out-of-distribution (OOD) test sets (e.g., MNIST-C or SVHN-C) to verify whether the optimized scaling parameters generalize beyond the exact in-distribution test sets.

### C. Verification of Resolved Reporting Bug
We have verified that a potential reporting bug in the advanced evaluation script (`task_vectors/qp_merge_advanced.py`) is **fully resolved** in the current codebase. In previous versions, there was a risk that the `res_M` variable from the calibration size sweep (which runs in INT4) would leak and overwrite the reported INT8 results, leading to corrupted text reports (showing ~94.59% instead of ~95.13%). 

Our audit of the current codebase confirms that:
- The script correctly saves the calibration sweep results in a dictionary `m_sweep_results` and evaluates the sweep with the loop variable `res_sweep_M`.
- The baseline results are correctly printed and written to file using `res_2026` (the result of the 2026 seed run).
- Consequently, both `qp_merge_advanced_report_bits4.txt` and `qp_merge_advanced_report_bits8.txt` contain accurate, pristine reports.

This resolution ensures high codebase quality, reliability, and exact reproducibility.

### D. Inconsistencies in Reported Results in the Manuscript
Likely due to the aforementioned reporting script bug or manual compilation errors, there are noticeable discrepancies in reported accuracy values across different sections of the paper:
- **Table 1 vs. Table 2 / Intro / Conclusion (INT8):**
  - **Table 1 (Primary Results):** Reports `QP-Merge (Ours) INT8` as **95.14% $\pm$ 0.03%** (3-seed average).
  - **Abstract & Introduction:** Reports `QP-Merge INT8` as achieving **95.08%** (+0.15% gain).
  - **Table 2 (Ablation Results):** Reports `Full QP-Merge INT8` as **95.08%**.
  - **Section 5 (Conclusion):** Refers to `QP-Merge INT8` as achieving **95.08%** (within 0.04% of the FP32 optimized baseline).
- **Table 1 vs. Table 2 (INT4):**
  - **Table 1:** Reports `QP-Merge (Ours) INT4` as **94.70% $\pm$ 0.13%** (3-seed average).
  - **Table 2 (Ablation Results):** Reports `Full QP-Merge INT4` as **94.71%**.
The authors should resolve these inconsistencies. If Table 1 reports 3-seed averages and the other sections report a single representative seed (seed 2026, which indeed gets 95.08% in INT8 and 94.71% in INT4), the authors must explicitly state this in the text and table captions to avoid giving the impression of arbitrary numbers or sloppy reporting.
