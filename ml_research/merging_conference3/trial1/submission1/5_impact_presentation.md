# 5. Impact and Presentation

An assessment of the presentation, clarity, and broader impact of **QP-Merge** highlights positive structural aspects and identifies key edge deployment trade-offs.

## 1. Presentation and Clarity
- **Strengths:** 
  - The paper is highly structured and clearly written. The narrative flow from introducing the merging-quantization gap to the methodology is engaging and easy to follow.
  - The motivation (the dual challenge of heavy-tailed weight outliers and activation scale mismatches) is clearly articulated.
  - Figures and tables are well-placed, and the description of the experimental findings in Section 4 is clear and thorough.
  - The equations and mathematical notation are mathematically valid and match the implementation details in the code.
- **Weaknesses:**
  - **Single vs. Joint Calibration Ambiguity:** Figure 1's caption and introductory text imply that calibration is local or layer-wise, but Section 3.3 shows that the parameters are optimized jointly using end-to-end backpropagation. The authors should clarify this distinction to avoid confusing expert readers.
  - **Text-Table Inconsistencies (95.08% vs. 95.14%):** There are multiple locations in the text (Abstract, Intro, Table 2, Table 3, Conclusion) where INT8 performance is reported as `95.08%` (which corresponds to a single representative run with seed 2026), but Table 1 and Section 4.3 report the 3-seed average as `95.14%`. The authors must synchronize these figures or clearly specify which numbers are 3-seed averages versus single representative runs.

## 2. Broader Impact and Deployment Limitations
The authors are highly commended for adding Section 5.1 ("Limitations and Future Work"), which provides an exceptionally honest, thorough, and scientifically mature discussion of the practical edge bottlenecks. This section directly addresses the key real-world constraints:

### A. Edge Hardware Compatibility Mismatch
As the authors point out, many low-power IoT microcontrollers, legacy edge NPUs, and hardware accelerators (such as Google Edge TPU, Apple Neural Engine) only support **homogeneous fixed-point execution** (pure INT8 or INT4). They lack floating-point execution units or the ability to run dynamic, hybrid representations.
Since QP-Merge requires executing a sparse high-precision floating-point (FP16) path alongside the dense path at every single layer, it is **completely incompatible with many target edge platforms**.
- *Reviewer Recommendation:* The authors should moderate the bold claims in the introduction and abstract (which present QP-Merge as immediately deployable on "edge, mobile, and IoT devices") to align with this hardware compatibility reality.

### B. Outlier Density Scaling with Number of Tasks
When merging $T$ tasks, if the location of task-specific outliers is completely disjoint, the union of outliers from multiple tasks will grow linearly. For $T=10$ tasks, the sparse outlier density could scale to $\approx 5.0\%$, which would significantly degrade SpMM execution efficiency and increase storage overhead, undermining the VRAM savings.
The authors' proposal of a global sparsity constraint as future work is highly appropriate and shows strong scientific foresight.

### C. Dependency on Multi-Domain Calibration Data
The calibration step (QE-Calib) requires a balanced set of unlabeled samples from *all* merged domains (e.g., 64 MNIST and 64 SVHN).
In real-world edge serving, a device might only have access to data from a single domain at a time, or domain data might be proprietary and unavailable during edge calibration.
The paper does not evaluate how QP-Merge behaves under **imbalanced or single-domain calibration data**:
- If we calibrate only on SVHN data, does it destroy MNIST accuracy?
- What is the sensitivity to distribution shift in the calibration set?
This is a remaining practical impact question that should be discussed or evaluated.
