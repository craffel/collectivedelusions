# Impact, Presentation, and Writing Quality

This file reviews the paper's significance, presentation, and clarity, and provides constructive, actionable suggestions for improvement.

---

## 1. Significance and Impact of the Contribution
Dynamic model ensembling and post-hoc model merging are highly active areas of machine learning research, particularly in edge-computing and multi-tenant serving where maintaining separate large models for each task is computationally prohibitive.

### Positive Impact
* **A Breath of Fresh Air:** The paper's philosophical stance is highly significant. By actively applying Occam's razor to stateful routing, the paper serves as a vital cautionary tale against "metaphor creep" in machine learning. It demonstrates that standard mathematical operators (like discrete EMAs) are often highly sufficient and computationally superior to convoluted physical/chemical analogies.
* **Trajectory Stability:** The paper establishes **routing jitter** as a crucial metric for dynamic model serving. Slashing jitter by multiple orders of magnitude (up to 195$\times$ lower than SABLE) while maintaining competitive accuracy is highly significant for serving stability and preventing cascading representational drift.

### Limitations of Impact
* **Synthetic Evaluation:** Because the empirical validation is situated entirely within the synthetic **Analytical Coordinate Sandbox (ICS)**, the paper's immediate practical impact on real-world systems is limited. Machine learning practitioners are unlikely to adopt Momentum-Merge until it is demonstrated to work on massive, real-world LLMs (e.g., LLaMA, Mistral) on diverse text streams.

---

## 2. Presentation, Structure, and Writing Quality
* **Excellent Writing:** The paper is exceptionally well-structured, clearly written, and academically rigorous. The transition from the problem formulation to the mathematical deconstruction of ChemMerge is highly compelling and elegant.
* **Clear Notation:** The mathematical notation is clean, elegant, and consistent throughout. The authors resolved previous notation collisions (e.g., renaming the LoRA scale parameter to $s_{\text{LoRA}}$ to avoid collision with ensembling weights $\alpha_k^{(l)}$).
* **Theorem and Proof:** The inclusion of Theorem 3.1 and its proof is a highlight of the paper. It is easy to follow and mathematically sound.
* **Aesthetic Layout and Figures:** The figures are of high quality and clearly illustrate the performance-stability trade-off and the Pareto sweep. Tables are beautifully typeset and wrapped in column-width scaling environments to prevent column overlaps or margin violations.

---

## 3. Actionable and Constructive Suggestions for Improvement

To achieve the standard of an excellent, flawless publication, the authors should address the following areas:

### A. Synchronize Minor Reference Mismatches in Baseline text and Appendices
The authors should resolve the minor discrepancies between the Table 1 metrics and the text in Section 4.2 and Appendix C/E:
* In Section 4.2 (Baselines), replace the verbal claim of matching joint accuracy within 0.05% (`76.15% vs. 76.20%`) with the actual Table 1 synchronized metrics (`74.85% vs. 74.71%`).
* In Appendix C and E, clearly state that the sweeps are uncalibrated and unanchored, explaining why they achieve slightly different baseline metrics (e.g., 76.15%) compared to the 10-seed perfectly synchronized results (74.85%) reported in Table 1.

### B. Frame the Accuracy-Stability Trade-off Transparently
The paper should be more explicit in the Abstract and Intro about the **accuracy cost of temporal smoothing**:
* Clearly state that stateless routing with layer centroids (`SABLE + Layer Centroids`) achieves the highest overall joint accuracy (**77.24%**).
* Frame adding momentum (EMA) as a low-pass filter that dampens high-frequency oscillations (reducing jitter by 76$\times$) but trades off a small fraction (2.26% absolute) of classification accuracy due to routing over-smoothing. This frames the performance curves as a genuine Pareto trade-off.

### C. Address the Scarce-Data Recurrence Trapping Vulnerability
The authors should include a brief discussion of the **Recurrence Trapping** vulnerability of Momentum-Merge under scarce-data regimes ($|\mathcal{C}_k| \le 16$), as detailed in Appendix E. 
* Propose potential mitigations, such as dynamically scaling down the momentum parameter $\beta$ in early layers if the calibration set size is small, or falling back to a uniform initial boundary condition.
