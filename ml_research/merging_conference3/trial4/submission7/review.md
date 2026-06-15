# Peer Review: SuiteMerge: Deconstructing the Task Suite Bias in Model Merging

## Meta-Information
*   **Overall Recommendation:** 5: Accept (Technically solid paper that advances the model-merging literature with a highly rigorous and corrective methodology, outstanding empirical verification, and clear actionable directions, though with minor limitations in scale)
*   **Soundness Rating:** Excellent
*   **Presentation Rating:** Excellent
*   **Significance Rating:** Excellent
*   **Originality Rating:** Excellent

---

## 1. Summary of the Submission
This paper presents a timely, independent, and rigorous methodological audit of the adaptive model-merging paradigm. Modern adaptive merging frameworks (such as AdaMerging and PolyMerge) claim state-of-the-art multi-task performance by dynamically optimizing layer-wise merging coefficients at test time (Test-Time Adaptation, or TTA) via unsupervised prediction entropy minimization. 

The authors critically audit this paradigm and expose a severe, previously unreported **Task Suite Bias**: standard evaluation protocols in prior publications rely on a single, highly arbitrary combination of visual classification datasets (MNIST, FashionMNIST, CIFAR-10, SVHN) that spans highly heterogeneous domains, masking critical limitations of online TTA. By systematically partitioning this task pool into five distinct sub-suites (**SuiteMerge**) based on domain distance and representational conflict, the authors demonstrate that unconstrained online TTA (AdaMerging) overfits to local transductive stream noise, resulting in performance degradation under representational conflict and catastrophic parameter collapse in physical weight-space deployments.

As a highly robust, zero-test-time-compute, zero-latency alternative, the authors propose **Offline Few-Shot Validation Tuning (OFS-Tune)**. OFS-Tune optimizes continuous low-degree polynomial trajectories (linear $d=1$ and quadratic $d=2$ configurations) offline on a tiny, stratified labeled validation set ($M=10$ samples per task) using Nelder-Mead simplex search. OFS-Tune acts as a powerful analytical low-pass filter, matching or exceeding online TTA methods across all suites in simulation, and outperforming online PolyMerge and AdaMerging by up to **3.70%** and **4.20%** respectively in physical deep network weight-space validation. The authors also introduce localized trajectory constraints (Piecewise Splines, Block-wise Parameter Sharing) to capture non-smooth sensitivity profiles in Transformers, and provide an actionable roadmap to scale OFS-Tune to billion-parameter foundation models.

---

## 2. Key Strengths of the Submission

### A. Exceptional Methodological Rigor and Soundness
*   **Calibrated Model II Landscape Simulator:** The simulated sensitivity landscape is not a simple toy function; it is a highly rigorous coupled non-convex model calibrated directly against empirical Vision Transformer (ViT-B/32) classification statistics. It mathematically models layer-wise sensitivity, quadratic/quartic curvatures, and pairwise task representational conflicts.
*   **Exemplary Optimization and Budget Controls:** The authors address potential optimizer asymmetries with outstanding rigor. They construct symmetrical control baselines—proving that restricting OFS-Tune to limited first-order Adam has zero performance cost, whereas allowing online AdaMerging to fully converge via second-order L-BFGS-B actually degrades performance due to deeper overfitting to the misaligned unsupervised entropy surface under stream noise.
*   **Thorough Physical Weight-Space Validation:** The paper goes far beyond simulation by training physical Convolutional Neural Networks (CNNs) and validating weight-space merging across two distinct initialization paradigms: (1) scratch-trained experts in disjoint loss basins (high representation conflict) and (2) pre-trained experts sharing a linearly mode-connected loss basin (low representation conflict). This establishes the physical validity of the paper's core claims.

### B. High Conceptual and Algorithmic Novelty
*   **Dismantling Benchmark Bias:** The paper is the first to identify and systematically analyze the **Task Suite Bias** in the model-merging literature, exposing how evaluating on a single fixed suite masks catastrophic local failures.
*   **The "Privilege Trap" and Multi-Task Routing:** The paper exposes a critical, hidden deployment assumption in standard online TTA evaluations—the need for oracle task-routing labels at inference time—and demonstrates that online TTA collapses when forced to perform joint entropy minimization on mixed streams.
*   **The OFS-Unconstrained Ablation:** By introducing the *OFS-Unconstrained* ablation baseline, the authors successfully isolate the regularizing value of the continuous polynomial trajectory constraint from the effect of having supervised few-shot validation data, proving that validation data alone is insufficient to prevent high-frequency noise overfitting.
*   **Handling Non-Smooth Sensitivity Profiles:** To resolve the potential circularity of assuming global polynomial profiles, the authors introduce and validate Piecewise Splines and Block-wise Parameter Sharing. These formulations successfully capture high-frequency localized sensitivity spikes in actual Transformer architectures (e.g., MHA vs. MLP projections) while preserving low-pass noise filtering.

### C. Exemplary Presentation, Openness, and Scale-up Actionability
*   **Exceptional Writing Clarity:** The paper is beautifully written, highly structured, and easy to follow. It maintains a professional, constructively skeptical, and intellectually honest tone.
*   **Actionable Scaling Roadmap for Large Foundation Models:** Section 5 outlines concrete, easily implementable strategies to scale the offline trajectory-tuning framework to massive billion-parameter LLMs/VLMs under PEFT (using representative subsets, coordinate gradient descent via OFS-Adam, and expert parameter offloading). This makes the work immediately relevant for modern NLP and multi-modal practitioners.
*   **High Statistical Rigor:** All simulated results are evaluated across **30 independent random seeds**, ensuring tight confidence intervals and highly robust findings.
*   **Commitment to Open Science:** The authors explicitly commit to releasing their complete PyTorch code, simulator, checkpoints, and LLM scaling utilities under the permissive **Apache 2.0 open-source license**, guaranteeing reproducibility.

---

## 3. Suggestions for Improvement and Future Work

While the paper is technically solid and highly significant, we raise the following constructive points to further enhance its completeness and polish:

### 1. The Validation Class-Imbalance / Missing Class Risk in Ultra-Few-Shot Calibration ($M=10$)
The paper leverages a tiny labeled validation set containing $M=10$ samples per task for offline tuning. While highly practical, this design introduces a major structural vulnerability when a task contains multiple classes. For a standard 10-class dataset (like CIFAR-10 or SVHN), a random draw of $M=10$ samples means there is exactly 1 sample per class on average. Mathematically, the probability that at least one class is completely omitted from a random uniform draw of size $M=10$ from a 10-class pool is:
$$P(\text{missing class}) = 1 - \frac{10!}{10^{10}} \approx 99.96\%$$
This means that in almost 100% of cases, random few-shot selection will omit at least one class entirely, leading to severe class-imbalance and missing class representations in the supervised loss. For high-conflict tasks, this can cause the offline optimized trajectory to underperform on classes missing during validation. 
*   **Actionable Suggestion:** The authors should discuss the necessity of **stratified sampling** (ensuring exactly 1 sample per class is drawn) rather than naive random sampling for few-shot validation sets. They should also address how the minimum validation budget scales when tasks feature massive label spaces (e.g., CIFAR-100 or ImageNet), where $M \ge 100$ would be required just to cover all classes.

### 2. Inference-Time Task Routing in Multi-Head Deployments
In Section 1 and Section 3, the authors state that OFS-Tune "naturally bypasses this task-routing dilemma entirely." To maintain absolute presentation accuracy and intellectual honesty, this statement should be qualified. While OFS-Tune completely bypasses the task-routing requirement for *gradient backpropagation* (since there is no test-time adaptation), a multi-head model merged under OFS-Tune and deployed on an interleaved stream still requires a routing mechanism at *inference time* to select the correct task-specific head for final prediction. 
*   **Actionable Suggestion:** The authors should clarify this distinction in the text, acknowledging that inference-time routing remains an open challenge for all multi-head merged models on mixed streams, independent of whether the merging coefficients are static or adapted online.

### 3. Tabular Reporting of Alternative Trajectory Configurations Across All Suites
While the authors introduce highly promising alternative parameterizations (such as Piecewise Splines and Block-wise layer-grouping) and discuss their benefits under non-smooth landscapes in Section 4.4 and Appendix D, the simulated multi-suite tables (Table 1, 2, and 3) currently only report global linear ($d=1$) and quadratic ($d=2$) trajectories.
*   **Actionable Suggestion:** It would strengthen the empirical completeness of the paper to report the performance of Piecewise Splines and Block-wise parameter sharing across ALL five evaluation suites in the supplementary tables. This would provide practitioners with a complete reference for choosing between global and localized trajectory constraints depending on the smoothness of their target models.

### 4. Delineating the Dimensional Scaling Limits of Nelder-Mead vs. OFS-Adam
The authors suggest that Nelder-Mead local search is extremely effective offline because the search space of low-degree polynomials is very small (e.g., 4 to 6 parameters). However, when applying localized constraints (e.g., Piecewise Splines with many knots) or scaling to joint merging of many tasks ($K \ge 5$), the parameter count will scale up. 
*   **Actionable Suggestion:** The paper would benefit from a brief discussion or analysis of the dimensional crossover point where derivative-free solvers (like Nelder-Mead) become computationally inefficient compared to first-order coordinate gradient descent (OFS-Adam) in terms of function evaluations.

### 5. Colorblind Accessibility of Figures
*   **Actionable Suggestion:** The authors should ensure that all lines and markers in Figure 1 and Figure 2 are distinguished not only by color but also by distinct line styles (e.g., dashed, dotted) or marker shapes (e.g., circles, triangles) to guarantee complete colorblind accessibility for readers.

---

## 4. Overall Recommendation

This is a **highly solid, methodologically rigorous, and outstandingly written paper** that provides a much-needed "reality check" to the model-merging community. It challenges a monolithic evaluation protocol and exposes critical transductive overfitting failure modes of online TTA. The statistical standards (30 independent seeds), thoroughness of the symmetrical optimization ablations, and honesty regarding limitations make it an excellent candidate for publication. Addressing the minor suggestions above will make this paper even stronger.
