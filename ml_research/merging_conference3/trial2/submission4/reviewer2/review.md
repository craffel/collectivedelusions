# Peer Review

## Strengths and Weaknesses

Please provide a thorough assessment of the strengths and weaknesses of the paper, touching on each of the following dimensions: soundness, presentation, significance, and originality.

### Strengths
- **Exemplary Writing and Structural Clarity:** The paper is exceptionally well-written, with a clear narrative, logical transitions, and mathematically precise definitions. The methodology is presented in a very structured, easy-to-follow manner that conforms perfectly to top-tier ICML standards.
- **Rigorous Appendices and Empirical Transparency:** The appendices are outstanding. The inclusion of Table 7 (hyperparameters), Appendix B (entropy analysis), and Appendix C (calibration size and data-free synthesis ablations) provides a high level of empirical transparency.
- **High-Quality Visualizations:** The figures (such as the Pareto frontier in Figure 1, the hyperparameter sweeps in Figure 3, the gating coefficient visualizations in Figure 2, and the professional TikZ flowchart in Figure 4) are beautifully designed, highly legible, and significantly enhance the reader's understanding.
- **Scientific Honesty Regarding Accuracy Trade-Offs:** The authors are highly commendable for their intellectual transparency. They do not attempt to obscure or downplay the massive **21.05% absolute accuracy gap** between EdgeMerge and SyMerge. This honest characterization of performance-resource trade-offs is rare and refreshing.

### Weaknesses
- **The Ablation Paradox (Functional Inertia of Adaptive Gating):** The central claim of the paper is that activation statistics enable dynamic channel-wise weight routing to resolve inter-task interference. However, Table 5 shows that replacing this complex adaptive gating with simple **Uniform Gating** ($\alpha_k = 1/K = 0.125$) yields the exact same average accuracy of **69.58%** (and Layer-wise Gating yields 69.59%). This proves that the proposed Forward-Only Activation Sampling (FOAS), Scale-Normalized Delta Activation Salience (SNDAS), and Channel-Wise Softmax Gating (CWSG) mechanisms are functionally inert. The performance gains are entirely driven by **Decoupled Scale Routing (DSR)**—specifically, setting a large scale for the transformer layers ($\lambda_{static} = 0.25$) and a highly regularized, small scale for the visual projection layer ($\lambda_{proj} = 0.025$, which is $0.20 / 8$). Thus, a simple static 2-scale model achieves the identical performance with zero calibration data, zero forward passes, and zero latency.
- **The Motivation-Workflow Contradiction:** The authors motivate EdgeMerge by arguing that edge devices cannot handle backpropagation. However, on-device calibration requires having access to all $K=8$ expert checkpoints, which requires 1.2 GB of storage—a requirement that the authors note triggers OOM errors on edge devices. To solve this, they propose an "Offline Developer Workflow" where the calibration is executed on a workstation and the single merged model is shipped to the edge. But in an offline workstation environment, the computational, latency, and memory constraints of the edge do not apply. In this setting, any developer would gladly wait 10 minutes to run SyMerge and achieve a single merged model with a **20.16% absolute accuracy advantage** rather than use EdgeMerge. This leaves the proposed method without any viable, practical deployment scenario.
- **Suspicious Numerical Invariance & Potential Division-by-Zero Bug:** In Table 2, the authors report that calibrating with physical validation images, random Gaussian noise, and synthetic zero tensors all yield the *exact* same average accuracy of **68.689%** down to three decimal places. Across the 8192 test images evaluated, this means not a single prediction changed. If zero tensors are used as input, the activation delta $\Delta H_k = X_k (W_k - W_{base})^T$ evaluates to zero, leading to a division-by-zero when normalized by its Frobenius norm (Equation 6):
  $$\tilde{\Delta} H_k = \frac{\Delta H_k}{\|\Delta H_k\|_F}$$
  If the code silently handles NaNs by replacing them with zeros (using e.g., `torch.nan_to_num`), the gating weights collapse to a perfectly uniform $1/K$. This explains why synthetic zero calibration behaves identically to uniform gating, and suggests that the authors' "manifold-projection hypothesis" is a post-hoc rationalization of a silent implementation bug.
- **Mismatched vs. Correct Calibration Invariance:** In Table 5, Mismatched Calibration (using $X_k^{base}$) and Correct Calibration (using $X_k^{expert}$) yield *exactly* the same average accuracy to three decimal places in 8 out of 9 rows. For the remaining row, the difference is exactly one image out of 8192. This level of invariance is statistically extraordinary and further indicates that the computed gating weights $\alpha_k$ have no functional impact on the final classification decision.
- **Insufficiency of Baselines & Statistical Significance:** The paper fails to compare against standard low-compute baselines like Fisher-Weighted Averaging. Furthermore, the 0.13% difference between Decoupled EdgeMerge (69.58%) and Decoupled TA (69.45%) is statistically insignificant given the estimated standard error of 0.51% for the multi-task evaluation suite.

---

## Soundness

**Poor**

The core claims of the paper regarding activation-based dynamic weight routing are empirically contradicted by the authors' own ablation studies. The adaptive gating mechanism does not outperform uniform gating, and its reported invariance to random noise and zero inputs points to a silent division-by-zero implementation bug that collapses the gating coefficients to uniform values.

---

## Presentation

**Excellent**

The paper is exceptionally well-structured, with clear explanations, precise mathematical formulas, and high-quality figures and tables.

---

## Significance

**Poor**

The paper has negligible practical significance. For offline development, developers will run SyMerge to get a 20% absolute accuracy boost for zero inference-time cost. For online on-device deployment, the method is blocked by the memory limits of storing all expert checkpoints. Furthermore, the performance of EdgeMerge can be replicated statically in 0.0 seconds without any data or calibration by simply setting two scaling parameters.

---

## Originality

**Fair**

While the paper attempts to translate activation-based channel selection techniques from network pruning to the model merging space, the core ideas are heavily drawn from established compression literature. Crucially, because the proposed adaptive gating is functionally inert and performs identically to uniform static averaging, the scientific novelty of this framework is severely limited.

---

## Overall Recommendation

**2: Reject**

The paper exhibits major technical flaws, including a complete lack of empirical delta over static uniform averaging (rendering the proposed dynamic routing redundant), a fatal motivational contradiction that leaves the method with no viable deployment scenario, and highly suspicious numerical invariances that strongly point to an implementation bug (division-by-zero NaNs collapsing to uniform weights). While the presentation quality is exceptional, these fundamental conceptual and empirical issues make the paper unsuitable for publication.
