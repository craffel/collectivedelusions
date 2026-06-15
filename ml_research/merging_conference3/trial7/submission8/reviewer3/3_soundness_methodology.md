# Evaluation Component 3: Soundness and Methodology

## 1. Clarity of the Description
* **Strengths:** The mathematical formulation of the entire framework—including the parametric routing, Parameter-Free Subspace Routing (PFSR), Confidence-Gated Hybrid Routing (CGHR), and Micro-Batch Homogenization (MBH)—is described with exceptional clarity.
* **Appendices:** The authors provide rigorous mathematical derivations in the appendices (e.g., Appendix A on the extreme value theory calibration factor $\sqrt{2\log C_k / d}$ and Appendix G on the UNC-PFSR Equivalence Theorem), making the underlying mathematical framework highly transparent and traceable.

---

## 2. Appropriateness of Methods
* **Dual-Pathway Design:** The confidence-gated design of CGHR is highly appropriate for balancing the expressive capacity of a trained parametric router against the robust, training-free stability of a parameter-free router.
* **Batch Partitioning (MBH):** MBH is a logically sound and appropriate method for resolving representation smoothing in heterogeneous deployment streams. By executing local ensembling over task-homogeneous segments, it directly tackles the root cause of "heterogeneity collapse."
* **Systems Optimizations:** The systems-level extensions, such as *Fusion Weight Caching* and *Warp Batch Padding*, are highly appropriate and address critical edge-case bottlenecks that are often neglected in theoretical ensembling literature.

---

## 3. Potential Technical Flaws and Limitations
As an evaluation of empirical soundness, several critical limitations and assumptions must be highlighted:
1. **Idealized Orthogonal Block Coordinates:** The entire framework is built upon the *Isolating Coordinate Sandbox* where the global feature vector $z_b$ is partitioned into $K$ disjoint block coordinates. This assumes that expert representational coordinates are completely decoupled and orthogonal. In actual neural networks (e.g., pre-trained Transformers), representation spaces are highly overlapping and non-orthogonal. While the authors propose SVD Subspace Projections (Appendix H) to address this, its evaluation is still limited to a simulated setup with random orthonormal bases rather than a real model.
2. **Structural Input Asymmetry:** There is a structural asymmetry where the parametric router (Pathway A) takes the global $D$-dimensional feature vector as input and must filter out Gaussian noise from non-active blocks, while the parameter-free router (Pathway B) takes localized, block-sliced feature representations. This structural advantage gives PFSR privileged coordinate boundary information, which may artificially boost its performance relative to the parametric baseline under small-$N$ training.
3. **Cascaded Routing Failures on Weak Experts:** The methodology is highly vulnerable to "cascaded routing failure" when dealing with weak or noisy experts (e.g., the SVHN expert ceiling of $26.4\%$). When an expert is weak, its representation block is highly corrupted, leading both PFSR and CGHR to frequently misclassify the active task and route inputs to cleaner, highly precise experts. This reveals a fundamental limitation of routing-based ensembling: the gateway router's performance is highly dependent on the baseline quality of the individual experts.

---

## 4. Reproducibility
* **Hyperparameter Specification:** Table 2 in Appendix B provides a highly complete list of all hyperparameters, including optimizer settings, learning rates, epochs, weight decay, noise scales, and temperature scales.
* **Algorithmic Flow:** Appendix C outlines the step-by-step algorithmic flow of Micro-Batch Homogenization (MBH), and the main text provides clear pseudocode-like formulations.
* **Sandbox Simplicity:** Because the evaluations are conducted in a simplified 1-layer coordinate simulation, reproducing the exact tables and figures presented in this paper should be extremely straightforward.
* **Real-World Gap:** However, reproducing these results on real-world networks (e.g., LoRA expert merging with LLaMA or CLIP backbones on actual GLUE or DomainNet datasets) is not directly reproducible from this paper, as no actual real-world implementations, model-weight structures, or training scripts for such backbones are provided.
