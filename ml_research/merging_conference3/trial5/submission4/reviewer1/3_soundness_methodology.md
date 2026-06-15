# Intermediate Review Evaluation: Soundness and Methodology

## 1. Clarity of the Description
The methodology of the paper is exceptionally well-described, mathematically rigorous, and structurally logical. 
* All mathematical formulas (Equations 1 through 16) are clearly defined, with standard, consistent notation for base weights, task vectors, dynamic coefficients, and global representations.
* The paper clearly articulates the mechanisms behind the unregularized Linear Router, the proposed **BL-Router** (which bounds Task Arithmetic coefficients), the **GLS-Router** (which introduces layer-wise scaling amplitudes), and the **BSigmoid-Router** (which replaces Softmax with independent Sigmoid projections).
* The Appendix provides excellent supplementary text, including Table 4 (expert training hyperparameters), details of the routing calibration protocol, a deep dive into the mathematical batch-averaging bottleneck, and a comprehensive complexity and latency benchmarking analysis (Table 5).

---

## 2. Appropriateness of Methods
The experimental and analytical methods chosen by the authors are highly appropriate for executing a critical methodological deconstruction:
* **Fully Converged Experts:** Training the task experts (MNIST, FashionMNIST, CIFAR-10, SVHN) to true convergence (accuracies ranging from 92.8% to 100%) is a critical methodological control. It eliminates any confounding variance arising from under-trained baseline networks, which has plagued prior work.
* **Isolating Confounders:** The use of targeted, surgical baselines (BL-Router for over-scaling, GLS-Router for layer specialization, L2 weight decay for overfitting) is highly effective at isolating specific variables. This allows the authors to dissect the exact drivers of performance rather than relying on global, entangled comparisons.
* **Few-shot Offline Calibration:** Restricting calibration to a tiny, balanced 64-sample set (16 samples per task) optimized over 100 steps is realistic, standard in the literature, and represents a fair playground for testing routing head stability.

---

## 3. Potential Technical Flaws & Methodological Nuances
The paper exhibits exceptional scientific honesty and rigorous self-evaluation. However, a few methodological details warrant discussion:
* **AdaMerging Stream Evaluation Modeling:** As discussed in Section 4.4, the online Test-Time Adaptation (TTA) baseline **AdaMerging** is modeled statically using its offline-calibrated homogeneous joint mean accuracy rather than executing the active online entropy-minimization gradient loop sequentially across the shuffled heterogeneous stream. While this is a pragmatic choice given AdaMerging’s severe latency (495 ms per batch), it does bypass the real-world temporal dynamics of TTA (e.g., temporal order effects, parameter drift, gradient noise at $B=1$). The authors openly acknowledge this simplification, which maintains high transparency.
* **Softmax Bounding vs. Uniform Merge Scaling:** The paper’s deconstruction of the BL-Router collapse is a major highlight. The mathematical deconstruction of the "structural under-scaling flaw" (Section 3.3) is highly sound and explains why a global coefficient cap of 0.3 on Softmax output collapses under uncertainty (limiting each task to 0.075, whereas Uniform Merge assigns 0.3 to each task independently). This is an excellent methodological insight that resolves a potential contradiction.
* **Unregularized Layer-wise Scaling in GLS-Router:** The authors show that GLS-Router overfits severely to the calibration set and collapsed on FashionMNIST. A deep analysis reveals that while weight decay was applied to the global projection weights, the layer-wise amplitudes $R_k^{(l)}$ (56 parameters) were unregularized, explaining their erratic scale profiles. Applying regularization directly to these scaling amplitudes remains an important optimization lesson.

---

## 4. Reproducibility
The reproducibility of this paper is **excellent**:
* The authors provide a link to their public repository containing code, converged task expert checkpoints, and evaluation scripts (`https://github.com/anonymous-researcher/bc-router`).
* Complete training hyperparameters for task experts are provided in Appendix A.1 (Table 4), and calibration details are provided in Appendix A.2.
* Exact optimization steps (100 steps), optimizer (Adam), learning rate ($1 \times 10^{-2}$), calibration set size (64 samples), and seeds (42 for data, and averaging over 3 random calibration seeds) are explicitly declared.
* The paper provides enough structural, algorithmic, and mathematical information for an expert to easily reproduce all proposed routers and findings.
