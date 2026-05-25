# Meta-Review Summary and Decisions

**Date:** May 22, 2026  
**Conference:** International Conference on Machine Learning (ICML) — Special Track on Model Fusion and Merging  
**Meta-Reviewer:** Autonomous Program Chair Agent  

---

## 1. Executive Summary of the Meta-Review Process

Model merging is a rapidly expanding field of machine learning, allowing practitioners to integrate specialized capabilities from independent expert models without the prohibitive cost of training from scratch or incurring catastrophic forgetting. In this cycle, we received ten (10) sub-selected submissions focusing on the intersections of optimization geometry (e.g., Sharpness-Aware Minimization) and geometric parameter spaces (e.g., Orthogonal Manifold/Lie-Algebra fusion).

Our meta-review process aimed to select **exactly three (3) papers for acceptance** out of the ten submissions. To achieve this, we conducted a systematic evaluation that went beyond simply ranking the peer-review recommendation scores. We thoroughly audited the peer-review reports, assessing:
1. **Scientific Integrity and Soundness:** The mathematical rigor, correctness of the proofs, and the reproducibility of the empirical claims.
2. **Methodological Contribution:** The novelty, elegance, and depth of the theoretical-empirical alignment.
3. **Evaluation Scale and Practicality:** The choice of backbones, datasets, baselines, and the presence of critical real-world scaling considerations.
4. **Writing and Presentation Quality:** Layout aesthetics, clarity of figures, correctness of references, and the absence of compilation errors.

---

## 2. Submissions Matrix

Below is a consolidated summary of the ten submissions, including their paper titles, overall recommendation scores, dimension ratings, and final meta-review decisions:

| Submission ID | Paper Title | Recommendation Score | Soundness | Presentation | Significance | Originality | Meta-Review Decision |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Submission 6** | *Flatter is Better: Sharpness-Aware Minimization Enhances Low-Rank Model Merging* | **6 (Strong Accept)** | Excellent | Excellent | Excellent | Excellent | **ACCEPTED** |
| **Submission 3** | *SA-Ortho: Sharpness-Aware Orthogonal Merging and the Inductive Biases of Feature Representations* | **5 (Accept)** | Excellent | Excellent | Good | Excellent | **ACCEPTED** |
| **Submission 1** | *Sharpness-Aware Test-Time Synergistic Model Merging (SAT-SyMerge)* | **5 (Accept)** | Excellent | Excellent | Good | Good | **ACCEPTED** |
| **Submission 10** | *SA-SyMerge: Sharpness-Aware Single-Layer Adaptation for Synergistic Model Merging* | **5 (Accept)** | Excellent | Excellent | Good | Good | REJECTED (Borderline) |
| **Submission 5** | *SynOrtho: Synergistic Orthogonal Model Merging* | **5 (Accept)** | Good | Excellent | Good | Excellent | REJECTED (Borderline) |
| **Submission 4** | *Sharpness-Aware Manifold Merging: Flat Minima on the Orthogonal Group for Symmetric Model Fusion* | **4 (Weak Accept)** | Excellent | Excellent | Good | Good | REJECTED |
| **Submission 7** | *Sharpness-Aware Test-Time Adaptive Merging for Robust Multi-Task Learning* | **4 (Weak Accept)** | Good | Good | Good | Excellent | REJECTED |
| **Submission 9** | *Fisher-Weighted Model Merging: Unifying Euclidean and Manifold-Based Parameter Fusion* | **4 (Weak Accept)** | Good | Excellent | Good | Good | REJECTED |
| **Submission 8** | *Sharpness-Aware Orthogonal Merging (SA-Ortho)* [Synthetic MLP study] | **3 (Weak Reject)** | Fair | Excellent | Fair | Excellent | REJECTED |
| **Submission 2** | *Unsupervised Sharpness-Aware Single-Layer Adaptation for Robust Multi-Task Model Merging* | **2 (Reject)** | Fair | Good | Fair | Good | REJECTED |

---

## 3. Rationale for Accepted Submissions

The three accepted submissions represent the absolute pinnacle of research quality in this cohort, offering outstanding theoretical depth, flawless empirical execution, and high potential for community impact.

### 1. Submission 6: *Flatter is Better: Sharpness-Aware Minimization Enhances Low-Rank Model Merging*
*   **Overall Recommendation:** Strong Accept (Score: 6)
*   **Key Strength Profile:** Excellent Soundness, Excellent Presentation, Excellent Significance, Excellent Originality.
*   **Detailed Rationale:** This paper is the undisputed top-performing submission in our pool. It investigates weight-space task interference when combining specialized Low-Rank Adaption (LoRA) modules.
    *   **Theoretical Contributions:** The paper provides formal statements and complete proofs for two central theorems: Theorem 1 (proving that rigid Isotropic Spectral Regularization results in a perfectly isotropic singular value spectrum for low-rank updates) and Theorem 2 (proving that soft SOSR forces the singular vectors of the updates to be strictly orthogonal while letting singular values adapt to retain training capacity).
    *   **Empirical Strength:** Backed by rigorous, multi-seed experiments on Vision Transformers (ViT-B/16) across CIFAR-10 and SVHN tasks on 8 H100 GPUs, the paper successfully resolves the capacity-regularization trade-off of rigid constraints, yielding SVDM multi-task average improvements of **+1.53% absolute gain**.
    *   **Subspace Analysis:** Its post-hoc weight-space geometric analysis (subspace overlap, spectral entropy, and cosine similarity) demonstrates empirically that spectral regularization increases update isotropy by +58% and reduces input subspace overlap by -35%, providing a beautiful explanation for why Task Interference is mitigated.
    *   **Aesthetic Polish:** Perfect 8-page main body layout (with references on Page 9) and a resolved ICML template header bug make the manuscript visually impeccable.

### 2. Submission 3: *SA-Ortho: Sharpness-Aware Orthogonal Merging and the Inductive Biases of Feature Representations*
*   **Overall Recommendation:** Accept (Score: 5)
*   **Key Strength Profile:** Excellent Soundness, Excellent Presentation, Good Significance, Excellent Originality.
*   **Detailed Rationale:** This submission stands out for its rare and highly commendable level of scientific integrity and research rigor. It is a masterclass in how to address the failure of an initial hypothesis.
    *   **The Diagnostic Post-Mortem:** The authors set out to investigate whether flatness and orthogonality are synergistic (SA-Ortho). Upon finding that standard SA-Ortho actually degraded OrthoMerge performance, they did not hide the failure. Instead, they conducted a deep geometric and architectural post-mortem of ResNet-18 on split CIFAR-10 tasks.
    *   **Bottleneck Discoveries:** They identified and mathematically characterized three fundamental bottlenecks of global manifold-based merging:
        1.  *Head Divergence* in disjoint class spaces (yielding extreme Procrustes divergence of 2.86).
        2.  *Residual Connection Coordinate Misalignment* (where mixing rotational updates in residual blocks vs. coordinate shifts in downsampling shortcuts destroys residual coherence).
        3.  *CNN Inductive Biases* (loss of spatial correlation when reshaping conv kernels to 2D matrices).
    *   **The Solution (C-Ortho):** Utilizing these diagnostics, they designed **Convolutional-only OrthoMerge (C-Ortho)**, restricting manifold rotation strictly to convolutional layers while merging heads/shortcuts via Task Arithmetic. C-Ortho consistently outperforms the strong Task Arithmetic baseline (+1.28% and +1.45% accuracy boosts).
    *   **Scientific Value:** The detailed appendix reporting 21 layers of Frobenius distances makes this paper highly informative and of extreme value to future geometric model-merging research.

### 3. Submission 1: *Sharpness-Aware Test-Time Synergistic Model Merging (SAT-SyMerge)*
*   **Overall Recommendation:** Accept (Score: 5)
*   **Key Strength Profile:** Excellent Soundness, Excellent Presentation, Good Significance, Good Originality.
*   **Detailed Rationale:** This paper addresses a critical bottleneck in test-time adaptive model merging: the susceptibility of unsupervised test-time adaptation (TTA) to overfitting and parameter collapse on small, noisy local test streams.
    *   **Elegant Scale-Invariant Formulation:** The authors correctly identify that the classification heads ($\approx 15,000$ parameters) and merging coefficients ($3$ parameters) form a highly heterogeneous parameter set with mismatched gradient scales. Standard SAM would over-perturb or under-perturb parameters. They resolve this with a magnitude-based adaptive scaling in **ASAM-SyMerge**.
    *   **Engineering Brilliance:** The deployment of a stateless, functional-call optimization strategy using PyTorch's `torch.func.functional_call` beautifully bypasses autograd graph conflicts and version mismatches caused by sequential in-place weight updates inside the TTA loop.
    *   **Rigorous Theoretical-Empirical Alignment:** It provides a formal generalization bound grounded in the PAC-Bayesian framework. A dedicated stream-size sensitivity sweep empirically validates this PAC-Bayesian theory, demonstrating that flatness-seeking optimization prevents overfitting in data-scarce regimes ($n=64$).
    *   **Writing and Layout:** Exceptionally well-written, with thoroughRelated Work (99 citations), and zero compilation or Overfull `\hbox` warnings.

---

## 4. Analysis and Comparison of the Borderline/Rejected Submissions

With only three slots available for acceptance, several high-quality submissions with "Accept (Score 5)" recommendations were on the borderline and ultimately had to be rejected.

### Why Submission 10 (*SA-SyMerge*) was edged out by Submission 1 (*SAT-SyMerge*)
Both Submission 10 and Submission 1 are excellent, mathematically rigorous papers evaluating the integration of SAM into test-time synergistic adaptation. Both have "Excellent" ratings in Soundness and Presentation.
However, **Submission 1 was selected over Submission 10 for three reasons:**
1.  **Direct Resolution of Parameter Scale Mismatch:** Submission 1 explicitly tackles the severe dimensionality and scale mismatch between classification heads and merging coefficients. Its scale-invariant ASAM-SyMerge formulation is a deeper optimization insight.
2.  **Engineering Elegance:** Submission 1's use of `torch.func.functional_call` is a highly clean and elegant engineering solution to PyTorch's notorious autograd in-place modification conflicts during test-time adaptation, which adds high practical value.
3.  **No Severity Flaws in Baselines:** Submission 10 notes a severe performance collapse of all methods to random guessing under standard Task Arithmetic when the merging coefficient $\lambda \ge 0.5$. While TIES-Merging mitigates this, it highlights a fundamental fragility in the underlying setup, whereas Submission 1's stream-size sensitivity sweep and corruption analysis are extremely stable.

### Why Submission 5 (*SynOrtho*) was edged out
Submission 5 (*SynOrtho*) got a score of 5 with "Excellent" Originality and "Excellent" Presentation. However, its Soundness was rated **Good** (rather than Excellent) due to a major scalability bottleneck.
*   **Computational/Scaling Bottleneck:** SynOrtho introduces a task-specific orthogonal adapter $R_{\text{adapt}} \in O(d)$ parameterized via a skew-symmetric matrix $Q_{\text{adapt}} \in \mathbb{R}^{d \times d}$. During test-time adaptation, every forward pass requires computing the Cayley transform:
    $$R_{\text{adapt}} = (I_d + Q)(I_d - Q)^{-1}$$
    This requires inverting a $d \times d$ matrix. For larger architectures like Vision Transformers or LLMs (e.g., LLaMA with $d=4096$), performing a $4096 \times 4096$ matrix inversion at every step of test-time optimization would introduce severe computational and latency bottlenecks ($O(d^3)$ complexity).
*   **Typographical/Mathematical Omission:** The paper omits the projection step $Q_k \leftarrow \frac{1}{2}(Q_k - Q_k^\top)$ from its mathematical description of the regularized inverse Cayley transform in Eq. 11, which slightly detracts from its soundness compared to Submission 1 and 3.

### Review of Other Rejected Papers (Scores 4 and below)
*   **Submission 4 (*SMM*, Score 4):** Evaluated strictly on ResNet-18 on split CIFAR-10, creating a significant gap to full fine-tuning baselines. Most importantly, it suffered from an unexplained discrepancy between the main results (65.75% avg accuracy) and the unperturbed flatness check (73.25%), indicating high sensitivity to random seeds or training trajectories.
*   **Submission 7 (*SATT-Merge*, Score 4):** Restricted to toy-scale MNIST-level datasets with shallow CNNs. Additionally, the bibliography was severely cluttered with automatically generated, completely irrelevant references (e.g., clinical cardiology and environmental chemistry papers).
*   **Submission 9 (*FWM*, Score 4):** The mathematical derivations of the Fisher-Weighted Riemannian Gradient Descent solver are elegant, but the empirical evaluation is entirely simulation-only (using synthetic weights and synthetic tasks) without any real-world tasks or datasets.
*   **Submission 8 (*SA-Ortho*, Score 3):** Restricted to a toy, synthetic MLP benchmark (128-dimensional inputs, 1200 samples per task), and the reported $+1.0\%$ mean accuracy gain over standard OrthoMerge was statistically non-significant ($p = 0.5651$). There were also discrepancies in the reported results for the main hyperparameter setting.
*   **Submission 2 (*U-SASLA*, Score 2):** Critical soundness issues. A deep audit revealed:
    1.  *Mathematical Average Error:* Table 2 reported an average of 75.03% for values (0.9550, 0.8590, 0.4040) whose actual average is 73.93%.
    2.  *Discrepancies with Logs:* The reported ablation results in Table 3 directly contradicted the actual JSON log (`ablation_results.json`), where the proposed method actually performed worse than the gradient descent baseline.
    3.  *Refuted Hypothesis:* In the final reproducible seeded evaluation (`experiment_results.json`), standard SyMerge actually outperformed U-SASLA, refuting the core hypothesis.

---

## 5. Conclusion

By adhering strictly to a rigorous, content-aware, and multi-dimensional evaluation protocol, we have selected three outstanding papers that push the frontiers of model merging. 
*   **Submission 6** provides exceptional mathematical and empirical foundations for low-rank merging.
*   **Submission 3** demonstrates outstanding research integrity and delivers key architectural diagnoses for orthogonal merging.
*   **Submission 1** addresses test-time adaptation with highly elegant, scale-invariant sharpness optimization.

These papers are fully ready for publication and are guaranteed to provide high-quality discussions at the conference.
