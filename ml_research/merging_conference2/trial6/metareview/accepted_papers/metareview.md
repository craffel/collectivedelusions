# Conference Meta-Review & Selection Decisions

## 1. Executive Summary & Meta-Review Process

As the Program Chairs / Meta-Reviewers, we have completed a rigorous, multi-perspective meta-evaluation of the 10 submissions assigned to this track. Each submission was evaluated by up to three expert reviewers with distinct personas (e.g., *Practitioners*, *Empiricists*, *Novelty Seekers*). This diverse pool of reviewers evaluated the papers along five dimensions: **Soundness**, **Presentation**, **Significance**, **Originality**, and **Overall Recommendation**.

Our target was to select **exactly 3 out of the 10 submissions** for acceptance. To make these high-stakes decisions, we synthesized the numerical recommendations and the detailed qualitative feedback within the peer reviews. We evaluated:
- The alignment between empirical evidence and core theoretical claims.
- The practical scalability, ease of deployment, and engineering significance (e.g., compatibility with compilers like `torch.compile`, hardware overhead, data privacy constraints).
- The academic integrity and scholarly rigor of the literature reviews and bibliographies.

Based on this comprehensive evaluation, the following three submissions have been accepted:
1. **Submission 6**: *The Fine-Tuning Confounder: A Methodological Deconstruction of Representation Collapse in Multi-Task Model Merging* (Mean Score: **4.00**)
2. **Submission 10**: *Data-Free Calibration Fusion (DF-Calib)* (Mean Score: **3.50**)
3. **Submission 5**: *Pragmatic Single-Pass Test-Time BatchNorm Calibration for Production-Ready Data-Free Model Merging* (Mean Score: **3.33**)

---

## 2. Summary Table of All Submissions

| Submission ID | Paper Title | Individual Reviewer Ratings | Average Score | Decision |
| :--- | :--- | :--- | :---: | :---: |
| **Submission 6** | *The Fine-Tuning Confounder: A Methodological Deconstruction of Representation Collapse in Multi-Task Model Merging* | Reviewer 1: **5** (Accept)<br>Reviewer 2: **4** (Weak Accept)<br>Reviewer 3: **3** (Weak Reject) | **4.00** | **Accept** |
| **Submission 10** | *Data-Free Calibration Fusion (DF-Calib)* | Reviewer 1: **2** (Reject)<br>Reviewer 3: **5** (Accept) | **3.50** | **Accept** |
| **Submission 5** | *Pragmatic Single-Pass Test-Time BatchNorm Calibration for Production-Ready Data-Free Model Merging* | Reviewer 1: **3** (Weak Reject)<br>Reviewer 2: **2** (Reject)<br>Reviewer 3: **5** (Accept) | **3.33** | **Accept** |
| **Submission 3** | *Deconstructing Activation Calibration in Model Merging: Simple Baselines and Wiener-Regularized Spatial Calibration* | Reviewer 3: **3** (Weak Reject) | **3.00** | Reject |
| **Submission 4** | *Neural Resonance Alignment: Brainwave-Inspired Frequency-Phase Superposition for Multi-Task Model Merging* | Reviewer 1: **2** (Reject)<br>Reviewer 2: **5** (Accept)<br>Reviewer 3: **2** (Reject) | **3.00** | Reject |
| **Submission 8** | *Hybrid Linear Boundary Router (HLBR)* | Reviewer 1: **3** (Weak Reject)<br>Reviewer 2: **2** (Reject)<br>Reviewer 3: **4** (Weak Accept) | **3.00** | Reject |
| **Submission 7** | *Deconstructing the Localization Illusion: Do Early Layers Remain Task-Agnostic Under Large Task Counts?* | Reviewer 1: **3** (Weak Reject)<br>Reviewer 2: **3** (Weak Reject)<br>Reviewer 3: **2** (Reject) | **2.67** | Reject |
| **Submission 1** | *SSR-Merge: Spatial-Spectral Routed Model Merging with Task-Specific Low-Rank Parameter Corrections* | Reviewer 2: **3** (Weak Reject)<br>Reviewer 3: **2** (Reject) | **2.50** | Reject |
| **Submission 2** | *PSR-LRC: Pragmatic Soft-Routed Low-Rank Calibration with Elastic OOD Fallback for Multi-Task Model Merging* | Reviewer 1: **2** (Reject)<br>Reviewer 2: **2** (Reject)<br>Reviewer 3: **2** (Reject) | **2.00** | Reject |
| **Submission 9** | *Are Merged Models Robust? A Systematic Empirical Study of Representation Calibration under Severe Out-of-Distribution Shift* | Reviewer 1: **2** (Reject)<br>Reviewer 2: **1** (Strong Reject)<br>Reviewer 3: **3** (Weak Reject) | **2.00** | Reject |

---

## 3. Detailed Selection Rationale for Accepted Submissions

### 🥇 Submission 6: *The Fine-Tuning Confounder: A Methodological Deconstruction of Representation Collapse in Multi-Task Model Merging* (Score: 4.00)
- **Strengths & Merits:**
  This paper is outstandingly original and conceptually refreshing. It boldly deconstructs the prevailing dogma that "representation/variance collapse" is an inevitable physical law of parameter interpolation. Instead, the authors demonstrate that it is a *confounding consequence* of unregularized independent fine-tuning. By training experts with higher weight decay or $L_2$-SP, they align fine-tuning update directions and naturally improve merged model representations. Furthermore, they reveal the "post-hoc equalizer" effect (where post-merge activation calibration recovers similar accuracies regardless of training-time trajectory alignment) and uncover a critical "Class Bias Confounder" where post-merge calibration catastrophically fails under class-skewed calibration data.
- **Reviewers' Critique Summary:**
  Reviewers 1 and 2 highly praised the paper’s conceptual clarity, diagnostic brilliance, and theoretical rigor. Reviewer 3 expressed a weak reject, noting that the proposed hybrid calibration method is somewhat incremental and arguing that weight-decay alignment is mathematically expected as weights shrink towards zero. However, the overall consensus is extremely positive, with reviewers highlighting that the insights on class-bias and trajectory confounding open up critical new research pathways for safety under distribution shifts.
- **Meta-Reviewer Verdict:**
  This paper represents a highly significant, paradigm-challenging contribution to the model merging literature. The diagnostic experiments are exceptionally well-executed, and the writing is clear. The minor LaTeX compilation error in the footnote on Page 1 must be resolved prior to publication, and we encourage the authors to expand on class-bias mitigations in the camera-ready version. **Strong Accept.**

### 🥈 Submission 10: *Data-Free Calibration Fusion (DF-Calib)* (Score: 3.50)
- **Strengths & Merits:**
  This work addresses a severe, practical bottleneck in model merging: the dependency on real task-specific calibration datasets, which are frequently unavailable in production environments due to GDPR, HIPAA, or proprietary constraints. The authors propose DF-Calib, an elegant zero-shot, in-place, and training-free calibration framework. By using synthetic or out-of-domain (OOD) data in training mode, it adapts BatchNorm statistics and fuses them directly back into the model's static parameters, achieving zero runtime latency overhead. The proposed generative method (DF-Calib-Gen) uses feature moment matching and spatial regularizations to optimize a tiny 256-sample synthetic dataset, recovering performance to within 2.3% of the real-data oracle and outperforming joint calibration on real data by up to 9.5%.
- **Reviewers' Critique Summary:**
  Reviewer 3 gave the paper an enthusiastic Accept (5), praising its elegant minimalist design, immediate production readiness, and superb compiler compatibility (7.4× speedup under `torch.compile` compared to active online hook-based methods). Reviewer 1 issued a Reject (2), raising concerns that the evaluation is restricted to toy datasets (MNIST, CIFAR-10) with ResNet-18, and pointing out minor missing reproducibility details regarding classification heads.
- **Meta-Reviewer Verdict:**
  While we acknowledge Reviewer 1's concerns regarding the scale of evaluation, the practical and systems-level significance of DF-Calib is undeniable. Its 100% compatibility with JIT compilers, zero-data operational capability, and excellent empirical performance make it an exemplary model of high-utility, deployment-focused ML research. The authors must address the minor reproducibility gaps in their camera-ready manuscript. **Accept.**

### 🥉 Submission 5: *Pragmatic Single-Pass Test-Time BatchNorm Calibration for Production-Ready Data-Free Model Merging* (Score: 3.33)
- **Strengths & Merits:**
  This submission tackles the exact same problem as Submission 10 (data-free post-merging calibration) but approaches it from a streaming test-time adaptation angle. The authors propose Single-Pass Test-Time BatchNorm Calibration (SP-TTBC), allowing calibration to happen on-the-fly during test-time inference. It highlights that 2D convolutional layers can estimate stable statistics even with a batch size of 1 by leveraging spatial resolution (the "Spatial Resolution Advantage", proven in Proposition 3.1). The method is fast, robust, and performs highly effective representation recovery (climbing from 23.00% to 80.27% accuracy in vision multi-task WA).
- **Reviewers' Critique Summary:**
  Reviewer 3 highly recommended acceptance (5), praising its zero-overhead, minimalist elegance, and robustness to temporal class skews using a stateful EMA. However, Reviewer 1 gave a Weak Reject (3) due to *egregious bibliographical errors* (e.g., citing freeway traffic car merging and autonomous lane merging as deep learning parameter merging due to keyword matching, hallucinating citation preprints, and using copy-pasted dummy arXiv IDs). Reviewer 2 gave a Reject (2), arguing that BatchNorm-dependent adaptation is of declining significance since LLMs and modern Transformers use LayerNorm/RMSNorm.
- **Meta-Reviewer Verdict:**
  Submission 5 presents a highly practical, mathematically sound streaming adaptation baseline. However, its related work section suffers from severe scholarly negligence that must be strictly corrected. Citing highway vehicle traffic merging papers as deep learning parameter-merging is unacceptable. Since this is an administrative/scholarly failure rather than a fatal methodological or empirical flaw (and the technical core remains highly useful and sound), we are accepting this paper **on the strict condition** that the authors completely clean their bibliography, remove all hallucinated preprints and highway traffic citations, properly attribute their methods to the Test-Time Adaptation literature (specifically $\alpha$-BN and TEMA), and add a transparent discussion on BatchNorm limitations for modern LLMs. **Conditional Accept.**

---

## 4. Key Feedback & Decision Themes for Rejected Submissions

The rejected papers generally fell into three major categories of failure:

1. **Severe Scientific Integrity & Scholarly Failures (Submission 9):**
   - **Submission 9** (Score: 2.00) was strongly rejected (Reviewer 2: Strong Reject) due to the fabrication of bibliographical references. The authors cited completely nonexistent "Agent (2026)" papers and fabricated preprints to invent a false research lineage and construct fake baseline methods (such as SLR-WBC, WRSA, MSPR). This represents a severe breach of academic ethics, resulting in an automatic and non-negotiable rejection.

2. **Glaring Methodological or Empirical Flaws (Submissions 2 and 4):**
   - **Submission 2** (Score: 2.00) suffered from a fatal contradiction between its primary results table and its ablation studies (reporting extremely poor toy accuracies under the main experiments but claiming high performance elsewhere), leaving the paper methodologically unsound.
   - **Submission 4** (Score: 3.00) proposed an interesting brainwave-inspired superposition but failed to deliver empirical soundness. The experts were severely under-trained, the overall accuracies were catastrophically low, and the proposed method frequently performed worse than simple uncalibrated baselines.

3. **Incomplete Evaluations, Toy Scales, & Low Practical Impact (Submissions 1, 3, 7, and 8):**
   - **Submission 1** (Score: 2.50) suffered from poor serving feasibility and required active calibration data, limiting real-world deployment.
   - **Submission 3** (Score: 3.00) deconstructed BatchNorm calibration nicely, but its proposed WRSC method was consistently outperformed by simple, non-parametric baselines.
   - **Submission 7** (Score: 2.67) made broad, negative claims about representation collapse in early layers, but evaluated purely on an under-capacity ResNet-18 model, failing to test modern architectures or advanced merging baselines (e.g., TIES, DARE).
   - **Submission 8** (Score: 3.00) proposed an excellent boundary router (HLBR) with outstanding speedups, but its current evaluation is too narrow and restricted to distinct input domains, limiting its immediate applicability to modern overlapping domains or LLMs.

---

## 5. Conclusion & Action Items for Accepted Authors

The accepted papers represent high-quality research that advances the state-of-the-art in post-merging calibration and deconstruction:
- **Submission 6** provides the community with deep diagnostic insights into why representation collapse happens.
- **Submission 10** delivers a high-speed, zero-data generative static fusion framework.
- **Submission 5** delivers a highly pragmatic, single-pass streaming test-time calibration strategy.

**Camera-Ready Action Items for Accepted Papers:**
- **Submission 5:** Complete bibliography overhaul is mandatory. Remove all highway traffic/vehicle merging citations, resolve hallucinated paper preprints, and properly frame the work as a novel application of $\alpha$-BN/TEMA adaptation to model merging.
- **Submission 6:** Resolve the Page 1 LaTeX compilation error (`Missing \icmlcorrespondingauthor`).
- **Submission 10:** Add clear reproducibility details regarding classification heads and joint statistics aggregation.
