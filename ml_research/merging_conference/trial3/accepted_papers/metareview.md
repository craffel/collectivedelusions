# Conference Meta-Review Report

**Date:** Friday, May 22, 2026  
**Meta-Reviewer:** Conference Program Chair (Gemini CLI Autonomous Agent)  
**Task:** Select the top 3 out of 10 submissions for acceptance, copy their papers into the `accepted_papers` directory, and provide a comprehensive meta-review summary of the decision-making process.

---

## 1. Executive Summary

In this meta-review cycle, we evaluated **10 neural network model merging and adaptation submissions** (`submission1` through `submission10`). Each submission underwent a rigorous peer-review process, resulting in detailed reviews (`review.md`) stored in their respective subdirectories.

Of the 10 submissions:
- **7 submissions** were recommended for **Acceptance (Score: 5)**.
- **2 submissions** were recommended for **Weak Acceptance (Score: 4)**.
- **1 submission** was recommended for **Weak Rejection (Score: 3)**.

Given the strict conference quota requiring us to choose **exactly 3 submissions for acceptance**, we conducted a secondary meta-evaluation. While numerical ratings are a helpful starting point, our final decisions are grounded in a deep qualitative comparison of:
1. **Technical Soundness and Theoretical Rigor**: The correctness and elegance of mathematical proofs, and the integrity of empirical baselines and experimental protocols.
2. **Presentation and Scholarly Quality**: The clarity, structure, styling (e.g., absence of overfull boxes/layout spillage), and historical literature contextualization of the manuscripts.
3. **Significance and Real-World Impact**: Whether the paper resolves a critical bottleneck, introduces a scalable method, or provides key insights that will shape future deep model fusion research.
4. **Originality and Insightfulness**: The novelty of the proposed ideas or the counter-intuitive nature of the scientific findings.

Based on this multi-dimensional assessment, we have finalized our decisions to **accept the following three outstanding submissions**:
1. **Submission 1**: *SATA-SBF & SATA-RGP: Convex Geometric Test-Time Adaptation for Robust Synergistic Model Merging*
2. **Submission 5**: *S2C-Merge: Teacher-Free Test-Time Model Merging via Self-Supervised Contrastive and Consistency Adaptation*
3. **Submission 7**: *EWC-TTA: Elastic Weight Consolidation-Guided Test-Time Adaptation for Dynamic Model Merging*

The corresponding PDF papers have been copied into the `accepted_papers` directory as `submission1.pdf`, `submission5.pdf`, and `submission7.pdf`.

---

## 2. Comprehensive Decision Matrix

Below is a structured comparative summary of all 10 submissions evaluated during this cycle, ordered by their peer-review recommendation and meta-review rank.

| Rank | Submission | Overall Rating | Soundness | Presentation | Significance | Originality | Final Status | Core Decision Rationale |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | **Submission 1** | **5: Accept** | Excellent | Excellent | Excellent | Excellent | **ACCEPTED** | Outstanding methodological rigor; complete clean sweep across all benchmarks; immaculate presentation; zero layout issues; excellent literature integration. |
| **2** | **Submission 5** | **5: Accept** | Excellent | Excellent | Excellent | Good to Exc. | **ACCEPTED** | Resolves the critical "Teacher-Overhead Paradox" (storing expert models at test time), showing a 3.0$\times$ VRAM reduction and 2.5$\times$ speedup; elegant mathematical proof of decision-boundary collapse. |
| **3** | **Submission 7** | **5: Accept** | Excellent | Excellent | Good to Exc. | Good | **ACCEPTED** | Resolves the "Autograd Graph Disconnection" using PyTorch's modern stateless `functional_call` API; extremely data-efficient (N=10) and zero-source viable prior; deep "temporal lag" study. |
| **4** | **Submission 6** | **5: Accept** | Excellent | Excellent | Good | Excellent | *Declined* | Fascinating and counter-intuitive discovery of the "Flatness-Geometry Paradox" showing inverse-Fisher weighting backfires; however, experiments are limited to smaller-scale ResNet-18 on split CIFAR-10. |
| **5** | **Submission 4** | **5: Accept** | Excellent | Excellent | Good to Exc. | Excellent | *Declined* | Elegant "FG-SPOR-Inverse" method reconciling flatness and coordinate alignment; excellent layer-wise drift analysis; however, limited to ResNet-18 on split CIFAR-10 and diagonal Fisher approximations. |
| **6** | **Submission 3** | **5: Accept** | Excellent | Excellent | Good | Excellent | *Declined* | Highly principled covariance-aligned low-rank projection (CALP) for LoRA merging; exposes standard TTA initialization biases; however, empirical accuracy gains over simpler baselines are marginal. |
| **7** | **Submission 8** | **5: Accept** | Good | Excellent | Good | Excellent | *Declined* | Elegant "Fisher-Weighted SPOR" resolving over-regularization; however, limited to small ResNet-18/CIFAR-10, uses a biased Fisher estimate, and suffers from classifier-head interference in global merging. |
| **8** | **Submission 2** | **4: Weak Acc.** | Good | Excellent | Fair | Good | *Declined* | Flawless mathematical derivations, but soundness and significance are capped due to inconsistent out-of-distribution performance gains across tasks. |
| **9** | **Submission 9** | **4: Weak Acc.** | Good | Good | Good | Good | *Declined* | Strong theoretical insights, but strictly limited to toy networks (2-layer CNN/MLP) on toy vision datasets; omitting the proposed scaling actually improved performance in some settings. |
| **10** | **Submission 10** | **3: Weak Rej.**| Fair | Excellent | Fair | Good | *Declined* | Proposes fine-grained coordination scaling, but suffers from severe soundness issues (unsupported claims and marginal gains) and low significance. |

---

## 3. In-Depth Meta-Reviews of Accepted Submissions

### Submission 1: *SATA-SBF & SATA-RGP: Convex Geometric Test-Time Adaptation for Robust Synergistic Model Merging*
* **Overall Recommendation:** **5: Accept** (Selected as the #1 Overall Submission)
* **Key Dimensions:** Soundness: **Excellent** | Presentation: **Excellent** | Significance: **Excellent** | Originality: **Excellent**
* **Summary of Contribution:**  
  This paper introduces two novel geometric test-time adaptive merging frameworks (**SATA-SBF** and **SATA-RGP**) to address parameter interference, representation collapse, and domain shift in online model merging. It proposes three primary technical innovations:
  1. *Convex Tensor-wise Model Merging (C-TMM)*: Constrains merging coefficients to reside strictly within the convex hull of the expert trajectories via a layer-wise Softmax.
  2. *Soft-Bounded Fisher-Guided SAM (SBF-SAM)*: Scales sharpness perturbations inversely with the online Fisher Information Matrix (FIM) diagonal to protect highly sensitive expert features.
  3. *Entropy-Adaptive Relative Geometry Preservation (EA-RGP)*: Dynamically aligns the adapted classification head's Gram matrix with that of the original expert head, preserving semantic class similarities.
* **Review Content & Metareview Evaluation:**  
  The review for Submission 1 is extraordinarily positive. On the technical side, the framework is mathematically elegant, rigorously derived, and thoroughly validated. Empirically, it achieves a **decisive, complete clean sweep**—consistently and significantly outperforming all baselines on MNIST/FashionMNIST/KMNIST across clean and corrupted environments (Gaussian Noise, Gaussian Blur, Contrast reduction). The ablation study is highly robust, successfully dissecting the contributions of C-TMM and EA-RGP. 
  In terms of presentation, the paper is impeccable: extremely well-written, mathematically precise, and typeset beautifully with **zero overfull `\hbox` warnings**, reflecting an outstanding level of scholarly polish. The bibliography contains 59 highly relevant, state-of-the-art citations. 
* **Core Justification for Acceptance:**  
  Submission 1 represents the highest standard of technical and empirical excellence in this cycle. It is a highly complete, flawless paper that is immediately publication-ready and will serve as a strong baseline for future test-time adaptation and model merging literature.

---

### Submission 5: *S2C-Merge: Teacher-Free Test-Time Model Merging via Self-Supervised Contrastive and Consistency Adaptation*
* **Overall Recommendation:** **5: Accept** (Selected as the #2 Overall Submission)
* **Key Dimensions:** Soundness: **Excellent** | Presentation: **Excellent** | Significance: **Excellent** | Originality: **Good to Excellent**
* **Summary of Contribution:**  
  This paper identifies and breaks the **Teacher-Overhead Paradox**—the practical bottleneck where state-of-the-art test-time model merging methods require all unmerged expert models to be held in GPU memory as "teachers" at inference time, scaling VRAM requirements linearly ($O(K)$) and defeating the resource-saving goal of merging. To resolve this, the authors propose **S2C-Merge**, a fully self-supervised, teacher-free test-time model merging framework. S2C-Merge optimizes layer-wise merging coefficients on-the-fly using prediction entropy minimization and task-aware augmentation consistency.
  To prevent **Decision Boundary Collapse** (where adapting classifier heads and merging weights without teacher supervision leads to degenerate predictions), they propose restricting online adaptation strictly to the low-dimensional merging coefficients while freezing classification heads. They also extend this to sharpness-aware optimization over the coefficients (**S2C-SAM**).
* **Review Content & Metareview Evaluation:**  
  The review highlights the exceptional practical significance of this work. Evaluated on a ViT-B/16 backbone with Rank-8 LoRA adapters on an alternating CIFAR-10 and SVHN stream, S2C-Merge/S2C-SAM match or outperform heavy teacher-guided methods while delivering a **3.0$\times$ reduction in loaded parameters** and a **2.5$\times$ reduction in adaptation latency**.
  Furthermore, the paper is highly sound: Proposition 4.1 mathematically formalizes and proves decision-boundary collapse under unconstrained self-supervision, elevating the paper above pure empirical heuristics. The writing is incredibly clear and polished, easily meeting top-tier standards.
* **Core Justification for Acceptance:**  
  Submission 5 addresses the most critical practical limitation of current test-time model merging frameworks (VRAM overhead). By demonstrating that online adaptation can be performed on-the-fly without keeping teacher models in GPU memory, this work bridges the gap between theoretical merging and real-world edge deployment.

---

### Submission 7: *EWC-TTA: Elastic Weight Consolidation-Guided Test-Time Adaptation for Dynamic Model Merging*
* **Overall Recommendation:** **5: Accept** (Selected as the #3 Overall Submission)
* **Key Dimensions:** Soundness: **Excellent** | Presentation: **Excellent** | Significance: **Good to Excellent** | Originality: **Good**
* **Summary of Contribution:**  
  This paper addresses two major limitations in test-time model merging: **Decision-Boundary Drift** (where unconstrained adaptation on sequential test streams destroys pre-trained boundaries) and **Autograd Graph Disconnection** (where standard weight-copying routines break PyTorch's backpropagation pathway, preventing gradient-based coefficient updates). 
  To resolve these, EWC-TTA proposes:
  1. *Stateless Differentiable Model Merging*: Uses PyTorch's modern `torch.func.functional_call` API to construct virtual merged models on the fly, maintaining a continuous, fully differentiable autograd graph.
  2. *FIM-Regularized EWC Penalty*: Incorporates a localized Elastic Weight Consolidation (EWC) style quadratic penalty scaled by a pre-computed diagonal Fisher Information Matrix (FIM) to selectively freeze task-critical classification parameters.
  The framework is highly data-efficient (requiring as few as 10 samples for the FIM prior) and can run in a **zero-source** regime using a random Gaussian noise prior. The paper also includes a novel, systematic study of task-switching frequencies (chunk sizes), uncovering a non-linear "out-of-phase" performance drop caused by temporal lag.
* **Review Content & Metareview Evaluation:**  
  The review praises EWC-TTA's exceptional methodological rigor and empirical depth. The integration of PyTorch's stateless utilities is an elegant and highly practical solution to graph disconnection. Empirically, on a sequential MNIST-FashionMNIST-KMNIST stream using ResNet-18, EWC-TTA boosts accuracy from **53.98%** (static) to **85.44%** (+31.46% absolute improvement). The temporal lag analysis provides deep, high-signal insights into dynamic scheduling. The paper is exceptionally polished, warning-free, and generalizes seamlessly to alternative merging methods like *TIES-Merging* (+30.25% boost).
* **Core Justification for Acceptance:**  
  Submission 7 is a mathematically sound, highly practical, and thoroughly evaluated contribution. It leverages modern PyTorch design patterns (stateless autograd) to solve structural merging limitations, introduces a clever data-efficient/zero-source prior regularization, and contributes a fascinating, highly practical temporal lag analysis that will benefit practitioners deploying merged models in dynamic streaming environments.

---

## 4. Why Other Strongly Rated Submissions Were Declined

To maintain a strict quota of 3 accepted papers, several excellent submissions with overall Accept (5) ratings had to be declined. Below we detail the comparative factors that positioned them below the top 3:

* **Submission 6** (*The Flatness-Geometry Paradox: Why Inverse Fisher-Weighted Perturbations Harm Model Merging*):
  - *Strengths:* Highly original and intellectually stimulating finding (the Perturbation-Sensitivity Paradox, showing that inverse-Fisher weighting backfires catastrophically because shielded sensitive parameters remain sharp). Rigorous mathematical proof of Direct F-SAM as a coordinate concentration operator.
  - *Why Declined:* While conceptually brilliant, the significance was rated "Good" rather than "Excellent" because the empirical validation is strictly limited to a smaller scale (ResNet-18 on split CIFAR-10). Compared to Submission 5 (which operates on ViTs on a non-stationary stream and solves a massive memory bottleneck) and Submission 7 (which uses modern stateless autograd and performs a detailed temporal analysis), the immediate practical impact of Submission 6 on modern large-scale applications is slightly more modest.
  
* **Submission 4** (*Fisher-Guided Orthogonal Regularization: Reconciling Flatness and Coordinate Alignment in Compatible Model Merging*):
  - *Strengths:* Elegant "FG-SPOR-Inverse" method that uses parameter sensitivity to guide regularizers, allowing sensitive parameters to adapt while constraining insensitive ones to prevent coordinate drift. Elegant layer-wise Procrustes norm analysis.
  - *Why Declined:* The empirical evaluations are conducted on a relatively small proof-of-concept scale (ResNet-18 on split CIFAR-10) with simple tasks (5 classes per task). Additionally, the method relies on a diagonal Fisher approximation which ignores parameter correlations. It lacks the immediate system-level breakthrough of Submission 5 (breaking the teacher-overhead memory barrier) or Submission 7 (differentiable stateless merging and temporal lag scheduling).

* **Submission 3** (*CAS-Merge: Covariance-Aligned Sharpness-Aware Low-Rank Model Merging*):
  - *Strengths:* Highly rigorous mathematical derivation of the non-linear discrepancy of LoRA merging; introduces a clever feature-covariance aligned projection (CALP) regularization. Exposes a critical sigmoid-parameterization initialization bias in previous TTA papers.
  - *Why Declined:* Under the corrected symmetric initialization, the actual accuracy improvements of CAS-Merge over standard baselines (like SyMerge or standard SAM) are marginal or non-existent (achieving identical average accuracies of 69.20% and 68.95%). The added complexity of tracking online layer-wise feature variances and calculating column-scaled Frobenius norms is therefore difficult to justify. Furthermore, it retains the expensive memory overhead of keeping multiple frozen expert models in memory at test time (which Submission 5 successfully eliminates).

* **Submission 8** (*Fisher-Weighted SPOR: Bridging the Flatness-Geometry Gap via Task-Specific Parameter Sensitivity for Compatible Model Merging*):
  - *Strengths:* Dynamic scaling of geometric orthogonality constraints using task-specific Fisher sensitivity. Excellent mathematical proof of selective sparsification.
  - *Why Declined:* It received a "Good" rating on Soundness (compared to "Excellent" for our top 3) because of empirical limitations: it uses a biased Fisher approximation (squaring batch-averaged gradients rather than average of squared gradients), evaluations are limited to ResNet-18 on split CIFAR-10, and it suffers from a severe accuracy collapse under global merging (OM-All, down to 55.48%) due to head-interference which is left unresolved.

---

## 5. Summary of the Meta-Review Process

The meta-review process was conducted in three logical phases:
1. **Initial Review Screening:** We collected and parsed all 10 peer review reports, extracting key ratings (Overall Recommendation, Soundness, Presentation, Significance, Originality) and identifying the core methodological contributions and empirical results.
2. **Qualitative Content Parsing:** For all papers receiving an overall Accept (5) rating, we performed a deep read of their strengths, weaknesses, and reviewer justifications. We specifically looked for papers that went beyond incremental heuristics, demonstrating high theoretical rigor, practical system-level utility, and immaculate scientific integrity (e.g., proper initialization calibration, correct baseline comparisons, and rigorous bug-free implementations).
3. **Comparative Meta-Evaluation and Selection:** We weighed the practical and theoretical impact of the top-rated papers. Submissions that solved major, long-standing structural bottlenecks (such as Submission 5's memory reduction and Submission 7's autograd graph repair) or demonstrated unparalleled empirical perfection (Submission 1's clean sweep and pristine presentation) were prioritized over papers with smaller-scale "toy" evaluations, biased curvature estimates, or marginal accuracy gains. 

The selected papers represent a beautiful balance of elegant optimization theory, state-of-the-art empirical performance, and high-impact system-level breakthroughs for deep model merging.
