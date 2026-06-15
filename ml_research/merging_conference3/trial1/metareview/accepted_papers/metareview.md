# Conference Meta-Review Process and Decisions

This document provides a comprehensive summary of the meta-review process, the individual submission evaluations, and the final decisions for selecting the top 3 papers out of the 9 available submissions.

---

## 1. Meta-Review Process Overview

The goal of this meta-review is to systematically evaluate all 9 active submissions under consideration and select exactly 3 papers for acceptance. The decision-making process was guided by the following principles:
1. **Quantitative Review Aggregation:** For each submission, the individual overall recommendations and scores of all three reviewers were compiled, and their average score was computed to establish a preliminary ranking.
2. **Qualitative Content Synthesis:** Beyond raw scores, we carefully analyzed the textual reviews of each paper—specifically balancing the listed strengths (methodological novelty, experimental rigor, empirical significance, physical profiling) against their core weaknesses (over-engineering, toy evaluations, lack of generalizability, mathematical non-equivalence, or empirical redundancy).
3. **Consensus and Impact Valuation:** We prioritized papers that represented either outstanding methodological audits (exposing false research paths and providing crucial course-corrections for the community) or genuine paradigm shifts with sound theoretical and empirical grounding.

---

## 2. Recommendation and Decision Summary Table

The table below lists all 9 submissions, the recommendations of Reviewers 1, 2, and 3, their average score, and the final meta-review decision:

| Submission | Title | Reviewer 1 | Reviewer 2 | Reviewer 3 | Avg. Score | Decision |
| :--- | :--- | :--- | :--- | :--- | :---: | :--- |
| **Submission 1** | QP-Merge: Quantization-Preserving Task Vector Merging | 3 (Weak Reject) | 3 (Weak Reject) | 3 (Weak Reject) | 3.00 | **Reject** |
| **Submission 2** | Deconstructing Sharpness-Aware Isotropic Merging | 5 (Accept) | 5 (Accept) | 6 (Strong Accept) | **5.33** | **Accept (Top 1)** |
| **Submission 7** | The Overfitting-Optimizer Paradox | 5 (Accept) | 6 (Strong Accept) | 4 (Weak Accept) | **5.00** | **Accept (Top 2)** |
| **Submission 10**| FoldMerge (Neural Origami) | 3 (Weak Reject) | 6 (Strong Accept) | 3 (Weak Reject) | **4.00** | **Accept (Top 3)** |
| **Submission 4** | FluidMerge: Fluid-Dynamic Parameter Coalescence | 4 (Weak Accept) | 4 (Weak Accept) | 3 (Weak Reject) | 3.67 | **Reject** |
| **Submission 8** | RIMO: Rigorous Isotropic Manifold Optimization | 3 (Weak Reject) | 5 (Accept) | 3 (Weak Reject) | 3.67 | **Reject** |
| **Submission 9** | Root-Mean-Square Scaling | 2 (Reject) | 5 (Accept) | 3 (Weak Reject) | 3.33 | **Reject** |
| **Submission 3** | ThermoMerge: Thermodynamic Test-Time Diffusion | 3 (Weak Reject) | 2 (Reject) | 3 (Weak Reject) | 2.67 | **Reject** |
| **Submission 6** | Winner-Take-All Sign Election | 2 (Reject) | 2 (Reject) | 2 (Reject) | 2.00 | **Reject** |

---

## 3. Justifications for Accepted Papers

### 1. Submission 2: "Deconstructing Sharpness-Aware Isotropic Merging: A Methodological Analysis of Component Contribution and Optimization Flatness"
* **Average Score:** 5.33 (Scores: 5, 5, 6)
* **Meta-Review Justification:** 
  This is an outstanding, technically flawless paper that provides exceptional impact on the model-merging community. Acting as a rigorous methodological audit, it successfully "de-bloats" the highly complex and over-engineered Sharpness-Aware Isotropic Merging (SAIM) framework. The authors demonstrate through systematic, symmetric $5 \times 3$ grid evaluations that a much simpler and unconstrained approach—standard globally perturbed SAM paired with naive Task Arithmetic—frequently matches or exceeds SAIM’s convoluted, coordinate-restricted training and expensive post-processing SVD steps. 
  
  In addition to its high empirical rigor and outstanding transparency (reproducibility across multiple random seeds), the paper features a mathematically beautiful proof (Proposition 3.1) linking optimizer-driven sharpness to post-hoc structural robustness (pruning resilience). It also identifies and corrects a fatal algebraic typo in SAIM's published formulas, and reveals a highly practical hardware-level bottleneck: coordinate-restricted optimizers are actually 18.5% *slower* in wall-clock training because sparse indexing breaks GPU parallelization and thread-coalescing. The introduction of LoRA-SAM further enhances the paper's completeness. This work represents the absolute gold standard of scientific inquiry and is a highly recommended **Strong Accept**.

### 2. Submission 7: "The Overfitting-Optimizer Paradox: A Sanity Check on Layer-Wise Model Merging Assumptions"
* **Average Score:** 5.00 (Scores: 5, 6, 4)
* **Meta-Review Justification:**
  This paper represents a timely and highly necessary "course-correction" for the model-merging community. The authors perform a meticulous, rigorous sanity check on the foundational assumption of layer-wise adaptive model-merging frameworks (e.g., AdaMerging, SyMerge)—namely, that layer-specific configs are critical to capture localized task-specific contributions.
  
  Their empirical analysis uncovers the highly significant **Overfitting-Optimizer Paradox**:
  - Under zero-order search (1+1 ES), layer specificity is revealed to be high-frequency optimization noise; replacing optimized coefficients with their simple spatial average actually regularizes training and improves average test accuracy.
  - Under first-order search (Adam GD), the highly precise layer configurations discovered by the optimizer are shown to be transductive overfitting artifacts that fail to generalize on unseen test data while multiplying seed-to-seed variance by 4x.
  
  The paper further reveals that high-level CKA representation alignment can decouple from weight-space decision boundary integrity, exposes a systematic joint-entropy optimization task-bias that sacrifices complex tasks, and proposes an elegant Proximity Regularization solution to prevent transductive drift. Despite evaluations being constrained to a CLIP ViT-B/32 backbone, the scientific depth and critical sanity checks of this paper are of immense value to the community.

### 3. Submission 10: "FoldMerge (Neural Origami): Continuous Non-Linear Coordinate Warping via Learned Diffeomorphisms"
* **Average Score:** 4.00 (Scores: 3, 6, 3)
* **Meta-Review Justification:**
  While split and controversial, this paper represents an exceptionally high-risk, high-reward paradigm shift that was accepted on the basis of its immense conceptual novelty. The authors break away from the dominant "Euclidean flat-space" paradigm of model merging, arguing that straight-line interpolation across non-convex loss landscapes traverses high-loss barriers. They propose to model merging as a continuous weight-space warping process, learning a smooth, continuous, bijective weight-space diffeomorphism (Normalizing Flow) via 4 RealNVP affine coupling layers.
  
  The paper goes far beyond standard empirical tuning: it provides advanced mathematical formulations like **Latent Task Vector Warping** to prevent scale distortion, introduces **LoRA-Flow** to compress trainable weights by $27\times$ while improving accuracy, and employs an elegant parameter-wise $\ell_2$ regularization to anchor the flow to a local perturbation around the identity mapping, bypassing expensive Jacobian calculations. Crucially, because the learned warping is decoded back to weight space, it incurs **zero deployment or inference overhead**. 
  
  Reviewers 1 and 3 raised reasonable concerns about over-engineering, the empirical scale (visual projection layer), and marginal gains over linear SyMerge. However, their concerns are outweighed by the paper's rigorous scientific honesty (including a full **Frozen Classifier Head Ablation** to isolate representation alignment) and its potential to open up a major, highly creative research avenue bridging differential geometry, normalizing flows, and weight-space optimization. It represents exactly the type of bold, paradigm-shifting idea that conferences should champion.

---

## 4. Justifications for Rejected Papers

### 1. Submission 4: "FluidMerge: Fluid-Dynamic Parameter Coalescence"
* **Average Score:** 3.67 (Scores: 4, 4, 3)
* **Meta-Review Justification:**
  While the continuous fluid-dynamic framing (advection-diffusion ODE) is highly creative and engaging, the paper suffers from severe empirical redundancy. The reviewers pointed out a fundamental methodological isomorphism: under Euler discretization, FluidMerge mathematically reduces to standard gradient descent with Elastic Weight Consolidation (EWC) regularization. This means the complex fluid-dynamics framing operates as a metaphorical repackaging of well-established, standard techniques rather than introducing a genuinely novel physical execution mechanism. Given this redundancy and the restricted evaluation scale (ViT-B/32 on digit datasets), the paper was rejected.

### 2. Submission 8: "RIMO: Rigorous Isotropic Manifold Optimization for Model Merging"
* **Average Score:** 3.67 (Scores: 3, 5, 3)
* **Meta-Review Justification:**
  This paper proposes a rigorous mathematical analysis of model merging on non-linear parameter manifolds (such as the orthogonal group $\mathrm{O}(d)$ and Lie algebra tangent space $\mathfrak{so}(d)$). It uncovers significant spectral and coordinate gauge distortion phenomena under post-hoc SVD projection, and proposes symmetry-preserving mitigations (RIMO-Schur, RIMO-Complex). However, the paper is hindered by extremely fragile assumptions: the manifold-level merging is highly sensitive to soft orthogonal regularization during training; without it, performance collapses catastrophically (from 84.55% to 42.07%). Additionally, there remains a persistent performance gap where even the optimized RIMO models fail to outperform flat Task Arithmetic. Due to these severe practical limitations and the toy-scale evaluation, the paper was rejected.

### 3. Submission 9: "Root-Mean-Square Scaling: Unifying Model Merging via Minimalist Scale Calibration"
* **Average Score:** 3.33 (Scores: 2, 5, 3)
* **Meta-Review Justification:**
  This paper introduces a minimalist, training-free and data-free layer-wise scale calibration method (RMS-Scale and PF-RMS) running in linear time. While the mathematical derivation linking element-wise RMS scaling to parameter-count-scaled Frobenius-norm normalization is highly elegant and computationally efficient, the empirical evaluation is fundamentally weak. The primary benchmark is restricted to a toy-scale 500k-parameter CNN, the accuracy improvements over baselines are statistically insignificant due to heavily overlapping standard deviations, and the CLIP foundation model evaluation is based purely on simulated updates with zero downstream task classification metrics. Furthermore, there are significant gaps in literature context (omitting relevant layer-wise scaling baselines like LARV and MAGIC). It was therefore rejected.

### 4. Submission 1: "QP-Merge: Quantization-Preserving Task Vector Merging"
* **Average Score:** 3.00 (Scores: 3, 3, 3)
* **Meta-Review Justification:**
  QP-Merge targets the co-design of post-training quantization (PTQ) and model merging via Outlier-Residual Decoupling (ORD) and scale calibration (QE-Calib). Despite the practical relevance of the problem, the submission was unanimously rejected due to critical flaws: (1) an extremely weak empirical evaluation restricted to toy digit classification tasks (MNIST/SVHN) using a massive, overpowered ViT-B-32 model; (2) severe empirical redundancy in the core architectural contribution, where the complex dense-quantized + sparse-unquantized ORD path adds massive execution complexity and a 6x GPU latency slowdown while yielding a statistically negligible $0.03\%$ average accuracy benefit over a standard homogeneous INT4 model; and (3) methodological misdirection in QE-Calib, which lacks mathematical equivalence and functions as unsupervised scale fine-tuning with 55k learnable parameters on 128 samples, exhibiting severe domain-overfitting.

### 5. Submission 3: "ThermoMerge: Thermodynamic Test-Time Diffusion for Synergistic Model Merging"
* **Average Score:** 2.67 (Scores: 3, 2, 3)
* **Meta-Review Justification:**
  ThermoMerge introduces a thermodynamic, physics-inspired test-time adaptation framework utilizing Stochastic Gradient Langevin Dynamics (SGLD) with Simulated Annealing to escape local minima in joint-task landscapes. The framework was rejected due to substantial weaknesses: (1) the evaluation is restricted entirely to toy-scale MLPs and digit datasets (MNIST/FashionMNIST/KMNIST); (2) the empirical performance gains are statistically weak or negative compared to simple deterministic optimizers and static task arithmetic in 10 out of 12 evaluated configurations; and (3) the system is heavily over-engineered, relying on an accumulation of nested thresholds and emergency quenching rollbacks that make it brittle and impractical for real-world production.

### 6. Submission 6: "Winner-Take-All Sign Election: A Minimalist Approach to Model Merging"
* **Average Score:** 2.00 (Scores: 2, 2, 2)
* **Meta-Review Justification:**
  This paper proposes a hyperparameter-free merging method that replaces sign-voting consensus with a "Winner-Take-All" magnitude-driven sign election. It was unanimously rejected due to several critical issues: (1) it is a highly incremental variation of TIES-Merging with no theoretical paradigm shift or error bounds; (2) it suffers from extreme scale sensitivity due to a complete lack of task-vector normalization; and (3) empirically, the oligarchic Winner-Take-All selection rules cause severe representation destruction, leading to performance that is worse than simple, unoptimized Task Arithmetic on 2 out of the 3 evaluated datasets (SVHN and CIFAR-10).
