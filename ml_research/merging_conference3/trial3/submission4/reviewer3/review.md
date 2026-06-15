# Peer Review for Conference Submission

**Title:** ZipMerge: Joint Weight Pruning and Test-Time Coefficient Tuning for On-Device Model Merging

---

## 1. Paper Summary

This paper investigates the highly practical challenges of deploying merged multi-task models onto resource-constrained edge hardware. While model merging (e.g., via task arithmetic) is a powerful paradigm for combining specialized expert models without training overhead or access to full datasets, the resulting model is fully dense, creating a major physical memory bottleneck. 

To resolve this, the paper proposes and analyzes **ZipMerge**, a framework designed to co-optimize layer-wise merging coefficients and binary magnitude-pruning boundaries at test-time under an unsupervised minimum entropy objective on tiny calibration sets (16 images per task). It evaluates two optimization engines: first-order gradient descent via a Straight-Through Estimator (STE) and zero-order search via a 1+1 Evolution Strategy (1+1 ES).

Rather than presenting a curated success story, the authors conduct a highly rigorous, honest empirical study under extreme conditions (merging highly orthogonal visual classification tasks onto a compact 5.7M parameter ViT-Tiny backbone). The experiments expose severe empirical boundaries, including:
1. **Catastrophic Representational Collapse:** Under extreme task conflict, all merged configurations suffer from complete representation collapse, performing near the random guessing baseline (~10% to 14% accuracy).
2. **The Overfitting-Optimizer Paradox:** Unconstrained minimum-entropy TTA on tiny calibration sets overfits transductively, minimizing calibration entropy while destroying generalizable features and driving test-set accuracy down.
3. **Prune-then-Merge (P-then-M) Outperformance:** A simple, unoptimized decoupled baseline (pruning individual task vectors *prior* to merging) consistently outperforms joint test-time optimization because pre-merging pruning acts as a spatial regularizer that removes conflicting parameter noise.

The authors translate these boundary failures into concrete, actionable architectural guidelines for edge deployment. Crucially, they propose and empirically validate **Orthogonal Procrustes SVD Alignment** for low-rank (LoRA) adapters, demonstrating that rotating separately learned adapter coordinate spaces *post-hoc* before averaging dramatically boosts Joint Mean accuracy from 42.30% to 58.75% (a massive **+16.45%** absolute improvement) with zero data and negligible sub-millisecond CPU overhead.

---

## 2. Strengths and Weaknesses

### Strengths
1. **Exceptional Empirical Honesty and Rigor:** The paper is a breath of fresh air. Instead of masking failures or presenting selective positive results, the authors actively expose the fundamental limits of linear weight-space operations and adaptive test-time compression. This provides enormous educational and practical value to the community.
2. **Actionable Systems-Level Insights:** The paper provides highly concrete, practical recipes for edge-device deployment. Identifying the unoptimized **Prune-then-Merge (P-then-M)** baseline as a robust spatial regularizer and proving that **Orthogonal Procrustes SVD Alignment** analytically rotates PEFT adapter spaces to resolve coordinate mismatches with zero data and sub-millisecond CPU overhead are outstanding, high-yield system contributions.
3. **Exhaustive Analytical Depth:** The authors validate their findings through an extensive array of auxiliary studies:
   - Scaling up model capacity to **ViT-Base** (86M parameters).
   - Verifying architectural diversity on a CNN backbone (**ResNet-18**).
   - Isolating task conflict using a domain-aligned Visual Suite (**DomainNet**).
   - Performing extensive hyperparameter sweeps (learning rate sensitivity, regularization weights $\gamma$, distillation scale $\beta$, and LoRA rank $r$).
   - Statistical checking across 5 independent seeds with extremely low variance ($\pm 0.32\%$), proving robustness.
4. **Deep Geometric Analysis of Optimizers:** The explanation of why zero-order search (1+1 ES) is superior at moderate (50%) sparsity due to bypassing gradient-approximation noise, whereas first-order gradient descent (STE) is superior at high (80%) sparsity due to focused, variance-reduced active paths, is incredibly insightful and mathematically sound.

### Weaknesses
1. **Implicit SVHN Expert Limitation:** The SVHN expert achieves only 19.59% accuracy. While the authors thoroughly analyze this as "The Noisy Expert Noise Injection Constraint" (Section 4.2.4), the fact that the SVHN expert is poorly converged should be explicitly noted in the introduction or Table 1 caption to immediately clarify the baseline context for readers.
2. **Preliminary Empirical Verification for Theoretical Extensions:** The sections on Joint PTQ-Pruning and Scheduled Pruning are mathematically elegant and promising. However, they are currently reserved as purely theoretical proposals. Including even a small, preliminary test of scheduled pruning (e.g., linear vs. cubic schedule on ViT-Tiny at 50% sparsity) would significantly strengthen these sections.
3. **Missing Quantitative Profiling of SVD Runtime:** The paper notes that SVD-based Orthogonal Procrustes has "negligible sub-millisecond overhead." To further support its practical edge utility, providing an explicit runtime profile (e.g., actual execution time in milliseconds on a standard CPU vs. standard GPU) would be highly valuable for systems engineers.

---

## 3. Detailed Review

### Soundness
**Rating: Excellent**

The submission is technically flawless and methodologically exceptional:
- Every step of the co-optimization framework (ZipMerge), the regularized variants (Reg-ZipMerge), and the SVD alignment are mathematically formulated and clearly laid out.
- The use of realistic edge backbones (ViT-Tiny and ResNet-18) is highly appropriate.
- The experiments are well-designed, evaluating under both high-conflict (MNIST/FashionMNIST/CIFAR/SVHN) and low-conflict (DomainNet) regimes.
- The authors are highly transparent about their limitations and the convergence quality of their input experts (SVHN).
- The low variance ($\pm 0.32\%$) across independent seeds confirms the findings are robust and reproducible.

### Presentation
**Rating: Excellent**

The paper is exceptionally clearly written, well-structured, and easy to follow:
- The narrative flows logically from motivation and formulation to boundary mapping, and finally to highly creative and practical solutions.
- Algorithm 1 is a model of clarity, tracing the entire dual-engine co-optimization flow.
- The discussion of results is critical and analytical rather than defensive.
- The paper properly positions itself in the context of prior/concurrent literature (such as AdaMerging, TIES-Merging, and test-time adaptation).

### Significance
**Rating: Excellent**

This paper addresses a vital, real-world bottleneck: deploying multi-task models onto memory-constrained edge hardware. 
- It advances understanding by re-anchoring the model merging field to physical system limits, proving that unconstrained test-time adaptation can destroy generalization.
- It provides immediate, actionable utility to practitioners. Composing multi-task networks using P-then-M or SVD-based Orthogonal Procrustes represents a massive leap forward in training-free edge-device deployment.
- The scope of impact is broad, showing that the findings generalize across both CNN and Transformer backbones.

### Originality
**Rating: Excellent**

The paper provides significant original insights and highly creative solutions:
- Co-optimizing continuous merging coefficients and binary magnitude-pruning boundaries simultaneously at test-time is highly novel.
- The derivation and empirical validation of Orthogonal Procrustes SVD Alignment for LoRA adapters represents a major mathematical and empirical step forward. Unlike prior methods that require data or training to align representations, Procrustes resolves the coordinate basis mismatch *analytically* in a training-free manner, boosting Joint Mean accuracy by **+16.45%** absolute.
- The geometric analysis of optimizer trajectories (1+1 ES vs. STE) under different sparsity regimes provides novel, deep physical understanding.

---

## 4. Overall Recommendation

**Rating: 6 (Strong Accept)**

**Justification:**
This is an outstanding, technically flawless paper that has exceptional impact on the field of deep learning efficiency and on-device model composition. It stands out for its scientific honesty and rigorous post-mortem analysis. By thoroughly mapping out representational collapse and the Overfitting-Optimizer Paradox under extreme domain shift, the paper provides immense value to researchers and practitioners alike. Furthermore, it goes far beyond a typical failure analysis by proposing and empirically validating highly creative, computationally efficient, and data-free solutions (such as Prune-then-Merge as a spatial regularizer and Orthogonal Procrustes SVD Alignment for PEFT adapters). The paper is incredibly thorough, boasts excellent presentation and mathematical formulations, and has exceptionally strong evaluation, reproducibility, and open-source resources.

---

## 5. Questions & Constructive Suggestions for the Authors

1. **Baseline Contextualization:** Could you please explicitly note in the Introduction or Table 1 caption that the SVHN expert baseline is poorly converged (19.59% accuracy) to immediately prepare the reader for its "poison pill" behavior during full-backbone merging?
2. **Preliminary Scheduled Pruning Check:** If feasible, could you provide a preliminary empirical check comparing a simple linear schedule versus the proposed cubic schedule under ZipMerge-ES at 50% sparsity on the ViT-Tiny suite? Even a small, five-step approximation would provide valuable support for this promising theoretical section.
3. **Runtime Profiling:** Could you include a small, quantitative table profiling the actual execution latency of the SVD-based Orthogonal Procrustes alignment (e.g., in milliseconds on a standard edge-device CPU) to further solidify its negligible-overhead claim?
