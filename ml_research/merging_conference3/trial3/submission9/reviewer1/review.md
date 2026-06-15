# Peer Review of FlatQ-Merge

## 1. Summary of the Paper
This paper presents **FlatQ-Merge** (Flatness-Aware Quantization-Aware Model Merging), a comprehensive empirical study and framework analyzing the relationship between the geometric loss landscape flatness of pre-merging task-specific expert models and their subsequent resilience to post-training quantization (PTQ) and test-time coefficient adaptation. 

The authors pre-train task-specific experts on a Vision Transformer (`vit_tiny_patch16_224`) backbone across five different Sharpness-Aware Minimization (SAM) perturbation radii ($\rho \in \{0.0, 0.01, 0.05, 0.1, 0.2\}$). They then merge and compress these experts using per-channel symmetric uniform PTQ (8-bit and 4-bit), and optimize layer-wise merging coefficients $\Lambda \in [0, 1]^{14 \times 4}$ using joint prediction entropy minimization on a small, unlabeled calibration dataset with Straight-Through Estimator (STE) gradients. 

The paper uncovers several impactful insights: (1) a precision-dependent flatness-robustness synergy that peaks at $\rho=0.05$ under extreme 4-bit quantization, yielding up to **+7.44%** absolute multi-task accuracy improvements over standard sharp SGD experts; (2) that pre-merging expert geometry (flatness) dominates downstream test-time adaptation, with a simple, zero-adaptation uniform merge on flat experts (NaiveUniform) outperforming a highly sophisticated adaptation on sharp SGD experts by **+6.03%** absolute accuracy; (3) a non-linear "over-perturbation threshold" at $\rho \ge 0.1$ where performance collapses due to representation convergence (task vectors becoming highly correlated); and (4) that SAM's adversarial formulation is a necessary ingredient for 4-bit noise resilience, whereas passive trajectory averaging (SWA) fails under extreme compression.

---

## 2. Strengths and Weaknesses

### Strengths
* **High-Impact, Paradigm-Shifting Insights:** The core discovery that pre-merging weight-space geometry (flatness) dominates downstream test-time adaptation is profound and highly original. It reframes the priorities of the model merging field, shifting the focus from designing increasingly complex blending optimization algorithms on the test set to preparing robust pre-merging parameter geometry during fine-tuning.
* **Outstanding Scientific and Empirical Rigor:** The paper is exceptionally thorough. Rather than simply presenting a combined pipeline, the authors systematically deconstruct their framework. They conduct an elegant SWA vs. SAM comparison (Section 4.8) to isolate the necessity of adversarial training; perform task vector correlation analysis (Section 4.4-B) to explain the over-perturbation collapse; validate independent clipping bounds over Softmax normalization (Section 4.5); and ablate low-dimensional coefficient adaptation against full-parameter TENT adaptation to validate their implicit regularization hypothesis (Section 4.7).
* **Direct Curvature Measurement (Closing the Theoretical Loop):** In Section 4.9, the authors directly perturb the weights of the experts to empirically proxy the weight-space Hessian trace. Showing an $8\times$ reduction in sharpness/curvature for SAM experts ($\rho=0.05$) compared to SGD experts, and demonstrating a perfect correlation between low curvature and downstream 4-bit merging resilience, is an exemplary piece of empirical validation that beautifully supports the paper's second-order Taylor expansion foundations.
* **Intellectual Honesty and Transparency:** The authors provide a highly detailed, comprehensive Limitations section. They address the absolute accuracy gap caused by their low data budget, analyze the systems-level peak memory advantages of Direct Quantized Adaptation, discuss the extension to joint weight-activation quantization, and theoretically prove how flat pre-training is expected to suppress activation outliers by bounding Lipschitz constants.
* **Exceptional Presentation and Clarity:** The writing is professional, mathematically precise, and easy to follow. Figure 1 is highly effective, and the inclusion of full hyperparameters (Table 3) and pseudocode (Algorithm 1) makes the work completely reproducible.

### Weaknesses
* **Incremental Methodological Novelty:** From a pure algorithmic engineering perspective, the submission does not introduce a fundamentally new mathematical optimizer or blending method. "FlatQ-Merge" is a straightforward concatenation of two existing building blocks: pre-training task-specific experts with SAM (from SAFT-Merge, ICLR 2025) and optimizing layer-wise merging coefficients under PTQ via STE (from Q-Merge, 2026). The test-time adaptation itself employs a standard prediction entropy minimization objective. The algorithmic novelty is therefore moderate, representing a straightforward block combination.
* **Scale and Absolute Accuracy Constraints:** The experiments are restricted to a tiny Vision Transformer (`vit_tiny_patch16_224`) pre-trained on a small budget of 512 images per task. Consequently, absolute accuracies are degraded compared to full-scale converged models (e.g., individual unquantized experts achieve ~64% average accuracy, and the merged 4-bit models achieve ~30%). While this is acceptable as a controlled "empirical sandbox" for massive grid sweeps, validation on a larger-scale benchmark (e.g., LLaMA or ViT-Base on larger datasets) would have strengthened the practical impact of the work.

---

## 3. Ratings

### Soundness: Excellent
The submission is technically flawless. The theoretical formulation (second-order Taylor expansion and Hessian projection proof) is mathematically sound, and the empirical validations are exceptionally rigorous. The authors successfully address key technical challenges, including the STE gradient mismatch and the threat of degenerate class collapse, with solid empirical and structural arguments. The direct weight-space curvature measurements in Section 4.9 provide definitive proof of their physical claims.

### Presentation: Excellent
The paper is exceptionally well-written, clearly structured, and easy to follow. Key insights are highlighted in bullet points, and the figures are highly polished and informative. The authors' discussion of mathematical properties (such as why negative expected entropy changes imply saddle-point dominance on the quantized plateau) is highly rigorous and educational.

### Significance: Excellent
The paper addresses an important and highly practical problem: the deployment of parameter-fused, multi-task models on resource-constrained edge hardware. By demonstrating a data-efficient (16 calibration images per task) and systems-friendly (8$\times$ peak memory reduction) path to 4-bit model merging, the work has high practical significance. Furthermore, its paradigm-shifting conceptual insights have the potential to influence future research directions in model merging.

### Originality: Good
The algorithmic novelty is moderate, as the methodology consists of a straightforward block-concatenation of SAFT-Merge and Q-Merge. However, the conceptual and empirical originality is outstanding. The systematic comparison between SAM and SWA, the representation convergence analysis at the over-perturbation threshold, and the direct weight-space flatness profiling provide highly original and novel scientific discoveries that are of significant interest to the community.

---

## 4. Overall Recommendation

**5: Accept**

**Justification:** 
This is an outstanding, highly rigorous, and exceptionally well-written paper. While the pure algorithmic engineering novelty is somewhat incremental (representing a straightforward combination of SAM pre-training and quantization-aware merging coefficients), the paper's scientific discoveries and conceptual insights are highly original and paradigm-shifting. 

The empirical proof that pre-merging weight-space geometry (flatness) dominates downstream test-time adaptation is a major contribution that will simplify edge-deployment pipelines and reframe the research priorities of the model merging field. 

Supported by exhaustive baseline comparisons (including SWA, TENT, Softmax, and DARE), statistical rigor across 3 independent seeds, and direct weight-space curvature measurements, the authors' claims are technically flawless and thoroughly supported. The paper sets an exemplary standard for empirical validation and is a highly valuable contribution to the fields of model merging and model compression.
