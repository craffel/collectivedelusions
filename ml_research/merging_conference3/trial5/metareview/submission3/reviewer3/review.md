# Peer Review

## 1. Summary of the Paper
This paper presents a diagnostic evaluation and stabilization framework for dynamic model merging (parameter fusion) under out-of-distribution (OOD) shift and heterogeneous test streams. The authors critically re-evaluate a recently proposed, complex dynamic merging framework: Quantum Wavefunction Superposition Merging (QWS-Merge; Vance et al., 2025). QWS-Merge introduced convoluted, quantum-inspired mechanisms under the premise that classical linear routers collapse catastrophically on high-variance domains like SVHN (collapsing to $15.30\%$ accuracy).

Using the guiding principle of Occam's razor, this paper deconstructs that assumption. The authors demonstrate that the reported collapse of classical routing is not a fundamental structural limitation, but rather an artifact of sub-optimal optimization choices (routing from deep task-warped layers, using excessively high learning rates, and over-optimizing for too many steps on a tiny calibration set). They show that a properly configured unregularized classical Linear Router is highly robust, achieving an average Joint Mean accuracy of $91.53\% \pm 0.41\%$ across 5 random calibration seeds.

To handle OOD shifts and heterogeneous serving streams (where batch-level coefficient averaging causes performance degradation), the authors propose **Robust Linear Routing (RLR)**. RLR regularizes a simple 768-parameter classical linear gating layer using two standard techniques: $L_2$ weight decay (Frobenius norm penalty) and Softmax Temperature scaling ($T \ge 1$). While RLR is statistically indistinguishable from unregularized routing under homogeneous conditions, it serves as a specialized stabilizer in heterogeneous streams, consistently maintaining a modest accuracy advantage (e.g., $+1.88\%$ absolute benefit at batch size $B=256$) over the unregularized router. RLR is highly parameter-efficient, calibrates in under a second, and introduces zero runtime overhead.

---

## 2. Strengths and Weaknesses

### Strengths
1. **Outstanding Scientific Rigor and Transparency**: The authors local re-implementation of QWS-Merge under identical conditions is highly commendable. This ensures a fair, unified benchmark, removing any confounding checkpoint differences. Furthermore, evaluating the routing models across 5 random calibration seeds is a standard of statistical rigor too often omitted in modern machine learning papers.
2. **Adherence to Occam's Razor**: The paper does an excellent job of advocating for simplicity, reminding the deep learning community that simple baselines must be thoroughly understood and regularized before introducing complex, over-engineered architectures.
3. **Valuable Diagnostic Analysis**: The systematic diagnostic comparison (Table 2) identifies the exact configuration choices that trigger the SVHN collapse, providing a useful, transparent recipe for future researchers to configure stable linear routers.
4. **Excellent Writing and Presentation**: The paper is exceptionally well-written, engaging, and clear. The figures and tables are informative and support the text perfectly.

### Weaknesses
1. **Very Low Algorithmic/Methodological Novelty**:
   While the diagnostic and deconstructive portion of the paper is highly valuable, the constructive contribution—Robust Linear Routing (RLR)—lacks conceptual novelty or ambitious algorithmic design. Applying $L_2$ weight decay and Softmax Temperature scaling to a gating layer is a standard, decades-old practice in machine learning (ubiquitous in Mixture-of-Experts routing, classification heads, and temperature scaling in calibration). The combination of these techniques does not represent a conceptual leap or a bold, paradigm-shifting methodology. It is a highly defensive and straightforward application of common deep learning tools.
2. **Limited Experimental Scale**:
   The empirical validation is conducted entirely on a compact Vision Transformer backbone (`vit_tiny_patch16_224`) on small-scale image classification tasks (MNIST, FashionMNIST, CIFAR-10, SVHN). Since model merging is primarily motivated by reducing deployment costs for massive foundation models, validating this approach on Large Language Models (LLMs) or large-scale multimodal models would significantly increase the paper's significance. The LLM scaling pathways proposed in Section 5 are purely theoretical and remain unverified.
3. **Failure to Structurally Resolve Heterogeneity Collapse**:
   Both RLR and the unregularized Linear Router experience severe performance degradation under mixed-task heterogeneous serving as the evaluation batch size increases (Table 4). At $B=256$, RLR's accuracy drops to $75.03\%$. While RLR outperforms the unregularized router, it fails to compete with the static supervised OFS-Tune baseline ($86.23\%$). The proposed regularization is only a minor, defensive patch rather than a major architectural or structural solution to this fundamental limitation of dynamic model merging.

---

## 3. Soundness
**Rating**: **Excellent**

**Justification**:
The paper is technically flawless and methodologically rigorous. The authors' hypotheses are clearly stated and thoroughly tested. The local re-implementation of QWS-Merge, the 5-seed random calibration sweep, and the 2D hyperparameter sensitivity sweep provide absolute clarity and confidence in the empirical results. The authors are refreshingly honest and transparent about their findings, openly acknowledging that RLR is statistically indistinguishable from the unregularized classical router in homogeneous settings, and acknowledging the superiority of static methods in large heterogeneous batches.

---

## 4. Presentation
**Rating**: **Excellent**

**Justification**:
The manuscript is beautifully written, exceptionally well-structured, and easy to follow. The mathematical notation is clean and standard. The figures (Figures 1, 2, and 3) are professional, self-contained, and highly effective at conveying the key results and sensitivity sweeps.

---

## 5. Significance
**Rating**: **Good**

**Justification**:
The significance of the paper is primarily **reductive and diagnostic** rather than constructive. By successfully demystifying the reported SVHN collapse and debunking the necessity of convoluted, quantum-inspired frameworks like QWS-Merge, the paper performs a highly valuable service to the community. It has the potential to steer model merging research back toward elegant, robust, and transparent classical baselines. However, the significance of the proposed RLR method itself is limited, as it represents a defensive baseline correction rather than a major constructive paradigm that researchers are likely to build upon.

---

## 6. Originality
**Rating**: **Fair**

**Justification**:
The paper's originality is highly asymmetric. The diagnostic insight (deconstructing the SVHN collapse) and the systemic configuration analysis are original and highly valuable. However, the proposed method (RLR) has very low originality. The application of standard $L_2$ weight regularization and softmax temperature scaling to a linear projection layer is a standard practice in deep learning and lacks any conceptual or mathematical leaps.

---

## 7. Overall Recommendation
**Rating**: **4: Weak Accept**

**Justification**:
This is a technically solid, exceptionally well-written, and scientifically rigorous paper that performs a highly valuable "sanity check" service for the model merging community. Its deconstruction of QWS-Merge's reported SVHN collapse is convincing and statistically robust, effectively advocating for Occam's razor.

However, from the perspective of conceptual and algorithmic novelty, the contribution is highly incremental. The proposed RLR algorithm is a straightforward application of standard regularization tools ($L_2$ weight decay and Softmax Temperature scaling) to a standard linear gating layer. It lacks the bold, paradigm-shifting originality or the conceptual ambition expected of a major new method. Additionally, the experimental scope is confined to a compact ViT-Tiny model, leaving its performance on modern Large Language Models unverified. Finally, the proposed method does not structurally resolve the core challenge of heterogeneity collapse under large batch sizes, where static methods remain vastly superior.

Overall, the merits of the diagnostic deconstruction outweigh the incremental nature of the proposed solution, making this paper a valuable contribution to the community, albeit one with limited methodological novelty. I recommend a Weak Accept.

---

## 8. Constructive Feedback & Questions for the Authors

1. **Empirical Validation on Large Models / LLMs**: 
   Since model merging is highly relevant to massive models, do you have any preliminary empirical results scaling RLR to modern LLM experts (e.g., merging specialized LoRAs of LLaMA or Mistral)? Demonstrating that RLR's stabilizing effects hold in high-dimensional language representations would significantly strengthen the paper's impact.
2. **Structural Mitigation of Heterogeneity Collapse**: 
   In Section 4.4, you provide a useful design guideline proposing a "lightweight, zero-shot pre-sorting layer" in front of the dynamic model to group heterogeneous queries. Have you experimented with or implemented a simple version of this pre-sorting layer? If so, what are the empirical results, and does it successfully eliminate the heterogeneity collapse?
3. **Sensitivity to Calibration Dataset Size**: 
   How does RLR perform as the few-shot calibration dataset size increases (e.g., from 16 samples per task to 64 or 256)? Does the unregularized classical router eventually become robust on its own without requiring $L_2$ weight decay or temperature scaling under deep-layer routing, or is regularization always necessary when extracting representations from deeper layers?
