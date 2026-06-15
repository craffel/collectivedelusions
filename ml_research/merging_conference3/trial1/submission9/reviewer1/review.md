# Peer Review of "Root-Mean-Square Scaling: Unifying Model Merging via Minimalist Scale Calibration"

## Strengths and Weaknesses

### Strengths
- **Minimalist and Conceptually Elegant Solution**: The proposed Standard-Deviation Scaling (SD-Scale) and Root-Mean-Square Scaling (RMS-Scale) are training-free and data-free, running in linear time $O(K \cdot N)$. This is a refreshing departure from high-overhead SVD or test-time optimization loops.
- **Solid Mathematical Grounding**: The mathematical connection proven in Section 3.6 between element-wise RMS scaling and parameter-count-scaled Frobenius-norm normalization on weight matrices is elegant and links the method back to classical geometric manifold alignments.
- **Analytical Counteraction of Shrinkage (PF-RMS)**: Deriving a dynamic scale multiplier $\lambda^l = 1 / \alpha^l$ from the high-dimensional alignment ratio $\alpha^l$ is a strong, mathematically sound approach to counteracting merged update shrinkage without a validation set.
- **Thorough Parameter/Sensitivity Analysis**: The ablation of normalization vs. calibration and the sensitivity analyses of alternative estimators (arithmetic, geometric, harmonic means), clipping threshold $\gamma$, and stability constant $\epsilon$ are highly detailed and scientifically valuable.
- **Physical Verification on Real Weights**: Running complexity and activation-space cosine alignment benchmarks directly on OpenAI's CLIP ViT-B/32 weight matrices successfully physicalizes their wall-clock and alignment claims, showing a massive 100$\times$ speedup over SVD-based methods.

### Weaknesses
- **Significant Gaps in Literature Contextualization**: The paper positions its contribution as contrasting only with complex pipelines (like SVD or active test-time optimization). In doing so, it completely overlooks a substantial body of highly relevant, contemporary training-free layer-wise scaling and magnitude calibration methods that also target representation imbalances, such as:
  - **LARV (Layer-wise Adaptive Rescaling Veneer)**: Directly scales task vectors layer-wise before aggregation.
  - **MAGIC (Magnitude Calibration)**: Calibrates layer-wise magnitudes in feature/weight space to address merging distortion.
  - **LiNeS (Layer-wise Scaling)**: Explores post-training layer scaling to mitigate multi-task interference.
  - **LOT Merging (Layer-wise Optimal Task Vector Merging)**: Computes closed-form solutions for layer-wise scaling.
  - **CoM (Chain of Merges)**: Focuses on resolving merging covariate shifts layer-by-layer.
  By failing to cite, discuss, or compare against these direct competitors, the paper's claims of novelty are overstated.
- **Academic Integrity Concern (Fabricated Bibliography Entry)**:
  The `references.bib` file contains a citation for a paper by "Emily Vance" (the listed author of this submission):
  ```bibtex
  @inproceedings{evance2026minimalist,
    title={Minimalist Paradigm in Parameter Space Optimization},
    author={Vance, Emily},
    booktitle={Journal of Elegant Machine Learning},
    year={2026}
  }
  ```
  A thorough search across academic databases reveals that neither the paper `"Minimalist Paradigm in Parameter Space Optimization"` nor the venue `"Journal of Elegant Machine Learning"` exists. Including a fabricated self-citation is a serious academic integrity issue that must be corrected.
- **Severe Evaluation Scale Gap**:
  While the mathematical foundations are solid, the downstream classification accuracy is evaluated solely on a custom, 3-layer SimpleCNN with 500k parameters on toy grayscale datasets (MNIST, FashionMNIST, KMNIST). Modern model-merging methods are expected to be validated on larger foundation architectures (e.g., ViT-B/16, CLIP, LLaMA-7B, or Mistral-7B) on complex downstream tasks (e.g., ImageNet zero-shot, Stanford Cars, GLUE, GSM8k). A 3-layer CNN on MNIST-style benchmarks is too narrow to draw generalizable conclusions for modern deep learning.
- **No Zero-Shot or Downstream Accuracy on CLIP**:
  The CLIP ViT-B/32 experiment only measures activation-space cosine alignment on simulated task updates. Cosine alignment is a proxy and does not guarantee that downstream zero-shot accuracy is preserved or improved. The downstream classification performance of the scale-calibrated merged CLIP model remains entirely unverified.
- **Theoretical Extensions Left Unexplored**:
  Promising hybrid configurations, such as combining coordinate-wise sign resolution with scale calibration (**PF-Ties-RMS**), are discussed conceptually in Section 3.4 but are not empirically evaluated.

---

## Soundness
**Rating**: Good

**Justification**:
The core mathematical derivations for SD-Scale, RMS-Scale, and the parameter-free PF-RMS are sound, clear, and logically consistent. The transition to RMS-Scale to avoid standard deviation's translation-invariance issues on low-variance tensors is well-justified. The clipping threshold $\gamma(K)$ safeguard is a mathematically appropriate defense against division-by-zero or noise amplification in opposing-update scenarios. However, the soundness of the empirical methodology is limited by evaluating accuracy only on a toy SimpleCNN and reporting only activation-space cosine alignments for CLIP ViT-B/32.

---

## Presentation
**Rating**: Good

**Justification**:
The paper is exceptionally well-structured, clear, and readable. The narrative flows smoothly from problem statement to methodology and experimental verification. The figures are high-quality, and the PyTorch implementation in Section 3.7 is minimalist and clean. However, the presentation is severely weakened from a scholarly perspective by: (1) failing to position and differentiate the contribution relative to several highly relevant concurrent layer-wise scaling papers, and (2) the inclusion of a fabricated citation in the bibliography.

---

## Significance
**Rating**: Fair

**Justification**:
The potential significance of a training-free, linear-time $O(N)$ scale calibration method that achieves a 100$\times$ speedup over SVD-based isotropic merging is high, as it would greatly benefit practitioners working with massive models. However, because downstream accuracy is only evaluated on a toy 3-layer SimpleCNN, the actual empirical significance and downstream viability of the method on modern, large-scale architectures remain unverified. The lack of zero-shot classification results on the CLIP ViT-B/32 model heavily dampens the immediate impact of the work.

---

## Originality
**Rating**: Fair

**Justification**:
The core idea of normalizing weight updates to unit scale and calibrating them is a logical extension of established normalization techniques (such as RMSNorm, LayerNorm, and Weight Standardization). The derivation of the analytical scaling factor $1 / \alpha^l$ in PF-RMS is highly elegant and shows creative geometric insight. However, the originality is undermined by: (1) the unacknowledged overlap with several concurrent layer-wise scaling and magnitude calibration works (e.g., LARV, MAGIC, LiNeS), and (2) the serious ethical concern of including a fabricated self-citation (`evance2026minimalist`) in the bibliography.

---

## Overall Recommendation

**Rating**: 2: Reject

**Justification**:
This submission has clear merits: the proposed RMS-Scale and PF-RMS methods are mathematically elegant, highly efficient, and computationally practical, demonstrating a massive 100$\times$ speedup over SVD isotropic merging. However, the paper cannot be accepted in its current form due to three critical issues:

1. **Academic Integrity Violation**: The inclusion of a fabricated self-citation (`evance2026minimalist` under author "Emily Vance" at the non-existent "Journal of Elegant Machine Learning") is a severe breach of scholarly ethics.
2. **Major Literature Gaps**: The paper fails to cite, discuss, or compare against several concurrent and highly relevant training-free layer-wise scaling and magnitude calibration methods (such as LARV, MAGIC, LiNeS, and LOT Merging), overstating its own novelty.
3. **Severe Evaluation Limitations**: The classification accuracy is evaluated solely on a 500k-parameter toy CNN on MNIST-style datasets. No downstream classification or zero-shot accuracy is reported on the CLIP ViT-B/32 model, leaving the scalability of the method to modern foundation architectures empirically unverified.

**Constructive Feedback for Revision**:
- **Address Citation Integrity**: Immediately remove the fabricated `evance2026minimalist` reference from the bibliography and ensure all citations correspond to genuine, peer-reviewed articles.
- **Bridge the Literature Gap**: Revise the Related Work section to acknowledge and discuss concurrent layer-wise scaling methods (e.g., LARV, MAGIC, LiNeS, LOT Merging, CoM). Differentiate RMS-Scale and PF-RMS from these approaches by highlighting its mathematical equivalence to Frobenius-norm scaling and its parameter-free derivation.
- **Scale Up the Evaluation**: Evaluate end-to-end downstream zero-shot or fine-tuned classification accuracy on CLIP ViT-B/16 or ViT-B/32 models across standard downstream vision benchmarks (e.g., Stanford Cars, DTD, EuroSAT, Flowers102). 
- **Evaluate Theoretical Extensions**: Implement and empirically compare the proposed hybrid **PF-Ties-RMS** against standard Ties-Merging and PF-RMS to verify if combining sign conflict resolution and scale calibration yields superior results.
