# Peer Review: Norm-Equalized Task Arithmetic (NETA)

## 1. Summary of the Paper
This paper introduces **Norm-Equalized Task Arithmetic (NETA)**, an analytical, closed-form, training-free, and data-free model merging technique. Standard Task Arithmetic (TA) suffers from representation dominance, where tasks with large distribution shifts or complex objectives undergo larger weight shifts during fine-tuning. This results in task vectors with disproportionately large Frobenius norms, which dominate the representation space of the merged model and destructively interfere with other tasks. NETA addresses this imbalance analytically by equalizing the Frobenius norms of task vectors at each layer before merging, ensuring isotropic representation strength across tasks.

The paper also presents an insightful conceptual contribution by identifying the **Overfitting-Optimizer Paradox** in unsupervised Test-Time Adaptation (TTA) merging (such as AdaMerging). Joint prediction entropy minimization on unlabeled calibration data biases the optimizer toward easy, low-entropy tasks (e.g., MNIST and SVHN), causing it to suppress the coefficients of harder, high-entropy tasks (e.g., FashionMNIST). NETA completely avoids this failure mode entirely zero-shot, data-free, and parameter-free. 

To address practical implementation challenges, the paper introduces three key components:
- **$\alpha$-Relaxed NETA**: A continuous relaxation framework ($\alpha \in [0, 1]$) to interpolate between standard TA ($\alpha=0$) and full NETA ($\alpha=1$), balancing peak performance and representation fairness.
- **Noise-Damping Stabilizer ($\beta \ge 0$)**: Soft-thresholding stabilizer to prevent noise amplification in inactive layers.
- **Composite Layer Grouping (Group 0)**: A grouping mechanism for early input-stage parameters to maintain positional and structural consistency.
- **Closed-Form Update Scale Compensation ($\gamma^l$)**: An analytical factor to scale the merged NETA update at each layer, mitigating directional norm contraction and restoring the cumulative update scale of standard TA.

---

## 2. Strengths and Weaknesses

### Strengths
- **Conceptual Clarity and Elegance (Occam's Razor)**: NETA resolves task dominance imbalances analytically and zero-shot, requiring zero optimization parameters, zero calibration images, and zero backpropagation passes. It is an extremely clean, reproducible, and mathematically elegant solution compared to complex test-time adaptation pipelines.
- **Rigorous Critique of Test-Time Adaptation**: The formalization of the **Overfitting-Optimizer Paradox** is a highly insightful contribution. Exposing the vulnerability of joint entropy minimization under task difficulty imbalances provides a vital diagnostic contribution to the model-merging literature.
- **Exceptional Mathematical Rigor**: The paper provides formal mathematical derivations of NETA's geometric properties (perfect magnitude isotropy and preservation of cumulative individual norms) and rigorously addresses subtle details like directional norm contraction through the proposed closed-form compensation factor $\gamma^l$.
- **Thorough and Honest Empirical Evaluation**:
  - The authors report performance across three independent random seeds with standard deviations, demonstrating high statistical confidence.
  - The paper is highly transparent and scientifically honest about trade-offs: it explicitly addresses that standard NETA acts as an isotropic regularizer that curtails peak SVHN accuracy to distribute representation strength fairly, and provides deep ablations (like the $\lambda_0$ grid search) to explore these mechanics.
  - Excellent explanations of subtle empirical details, such as explaining the $0.00\%$ standard deviations observed in some AdaMerging configurations as physical convergence to optimization boundaries (clamping boundaries) over discretized test sets.
- **Comprehensive Ablation Studies**: The depth of the ablation studies is highly impressive, successfully validating the continuous $\alpha$-relaxation, the composite layer grouping (Group 0), the noise-damping stabilizer $\beta$, and the $\gamma^l$ scale-compensation factor.

### Weaknesses & Areas for Improvement (Constructive Suggestions)
While the paper is technically flawless and highly thorough, the following minor suggestions could further strengthen its impact:
- **Minor Notation Discrepancy ($\beta$ vs. $\epsilon$)**: There is a slight mathematical/notational inconsistency between Section 3.3 and Section 3.4. In Section 3.3, Eq. 9 defines the scaling coefficient denominator using $\beta$ as a noise-damping stabilizer. However, in Section 3.4 (under "Perfect Magnitude Isotropy" and "Preservation of Cumulative Individual Norms"), the derivations substitute $w_k^l$ and change the denominator's symbol to $\epsilon$ without explanation (e.g., in Eq. 10 and 11, and the surrounding text). The authors should make this notation consistent by using $\beta$ throughout both sections (or clearly state that $\beta$ is replaced by $\epsilon$ in the zero-noise limit).
- **Minor Table Formatting/Bolding Inconsistencies**: There are a couple of small formatting/bolding errors in the tables:
  - In Table 1, under the "Test-Time Optimization" sub-table for the MNIST column, Task-Wise AdaMerging achieves $98.49 \pm 0.07\%$ while Layer-Wise AdaMerging achieves $98.44 \pm 0.00\%$. However, the lower value of $98.44\%$ is bolded instead of $98.49\%$.
  - In Table 2 (Ablations), under the CIFAR-10 column, the highest value is NETA (No Group 0) at $92.77 \pm 0.24\%$ (and the second highest is NETA ($\alpha = 0.5$) at $92.71 \pm 0.28\%$), but no value in this entire column is bolded. One of these should be bolded for consistency.
- **Evaluation Scale (Dataset Suite)**: Standard CLIP model-merging publications often evaluate across an 8-dataset suite representing a wider variety of domain shifts (including Cars, DTD, EuroSAT, RESISC45, SUN397, etc.). While the 4-task visual suite is standard and more than sufficient to prove the paper's claims and the Overfitting-Optimizer Paradox, validating NETA on the full 8-task suite would provide broader empirical validation.
- **Architecture Backbones**: The evaluation is restricted to the CLIP ViT-B/32 visual encoder. Expanding NETA to larger visual encoders (e.g., ViT-L/14) or evaluating its applicability to generative Large Language Models (LLMs) would demonstrate the generalizability of the analytical scaling across different model families and architectures.
- **Test Set Sub-sampling**: Sub-sampling to 1024 test images is a practical and necessary constraint under strict resource limits. While the authors offer robust statistical justifications, evaluating on the full test sets of these standard benchmarks would remove any remaining statistical concerns.
- **Depth-Dependent Anisotropic Scaling**: Standard NETA assumes that task experts should contribute equally at all stages of the network. However, early layers extract general, domain-agnostic features while deeper layers extract specialized, task-specific features. Exploring depth-dependent anisotropic scaling (e.g., enforcing strict isotropy in early layers but allowing selective task dominance in deeper layers) presents a very promising avenue for future research that could resolve the peak performance vs. representation fairness trade-off.

---

## 3. Ratings

- **Soundness**: **Excellent**
  - The mathematical formulas are correctly derived, the assumptions are physically grounded, and the authors are transparent and scientifically honest about the trade-offs of their work.
- **Presentation**: **Excellent**
  - The paper is brilliantly written, beautifully structured, and incredibly easy to follow. Concepts like the Overfitting-Optimizer Paradox and directional norm contraction are explained with high-signal clarity.
- **Significance**: **Good**
  - Resolving task representation dominance and interference is a major bottleneck in model merging. NETA provides a highly practical, zero-shot, closed-form baseline that practitioners can easily adopt. The identification of the Overfitting-Optimizer Paradox will likely influence future test-time weight adaptation research.
- **Originality**: **Good**
  - Applying analytical, layer-wise Frobenius norm equalization is a creative combination of geometric principles. The formalization of the Overfitting-Optimizer Paradox and the closed-form scale-compensation factor $\gamma^l$ represent valuable additions to the field.

---

## 4. Overall Recommendation

**Rating**: **5: Accept**

**Recommendation Rationale**: 
This is a technically solid, highly rigorous, and exceptionally well-written paper. The authors present a simple and elegant analytical weight-space scaling method (NETA) that successfully addresses task representation dominance zero-shot, data-free, and parameter-free. Crucially, the identification and formalization of the Overfitting-Optimizer Paradox represents a significant diagnostic contribution that critiques the prevailing SOTA test-time adaptation paradigm under task-difficulty imbalances. The mathematical derivations are sound, and the empirical evaluation is highly thorough, offering a massive depth of ablation studies (seeds, relaxation, stabilizers, grouping, scale compensation, and grid search). Although the evaluation scope is currently limited to a 4-dataset visual suite on CLIP ViT-B/32 and subsampled test sets, these are minor and understandable limitations. The paper is ready for publication and represents a strong, high-signal addition to the model merging community.
