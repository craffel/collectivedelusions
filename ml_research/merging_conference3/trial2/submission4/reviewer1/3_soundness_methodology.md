# Soundness and Methodology Evaluation

This document evaluates the mathematical rigor, methodological appropriateness, potential technical flaws, and reproducibility of the proposed EdgeMerge framework.

---

## 1. Clarity of the Description and Mathematical Rigor
The description of the EdgeMerge methodology is exceptionally clear, precise, and structured. Every stage of the framework is formalized with clear mathematical equations:
- **Stage 1 (FOAS):** Equations (1) and (2) clearly define the forward-only activation sampling process for the base model ($H_{base, k}$) and task experts ($H_k$).
- **Stage 2 (SNDAS):** Equations (5), (6), and (7) formalize the delta activations ($\Delta H_k$), Frobenius norm scale normalization ($\tilde{\Delta} H_k$), and absolute-average channel salience vector ($S_k[j]$).
- **Stage 3 (CWSG):** Equation (8) utilizes a parameterized softmax function to convert salience scores into channel-wise routing weights ($\alpha_k[j]$).
- **Weight Reconstruction & DSR:** Equations (9), (10), (11), and (12) detail the reconstruction process and mathematically demonstrate the scale-dampening effect of softmax routing compared to static summation.

The authors also include a structured pseudo-code description (Algorithm 1) that is logical and easy to follow.

---

## 2. Appropriateness of Methods under Edge Constraints
The training-free, forward-only design is highly appropriate for resource-constrained edge systems. By eliminating backpropagation, gradient tracking, and optimizer states, the framework dramatically reduces peak memory footprint and latency. 
- **Choke-Point Localization:** Restricting the dynamic channel-gating exclusively to the visual projection layer (`model.visual.proj`) is a highly appropriate, well-reasoned, and high-leverage choice. Localizing adaptation to a low-rank bottleneck layer ($768 \to 512$) right before the classification heads minimizes computation while functioning as an effective post-hoc "visual router."
- **Resource Shortcut:** Extracting representations ($X_k^{base}$) solely from the base model visual encoder and sharing them across all expert projections is an ingenious and highly appropriate engineering trade-off. It reduces encoder forward passes to $1\times$, avoiding the need to load or run $K$ separate encoders simultaneously, which is highly practical for edge systems.

The authors' defense of this shortcut (Section 3.3)—including addressing the *Encroached Encoder Fallacy* through representational alignment and implicit regularization—is conceptually sound and thoroughly reasoned.

---

## 3. Potential Technical and Logical Flaws
Despite the high mathematical clarity, there is a fundamental logical and scientific discrepancy between the paper's core motivation and its empirical results:

### The Channel Gating Inefficacy Flaw
The primary scientific hypothesis of EdgeMerge is that **channel-wise softmax gating** resolves inter-task weight conflicts by dynamically routing individual channels to the experts that find them most salient. This is motivated extensively in Sections 1, 2, and 3.

However, the ablation studies (Section 4.3.4) show that:
- **Full EdgeMerge (CWSG):** **69.58%** average accuracy.
- **Layer-wise Gating (LWG):** **69.59%** average accuracy.
- **Uniform Gating (Flat $\alpha_k = 1/K$):** **69.58%** average accuracy.

This demonstrates a major logical and methodological flaw in the proposed routing machinery:
1. **No Improvement over Uniform Blending:** Assigning every channel of the visual projection layer an equal weight across all tasks (uniform gating) performs identically to the highly complex activation sampling and salience gating machinery.
2. **Failure of the Saliency Metric:** If dynamic, channel-wise routing were actually resolving inter-task conflicts, it should outperform uniform blending, which simply blends all conflicting expert task vectors together. The fact that they perform identically implies that either the visual projection layer does not suffer from inter-task conflicts that require routing, or the proposed scale-normalized delta activation salience (SNDAS) metric fails to extract any meaningful routing signals.
3. **The Core "Working" Component is Minor:** The performance gain of the model (from 68.74% to 69.58%) is driven **entirely** by Decoupled Scale Routing (DSR). But DSR is simply independent hyperparameter tuning of two different layers ($\lambda_{static}=0.25$ and $\lambda_{proj}=0.20$). Thus, the paper's main technical contribution is a highly complex, computationally heavy (relative to TA) calibration loop that has zero empirical utility, while the actual benefit comes from a standard scale-tuning technique.

---

## 4. Reproducibility
The reproducibility of the paper is **outstanding**:
- **Dataset and Backbone Details:** The authors specify the exact datasets, class sizes, and the CLIP ViT-B/32 backbone used.
- **Calibration Settings:** The exact calibration batch size (32 per task) is clearly defined.
- **Hyperparameter Specifications:** The precise optimal values for coupled and decoupled configurations ($\lambda_{static} = 0.25$, $\lambda_{proj} = 0.20$, $\tau = 0.10$) are explicitly reported.
- **Statistical Rigor:** The authors provide a highly rigorous statistical analysis of their evaluation subset size ($N = 1024$), calculating the standard error of the multi-task average ($SE_{avg} \approx 0.51\%$) to mathematically validate that their fast-evaluation setting is highly representative. This ensures that other researchers can confidently reproduce the hyperparameter rankings using the same subset size.
