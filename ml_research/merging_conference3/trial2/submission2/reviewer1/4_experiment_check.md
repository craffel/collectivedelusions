# 4. Experimental Setup and Empirical Validation Check

From an **empiricist** perspective, the experimental evaluation is the most critical part of a machine learning paper. Below is a rigorous critique of the paper's experimental design, baseline choices, statistical soundness, and support for claims.

## Dataset and Architecture Scope
*   **Backbone:** The authors evaluate on the 86M-parameter CLIP-ViT-B/32 visual backbone, which is a reasonable and standard model for full-network merging experiments.
*   **Datasets:** The evaluation relies on MNIST, FashionMNIST, CIFAR-10, and SVHN. While standard, these are relatively simple and small-scale image classification tasks. It is unclear whether the low-rank characteristics and scale-invariance findings transfer to more complex visual domains (e.g., ImageNet-1K fine-tuning) or to larger architectures (such as decoder-only language models), although the authors acknowledge LLMs as a limitation in Section 4.6.

## Major Empirical Gaps and Methodological Flaws

### 1. Statistically Insignificant Performance Gains
The authors claim that SVS at rank $k=128$ "strictly matches or outperforms standard Task Arithmetic ($74.83\%$ vs. $74.78\%$)". 
However, an analysis of the evaluation setup reveals that this claim is statistically meaningless:
*   **Evaluation Subset size:** The authors evaluate on a restricted test subset of only **1,000 samples per dataset** (total of 4,000 samples across the four tasks).
*   **Marginal Difference:** An average accuracy difference of $0.05\%$ ($74.83\%$ vs. $74.78\%$) on 4,000 samples corresponds to an absolute difference of exactly **2 correct predictions** (2,993 correct vs. 2,991 correct). 
*   **Lack of Statistical Rigor:** There are **no error bars, confidence intervals, or standard deviations** reported anywhere in Table 1 or the figures. No multiple random seeds were run for the expert models or the evaluations. A difference of 2 samples out of 4,000 on a single run is well within the margin of random seed and evaluation subset variance. Claiming "strict outperformance" based on this is a major empirical overstatement.

### 2. SVS is Outperformed by Simpler, Heuristic Baselines
Table 1 shows that:
*   **TIES-Merging** achieves **$77.98\%$** average accuracy (beating SVS by **$3.15\%$**).
*   **DARE** achieves **$75.18\%$** average accuracy (beating SVS by **$0.35\%$**).
Although the authors provide an interesting discussion of the "Representation Gap" (spectral-domain low-pass filtering yields dense matrices that still overlap in the coordinate basis, causing cross-layer interference, whereas coordinate pruning eliminates localized collisions), the bottom-line empirical result is that SVS is **strictly inferior** to simpler, heuristic, coordinate-basis pruning methods. For a practitioner, TIES or DARE remains a more effective choice, reducing SVS's utility as a state-of-the-art merging operator.

### 3. Omission of the Most Relevant SVD-Based Baselines
SVS is positioned as a post-hoc SVD-based merging operator. Section 2 cites closely related prior and concurrent works:
*   **Task Singular Vectors (TSV-Compress)** (Gargiulo et al., 2025)
*   **SVD-Merging** (Stoica et al., 2025)
However, **none of these SVD-based methods are included as baselines in the experiments.**
To establish the empirical significance of SVS, it is essential to compare it directly against TSV-Compress at equivalent ranks. Since SVS and TSV-Compress share the same core post-hoc SVD projection formula, omitting TSV-Compress makes it impossible to judge what empirical value SVS actually adds over existing SVD-based merging approaches.

### 4. Lack of Critical Baselines for Entropy-SVS
In Section 4.5, Table 2 and Figure 5 evaluate the performance of **Entropy-SVS** across different scaling multipliers, tracing an accuracy-vs-rank compression curve.
However, **there is no comparison against a uniform-rank SVS baseline at the same average rank.**
For example:
*   At $m_{\text{entropy}}=0.8$, Entropy-SVS achieves $74.75\%$ accuracy with an average rank of **$87.15$**.
*   At $m_{\text{entropy}}=0.4$, Entropy-SVS achieves $74.55\%$ accuracy with an average rank of **$43.90$**.
*   To prove that the dynamic information-theoretic rank allocation of Entropy-SVS is actually beneficial, the authors must compare it directly against standard SVS using uniform ranks of $k=87$ and $k=44$. Without this comparison, a uniform rank of $k=44$ could perform equally well or better, which would render the entire Shannon entropy formulation empirically redundant.

## Verdict on Empirical Support for Claims
*   **Claim 1 (SVS matches/outperforms TA):** Unsupported/statistically insignificant due to tiny evaluation subsets, lack of seeds, and a marginal difference of 2 samples.
*   **Claim 2 (BWN preserves scale in un-normalized MLPs):** Weakly supported; the MLP experts are extremely poorly trained ($77\%$ on MNIST) and the accuracy gains are marginal ($0.75\%$) with no statistical significance tests.
*   **Claim 3 (Entropy-SVS traces robust Pareto curve):** Partially supported, but lacks the necessary baseline of uniform rank at equivalent average ranks to prove the value of dynamic rank allocation.
