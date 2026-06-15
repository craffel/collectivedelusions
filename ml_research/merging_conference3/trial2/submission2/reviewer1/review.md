# Peer Review

**Title:** Occam's Razor in Weight Space: Spectral Model Merging via Singular Value Slicing  
**Reviewer Recommendation:** 3: Weak Reject  

---

## 1. Summary of the Paper
This paper addresses the problem of multi-task model merging, which aims to combine multiple task-specific expert neural networks (fine-tuned from a shared base model) into a single multi-task model without performing additional training or incurring test-time optimization overhead. To achieve this, the authors propose **Spectral Model Merging via Singular Value Slicing (SVS)**, a non-parametric, training-free, and closed-form model merging operator. SVS performs a Singular Value Decomposition (SVD) on task-specific weight updates (task vectors), retains only the top $k$ principal singular components to filter out fine-tuning noise, and linearly combines the sliced task vectors. 

To address potential weight scale distortions across un-normalized layers, the authors propose **Barycentric Weight Normalization (BWN)** to scale the merged weights to match the weighted barycenter of the individual experts. They mathematically prove that in standard architectures with downstream feature normalization layers (e.g., LayerNorm, RMSNorm, L2-normalization), global positive weight scaling factors are neutralized. They validate SVS and Entropy-SVS—a dynamic rank allocation scheme based on Shannon spectral entropy of singular values—on the visual backbone of CLIP-ViT-B/32 across four datasets (MNIST, FashionMNIST, CIFAR-10, SVHN). Finally, they validate BWN's scale preservation in a non-normalized 3-layer MLP environment.

---

## 2. Strengths and Weaknesses

### Strengths
1. **Theoretical Rigor on Scale Invariance:** The mathematical derivations demonstrating that global positive weight-scaling factors cancel out under standard downstream feature normalization layers (L2, LayerNorm, RMSNorm) are correct, elegant, and highly insightful. It provides a solid explanation for why complex weight-scaling algorithms are often redundant in modern architectures.
2. **Conceptually Elegant Rank Allocation:** The proposed **Entropy-SVS** framework is a clever, information-theoretic approach to dynamic rank allocation. Using normalized Shannon spectral entropy of singular values to scale the slicing rank $k_l$ of each layer is a principled way to respect the hierarchical nature of deep neural networks.
3. **High Presentation Quality:** The paper is exceptionally well-written, logically structured, and easy to follow. The mathematical notation is clean and precise, and the figures/tables are professional and readable.
4. **Intellectual Honesty:** The limitations section (Section 4.6) is highly transparent, accurately discussing crucial challenges such as task-agnostic routing, extensions to SwiGLU gated MLPs in autoregressive language models, and spatial-spectral tradeoffs.

### Weaknesses
Despite the paper's mathematical elegance, there are several major empirical gaps, methodological flaws, and statistical limitations that severely undermine the strength of its core claims:

1. **Lack of Statistical Rigor and Insignificant Gains:** 
   The authors claim that SVS at rank $k=128$ "strictly matches or outperforms standard Task Arithmetic ($74.83\%$ vs. $74.78\%$)". However, the evaluation is performed on a highly restricted test subset of only **1,000 samples per dataset** (totaling 4,000 samples across the four tasks). An average accuracy difference of $0.05\%$ ($74.83\%$ vs. $74.78\%$) on 4,000 samples corresponds to an absolute difference of exactly **2 correct predictions** (2,993 correct vs. 2,991 correct). 
   Crucially, there are **no error bars, confidence intervals, or standard deviations reported** in any of the tables or figures. No multiple random seeds were run for either the fine-tuning of experts or the evaluations. A difference of 2 samples on a single run is well within the margin of random seed and subset selection variance. Claiming "strict outperformance" based on this is empirically irresponsible.
2. **Exceedingly Poor MLP Expert Training:** 
   To validate BWN in un-normalized architectures where scaling is not neutralized, the authors train a 3-layer MLP on MNIST and FashionMNIST (Section 4.5). However, the reported performances of the individual experts are extremely poor:
   * **Expert A (MNIST):** $77.00\%$ accuracy. (A simple linear model or 2-layer MLP on MNIST should easily achieve $>95-98\%$ accuracy).
   * **Expert B (FashionMNIST):** $69.00\%$ accuracy. (A standard MLP should easily get $>85\%$ accuracy).
   These exceptionally low accuracies indicate that the experts are severely under-optimized, did not converge, or were trained with a bug. Evaluating model merging on "broken" experts undermines the validity of the conclusions, as the task vectors represent incomplete optimization trajectories rather than stable semantic manifolds. Furthermore, the reported gains of BWN ($0.75\%$ at $\lambda=0.1$ and $0.25\%$ at $k=8$) are tiny and lack statistical validation.
3. **Omission of Direct Spectral Baselines:** 
   SVS is positioned as a post-hoc SVD-based merging operator. Section 2 cites closely related prior and concurrent works, such as **Task Singular Vectors (TSV-Compress)** (Gargiulo et al., 2025) and **SVD-Merging** (Stoica et al., 2025). However, **neither of these SVD-based methods is evaluated as a baseline.** To establish the empirical significance of SVS, it is essential to compare it directly against TSV-Compress at equivalent ranks. Without this, it is impossible to judge whether SVS's specific slicing formulation represents any real improvement over existing spectral merging operators.
4. **Missing Ablation for Entropy-SVS:** 
   Table 2 and Figure 5 sweep the entropy scaling multiplier $m_{\text{entropy}}$ for Entropy-SVS, tracing an accuracy-vs-rank curve. However, there is **no comparison against standard SVS using uniform ranks at the same average rank.** For example, at $m_{\text{entropy}}=0.4$, Entropy-SVS achieves $74.55\%$ accuracy with an average rank of $43.90$. To prove that the dynamic entropy-based rank allocation is actually beneficial, the authors must compare it directly against standard SVS using a uniform rank of $k=44$. Without this comparison, a uniform rank of $k=44$ could perform equally well or better, which would render the entire Shannon entropy formulation empirically redundant.
5. **Major Reproducibility Gaps:** 
   The paper is missing critical details regarding the experimental setup. There is absolutely no description of how the CLIP experts were fine-tuned. Crucial hyperparameters such as the learning rate, epochs, optimizer, batch size, weight decay, and learning rate scheduler are entirely missing. Similarly, the training parameters of the 3-layer MLP are omitted. While a placeholder GitHub URL is provided, it contains no code, making the experiments currently impossible to reproduce.

---

## 3. Detailed Evaluations

### Soundness: Fair
The mathematical soundness of SVS and the global scaling cancellation proofs is excellent. However, from an empirical perspective, the soundness of the claims is **fair** (bordering on poor) due to:
* The statistical insignificance of the core CLIP results (a 2-sample difference on a truncated test set).
* The extremely poor performance of the trained MLP experts.
* The lack of error bars, multiple random seeds, and standard deviations.
* The lack of comparative SVD-based baselines.

### Presentation: Good
The paper is clearly written, well-structured, and highly readable. The authors do an excellent job explaining the theoretical concepts. However, the rating is capped at **good** because crucial training details and hyperparameters required for reproducibility are completely omitted from the main text and there is no appendix to provide them.

### Significance: Fair
The significance of the paper is **fair**. While the theoretical insights on scale-invariance and the concept of entropy-based rank allocation are highly relevant and interesting, the empirical results show that SVS is strictly outperformed by simpler, coordinate-basis pruning methods like TIES-Merging ($77.98\%$ vs. $74.83\%$). Because SVS performs worse than simpler baselines and lacks direct SVD-based comparisons, its practical significance is currently limited.

### Originality: Good
The originality of the paper is **good**. SVD-based model merging is an active area with concurrent work, but the authors distinguish themselves by deriving formal scale-invariance proofs in normalized networks and proposing an information-theoretic entropy-based dynamic rank allocation scheme.

---

## 4. Questions and Actionable Feedback for Authors

To improve the paper and meet the standards of a rigorous empirical submission, the authors are strongly encouraged to address the following actionable feedback:

1. **Run Multiple Seeds and Report Error Bars:** re-run all merging experiments across at least 3-5 random seeds (both for expert training and evaluation) and include standard deviations or confidence intervals in Table 1, Table 2, and Figures 2, 4, and 5.
2. **Evaluate on Full Test Sets:** Replace the 1,000-sample test subsets with the full test sets of MNIST, FashionMNIST, CIFAR-10, and SVHN to eliminate evaluation noise and ensure that any reported accuracy differences are statistically sound.
3. **Compare Against Direct SVD Baselines:** Include **Task Singular Vectors (TSV-Compress)** (Gargiulo et al., 2025) and/or **SVD-Merging** (Stoica et al., 2025) as baselines in Table 1 at equivalent ranks to clearly demonstrate the empirical advantage of the proposed SVS method.
4. **Add Uniform Rank Baselines for Entropy-SVS:** In Table 2 or Figure 5, plot the performance of uniform SVS at the same average ranks as those obtained by the Entropy-SVS sweep. This is crucial to validate whether the dynamic Shannon entropy allocation performs significantly better than a simple flat uniform rank.
5. **Re-train the MLP Experts:** Re-optimize the un-normalized 3-layer MLP model. Ensure that Expert A on MNIST achieves standard accuracies of $>95\%$ and Expert B on FashionMNIST achieves $>85\%$. Re-run the BWN validation on these fully converged models to ensure the scale-preservation claims are demonstrated on representative neural networks.
6. **Provide Reproducibility Details:** Add a detailed section or appendix outlining all hyperparameters used for training the CLIP experts and MLP models (optimizer, learning rate, epochs, batch size, weight decay, etc.).
7. **Explain Tensor Flattening Choices:** Provide a brief discussion or pilot experiments analyzing how alternative tensor flattening axes (e.g., grouping by input channels or fully flattening) affect the singular value spectrum and the resulting Shannon entropy.
