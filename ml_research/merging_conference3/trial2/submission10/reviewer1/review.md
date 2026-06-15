# Peer Review

**Paper Title:** Deconstructing Adaptive Model Merging: Exposing the Overfitting-Optimizer and Spatial Averaging Paradoxes  
**Author:** Clara Sterling (Stanford University)  

---

## 1. Summary of the Paper
This submission presents a critical, deconstructive study of adaptive test-time model merging, specifically focusing on the state-of-the-art **AdaMerging** (Yang et al., ICLR 2024) framework. AdaMerging optimizes merging coefficients (either layer-wise or task-wise) at test time by minimizing the prediction entropy of the merged model on a small, unlabeled calibration batch. 

The authors investigate whether this high-dimensional optimization captures genuine multi-layer representational coordination or if it merely overfits to the test-time batch. To do so, they introduce two clever diagnostic treatments for optimized layer-wise coefficients:
1. **Intra-Task Layer Shuffling**: Permuting optimized layer-wise coefficients across different layers of the network to disrupt their architectural alignment.
2. **Spatial Averaging**: Replacing optimized layer-wise coefficients with their flat spatial mean over all layers.

The paper uncovers two primary optimization anomalies:
* **The Overfitting-Optimizer Paradox**: While learned layer-wise coefficients are structurally specialized and highly sensitive to their layer positions (shuffling them collapses accuracy from $88.05\%$ to $78.61\%$), they are also prone to transductive test-time overfitting. Post-hoc **Spatial Averaging** acts as an elegant spatial low-pass filter, smoothing away this high-frequency overfitting to achieve $84.96\%$ accuracy, which outperforms the unoptimized Task Arithmetic baseline ($84.64\%$).
* **The Spatial Averaging Paradox**: While post-hoc Spatial Averaging is highly effective, **direct** test-time optimization of flat task-wise scales (Task-wise AdaMerging) fails spectacularly, degrading accuracy to $81.19\%$ (well below its uniform initialization of $84.64\%$). The authors explain this through **multi-task gradient imbalance** under uncalibrated prediction entropy objectives and low-dimensional bottlenecks: the joint entropy loss is dominated by easy tasks (e.g., MNIST), leading the optimizer to scale up their task vectors pathologically, causing severe parameter interference and performance collapse on harder tasks (e.g., CIFAR-10 and SVHN).

The paper proposes a **Calibrated Prediction Entropy** objective as a remedy, proving through controlled experiments that the flat global bottleneck is fundamentally incompatible with joint prediction entropy minimization across shared layers. Finally, the authors analyze representational dynamics layer-by-layer using Linear CKA (Centered Kernel Alignment) to validate their hierarchical routing claims.

---

## 2. Strengths and Weaknesses

### Strengths
* **Creative and Effective Diagnostic Controls**: The design and application of *Intra-Task Layer Shuffling* and *Spatial Averaging* are highly elegant, elegant methods to isolate the structural specialization and transductive overfitting of learned weight-space coefficients.
* **Significant Theoretical Insights (The Spatial Averaging Paradox)**: Exposing and mathematically formalizing the Spatial Averaging Paradox is a highly valuable, counter-intuitive contribution. The theoretical analysis of multi-task gradient imbalance under uncalibrated prediction entropy and shared, low-dimensional weight bottlenecks is compelling and deeply insightful.
* **Exceptional Empirical Rigor**: The paper demonstrates exemplary scientific rigor by controlling for three independent random seeds and partitioning datasets into clean disjoint splits (head-training, calibration, and evaluation). Crucially, by evaluating on the full, standard test splits of all four datasets, the authors achieve an evaluation scale of **56,032 images**, providing high-precision confidence intervals and eliminating data selection bias.
* **Insightful Representational Analysis**: Tracing representation alignment layer-by-layer using Linear CKA (Figure 4) visually and empirically confirms the authors' hierarchical routing hypothesis, proving that the high-dimensional adaptive optimizer preserves early layers while specializing late layers, which post-hoc spatial averaging then regularizes.
* **Immediately Practical regularizer**: Post-hoc Spatial Averaging is shown to be a valuable, label-free scaling estimator, achieving $84.96\%$ accuracy without requiring ground-truth labels for a grid search.

### Weaknesses
* **CRITICAL SCHOLARLY ISSUE: Fabrication of Literature and Fictional Citations**: A deep analysis of the related work and bibliography reveals a **severe, fatal flaw** in the paper's scholarship: **the inclusion of completely fictional/fabricated references.**
  Specifically, the paper cites the following three non-existent papers:
  1. `\cite{convoluted_origami}`: *FoldMerge: Neural Origami for Multi-Task Parameter Basins*, authored by **Fictional, Author A.** and **Fictional, Author B.** (ICML 2025).
  2. `\cite{saim_deconstruction}`: *Deconstructing Sharpness-Aware Isotropic Merging: Redundancy in Coordinate Optimization*, authored by **Fictional, Author C.** (ICML 2025).
  3. `\cite{adamerging_paradox}`: *Exposing the Overfitting-Optimizer Paradox in Layer-Wise Adaptive Merging*, authored by **Fictional, Author D.** (NeurIPS Workshop on Parameter-Efficient Learning 2025).
  
  In Section 2, the authors discuss these papers as if they are actual, peer-reviewed prior literature (e.g., *"recent critiques of Sharpness-Aware Isotropic Merging (SAIM) [saim_deconstruction] have shown..."*, *"analyses of FoldMerge [convoluted_origami] have argued..."*, and *"While prior work on the overfitting-optimizer paradox [adamerging_paradox] indicated..."*).
  A comprehensive search reveals that none of these papers exist in any major academic graph or indexing database. Citing fictional papers written by authors literally named "Fictional, Author X" is a severe scholarly and academic integrity violation. This suggests that the related work section or bibliography may have been generated by a Large Language Model and included without verification, or that draft placeholders were not cleaned up.
  This issue completely undermines the paper's claims of novelty and historical framing: if `adamerging_paradox` is fabricated, then the "Overfitting-Optimizer Paradox" as a concept was *not* previously established in the literature as claimed, and the authors have fabricated a prior work to "build upon."
* **Evaluation Scope (Task Homogeneity)**: The empirical evaluation is restricted to highly heterogeneous datasets. Testing on a homogeneous benchmark (such as DomainNet or PACS) would define the exact boundary conditions under which the Spatial Averaging Paradox manifests, as task-wise entropy landscapes are naturally more balanced in homogeneous domains.
* **No Actual LLM Experiments**: Given the immense current interest in merging Large Language Models (LLMs), the paper's discussion on scaling to LLMs is confined entirely to the "Future Directions" section. Implementing actual experiments on instruction-tuned or domain-specialized language models (e.g., using generation perplexity as the test-time loss) would have vastly expanded the paper's significance.
* **Limited Backbone Architectures**: The experiments are confined to a single isotropic backbone (CLIP ViT-B/32). To prove that the hierarchical routing claims are architecture-agnostic, the authors should evaluate hierarchical vision transformers (like Swin) or convolutional backbones (like ConvNeXt).

---

## 3. Detailed Evaluations

### Soundness
**Rating:** Fair  
**Justification:**  
While the mathematical derivations, diagnostic methods (shuffling and averaging), and empirical findings are technically sound, well-executed, and highly convincing, the soundness of the paper is severely compromised by its literature framing. Citing fabricated/fictional prior work to establish the baseline of the "Overfitting-Optimizer Paradox" makes the paper's scientific and historical claims unsound. If the authors completely excise these fictional references and correctly frame the "Overfitting-Optimizer Paradox" as their own conceptual contribution, the technical soundness would easily be rated as **Excellent**.

### Presentation
**Rating:** Poor  
**Justification:**  
The overall clarity of the writing, the narrative structure, the formatting of the tables, and the design of the figures (especially the excellent Figure 4) are of outstanding quality. However, the presentation is rated as **Poor** because of the severe scholarly negligence of including and citing three fabricated references with author names literally containing the word "Fictional" (Fictional, Author A; Fictional, Author B; Fictional, Author C; Fictional, Author D) in the bibliography and discussing them extensively in the Related Work section. This is an unacceptable flaw for a conference submission.

### Significance
**Rating:** Fair  
**Justification:**  
If the bibliography and related work were correct, this submission would be of **Excellent** significance. It exposes key optimization pathologies in adaptive weight-space combinations, warns the community against over-engineering test-time adaptive pipelines, and introduces post-hoc Spatial Averaging as an elegant, practical, and label-free scaling estimator. However, because the scientific grounding and literature positioning rely on fictional papers, the current significance of the paper is heavily diminished.

### Originality
**Rating:** Fair  
**Justification:**  
The conceptual contribution of the Spatial Averaging Paradox and its mathematical explanation via multi-task gradient imbalance is highly original and brilliant. However, by citing the fictional `adamerging_paradox` as the prior work that "indicated that layer-wise coefficients overfit to the test sample" and claiming to "build upon and formalize these ideas," the authors confuse the boundary of what is their original contribution and what is prior work. If the "Overfitting-Optimizer Paradox" is indeed the authors' original concept, fabricating a prior work to cite it severely damages the originality framing of the submission.

---

## 4. Overall Recommendation
**Score:** 2: Reject  

**Justification:**  
This paper represents a classic case of a work with outstanding technical merits but a fatal scholarly flaw. The empirical deconstruction of AdaMerging, the discovery and formalization of the Spatial Averaging Paradox, and the layer-by-layer CKA analysis are brilliant, highly rigorous, and of exceptional quality. However, the active inclusion and discussion of three completely fictional/fabricated academic papers authored by "Fictional, Author X" in the bibliography and Related Work section is a severe scholarly and academic integrity violation that cannot be tolerated in peer-reviewed venues. 

Because a conference review process is not a collaborative editing service, a paper containing fabricated literature must be rejected. The authors are strongly encouraged to completely remove these fictional citations, thoroughly clean up their related work, correctly frame their original contributions, and resubmit this work to a future venue, where its technical brilliance will undoubtedly shine.

---

## 5. Constructive Comments and Questions for the Authors
1. **Literature Cleanup**: Can you explain the source and rationale of the references `convoluted_origami` (FoldMerge), `saim_deconstruction` (critique of SAIM), and `adamerging_paradox` (Exposing the Overfitting-Optimizer Paradox) by authors named "Fictional"? These papers do not exist, and citing them is a severe scientific error. You must completely excise these references, remove all discussions of them from Section 2, and thoroughly verify the authenticity of all other citations in your bibliography.
2. **Framing the Paradox**: If the "Overfitting-Optimizer Paradox" was not established by a prior work, is this your original conceptual contribution? If so, you should proudly and clearly frame it as your own contribution rather than attributing it to a fictional paper (`adamerging_paradox`).
3. **Task Homogeneity**: How does the Spatial Averaging Paradox behave on a homogeneous multi-task benchmark (e.g., DomainNet)? Since the prediction entropy across domains in DomainNet is naturally more balanced, does direct Task-wise AdaMerging still suffer from gradient imbalance, or does it achieve stable convergence?
4. **LLM Evaluation**: Since your Section 5 notes that token-level and task-level imbalance in LLMs would directly mirror the vision-model gradient imbalance, have you considered running a quick proof-of-concept LLM merge (e.g., merging two LLaMA experts using token perplexity on a calibration set) to verify if the Spatial Averaging Paradox holds in generative language settings?
5. **Architectural Generalizability**: To strengthen your hierarchical routing hypothesis (which is currently evaluated on the isotropic CLIP ViT-B/32), have you considered evaluating your diagnostic controls on a convolutional backbone (e.g., ConvNeXt) or a hierarchical transformer (e.g., Swin)?
