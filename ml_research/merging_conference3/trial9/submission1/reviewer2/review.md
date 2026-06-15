# Peer Review: PAC-Bayesian Smooth Trajectory Merging (PAC-STM)

## Summary of the Paper
The paper presents **PAC-Bayesian Smooth Trajectory Merging (PAC-STM)**, a learning-theoretic framework for dynamic, dynamic layer-wise ensembling of task-specific Parameter-Efficient Fine-Tuning (PEFT) experts (specifically, Low-Rank Adaptation or LoRA experts) on a shared pre-trained backbone. 

In multi-task streaming workloads, static weight-space merging techniques suffer from **Heterogeneity Collapse** (interference under mixed batches) and **Vectorization Collapse** (loss of GPU tensor parallelism when weights are merged dynamically per-request). Operating in activation space (activation-blending) avoids both collapses, but calibrating layer-wise routing parameters under ultra-low calibration data regimes (e.g., $N=16$) via standard Empirical Risk Minimization (ERM) suffers from severe **transductive overfitting**, leading to high-frequency parameter oscillations (temperature spikes) across depth.

PAC-STM solves this by modeling layer-wise routing log-temperatures as a probability distribution over trajectories. By defining the prior over these trajectories as an autoregressive Markov chain (a Gaussian random walk across network depth), the authors prove that the Kullback-Leibler (KL) complexity penalty in the PAC-Bayesian bound analytically derives a first-order finite-difference smoothness regularizer. This collapses the stochastic bound into a deterministic trajectory optimization problem with a principled smoothness penalty.

The authors also introduce:
1. **Unit-Norm PCA Subspace Projection (UN-PCA-SEP):** Normalizing early hidden representations and projecting them onto task-specific principal component bases to bound features in $[0, 1]$.
2. **UN-KPCA-SEP (Uncentered Kernel PCA):** A non-linear extension to handle curved representational manifolds, showing that local task-specific kernel centering must be omitted to preserve centroid identity.
3. **Skip-Aware Prior Topologies:** DAG-structured priors that mirror residual/skip connections.
4. **Sparse Top-$k$ Activation Blending:** A serving optimization with rigorous approximation error bounds.

The framework is evaluated in a 14-layer Analytical Coordinate Sandbox and validated on a pre-trained Vision Transformer (\texttt{ViT-B/16}). PAC-STM outperforms unregularized ERM by 2.05% in heterogeneous batch accuracy, exhibits absolute immunity to Heterogeneity and Vectorization Collapse, and enforces smooth, continuous parameter transitions across depth.

---

## Strengths and Weaknesses

### Strengths
1. **Exceptional Theoretical Rigor:** The mathematical foundation of the paper is highly elegant. Proving that a continuous depth-wise Gaussian random walk prior analytically derives a first-order finite-difference smoothness penalty (Theorem 3.1) establishes a formal, principled connection between physical depth continuity and PAC-Bayesian complexity.
2. **High Practical/Systems Relevance:** The paper addresses concrete, high-impact systems challenges (Heterogeneity Collapse and Vectorization Collapse in concurrent multi-task serving) rather than purely abstract theory. The sparse top-$k$ serving optimization (Equation 19) is highly practical, and the authors support it with a clean representation error bound (Theorem 3.2).
3. **Nuanced Representation Geometry Analysis:** The analysis of local task-specific coordinate projections—specifically the insight that centering local kernel matrices destroys the centroid task identity in the RKHS—is exceptionally sophisticated and represents a deep understanding of representation learning.
4. **Methodological Completeness and Reproducibility:** The paper is highly complete, providing detailed pseudocode for both offline calibration and online serving (Appendix Section A) and a complete hyperparameters table (Appendix Section C). The proofs are complete and mathematically solid.

### Weaknesses
1. **Critical Citation and Attribution Gaps:** Despite its strong theoretical focus, the paper has significant bibliographic gaps. Central baselines like **SABLE (Block)**, **SABLE (PCA)**, and **PAC-ZCA (Global)** are discussed and evaluated as "existing methods/prior work," but there are absolutely no bibliographic citations or references provided for them. This lack of proper attribution makes it impossible to locate the original publications, understand their specific formulations, or verify that the baseline implementations are faithful.
2. **Oversight of Directly Related Literature:** The paper completely misses the highly relevant recent work **"Model Merging is Secretly Certifiable: Non-Vacuous Generalisation Bounds for Low-Shot Learning" (Kim et al., UAI 2026 / arXiv:2505.15798)**. Since both works apply PAC-Bayes theory to model merging in low-shot data regimes, discussing and citing this paper is crucial to properly situate the contribution.
3. **Discrepancy between LLM Motivation and Empirical Scope:** The introduction and methodology heavily motivate the framework using Large Language Models (LLMs), text generation, and PEFT servers (like Punica or S-LoRA). However, the actual empirical validation is restricted entirely to image classification tasks (simulated sandbox and ViT-B/16 on MNIST/CIFAR-10). The lack of direct NLP/LLM text-generation experiments creates a minor gap between the paper's heavy LLM-focused motivation and its empirical validation.

---

## Evaluation of Specific Dimensions

### Soundness: Excellent
The paper is technically flawless. The proof of Theorem 3.1 is mathematically rigorous, and the decomposition of consecutive differences under the posterior $Q$ correctly accounts for variance transitions. The proof of Theorem 3.2 is clean and correct, establishing a tight error bound. The methods are highly appropriate, and the empirical results are validated across multiple seeds with paired t-tests confirming high statistical significance ($p < 0.008$ for the primary results). The uncentered Kernel PCA analysis is elegant and empirically verified.

### Presentation: Good
The paper is exceptionally well-written, clearly structured, and easy to follow. However, the presentation is downgraded from "Excellent" to "Good" due to the significant scholarly oversight regarding citations. Key baselines are introduced and evaluated without references, and directly related PAC-Bayesian model merging literature is ignored. Fixing these citation issues is mandatory to meet the highest standard of presentation and literature positioning.

### Significance: Excellent
Dynamic model ensembling and serving is a highly critical and active research area. By providing the first learning-theoretic foundation for dynamic, layer-wise activation-blending model merging and proving that depth-wise continuity corresponds strictly to limiting hypothesis complexity, this work represents a major theoretical advance that can guide future systems and algorithmic designs.

### Originality: Excellent
The core concept of formulating a joint Markovian trajectory prior over network depth within a PAC-Bayesian framework is highly novel and creative. The uncentered local Kernel PCA projection and the residual-aware skip prior are also highly original, nuanced extensions that demonstrate an impressive level of theoretical innovation.

---

## Overall Recommendation

**Recommendation:** **5: Accept**

**Justification:**
This is an exceptionally strong, technically sound paper that bridges learning theory (PAC-Bayes) and deep neural trajectory continuity to solve a very practical systems challenge (transductive overfitting in dynamic multi-task PEFT serving). The theoretical derivations are elegant, the proofs are correct, and the empirical validation is comprehensive and statistically verified. While the paper has a minor gap in its empirical scope (motivating LLMs but evaluating on image classification) and is missing crucial bibliographic citations and related work discussions, the core intellectual contribution is highly valuable and easily warrants acceptance. I strongly recommend acceptance, provided the authors address the bibliographic and literature-contextualization issues highlighted below.

---

## Detailed Comments, Questions, and Suggestions for Authors

### 1. Critically Missing Citations
In the text and tables, you introduce and evaluate **SABLE (Block)**, **SABLE (PCA)**, and **PAC-ZCA (Global)** as "prior systems/existing methods" (e.g., Section 2.1: *"Methods like SABLE and PAC-ZCA dynamically route inputs sample-by-sample..."*). However, there are absolutely no bibliographic citations or references provided for these methods anywhere in the paper or in `references.bib`. 
- **Action Required:** Please add the correct bibliographic citations for SABLE and PAC-ZCA. If these are concurrent, unpublished baselines, or specific configurations of other works, please explicitly define their origin, formulation, and implementation details so readers can locate the original publications and verify the faithfulness of your baseline setups.

### 2. Missing Related Work and Literature Contextualization
Your paper completely overlooks **"Model Merging is Secretly Certifiable: Non-Vacuous Generalisation Bounds for Low-Shot Learning" (Kim et al., UAI 2026 / arXiv:2505.15798)**. This is highly relevant because both of your works apply PAC-Bayes theory to model merging in data-scarce settings.
- **Action Required:** Cite and discuss Kim et al. in your Related Work (Section 2.2). Please clearly articulate the "delta" between your works:
  - *Kim et al.* focus on static, weight-space merging and use PAC-Bayes bounds *post-hoc* to certify the generalization of the merged model, treating the individual experts as the prior and the merged weights as the posterior.
  - *PAC-STM* focuses on dynamic, activation-space layer-wise ensembling and applies a Markovian random walk prior *over the ensembling log-temperatures across network depth* to directly optimize and regularize the layer-wise routing trajectory.
  - This discussion is critical to properly situating your paper within contemporary PAC-Bayesian model-merging literature.

### 3. Discrepancy between LLM Motivation and Empirical Scope
Your introduction and methodology heavily motivate PAC-STM using decoder-only Large Language Models (LLMs), text generation, and dynamic PEFT serving frameworks like Punica, S-LoRA, or vLLM. However, your actual empirical results are limited entirely to image classification.
- **Action Required:** Please acknowledge this empirical limitation in Section 4.4 or Section 5. Explain why text generation experiments (such as LLaMA-7B on instruction following or arithmetic benchmarks) were omitted from the current submission. Discussing this will help manage reader expectations and provide a clearer roadmap for the future work you mention.

### 4. Selection and Sensitivity of Step Variance ($\sigma^2$)
Section 4.5 and Table 6 demonstrate that the ensembling accuracy is sensitive to the transition step variance $\sigma^2$ (with performance collapsing as $\sigma^2 \to \infty$ or $\sigma^2 \to 0$, and peaking at $\sigma^2 = 0.5$).
- **Question for the Authors:** How should a practitioner select or tune $\sigma^2$ in practice when deploying PAC-STM on a new, uncalibrated backbone? Is there a theoretical rule of thumb (e.g., based on the number of layers $L$ or the dimensionality $D$), or is it purely an empirical hyperparameter? Providing guidance on this would greatly increase the practical utility of your framework.

### 5. Local Kernel Centering Theory
Your analysis of uncentered vs. centered Kernel PCA in Section 3.2 and Section 4.6 is outstanding. Subtracting the local task mean vector in centered Kernel PCA destroys the centroid task identity (which represents the class separation signal), collapsing the routing accuracy to near-random ($24.62\%$). This is a brilliant theoretical and empirical point.
- **Suggestion:** Consider highlighting this uncentered local KPCA finding more prominently in the Abstract or Introduction, as it is a highly general and valuable insight for any representation-alignment or local coordinate extraction framework in deep learning.
