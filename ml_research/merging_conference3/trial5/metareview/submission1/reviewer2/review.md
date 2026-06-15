# Peer Review

---

## 1. Summary of the Paper
The submission addresses the problem of parameter-space model merging during test-time adaptation (TTA). In this paradigm, independently fine-tuned task-specific expert models (which share a common pre-trained base model $\theta_0$) are merged in weight space, and the merging coefficients $\boldsymbol{\lambda}$ are optimized online on unlabeled local test streams using Shannon entropy minimization. 

The authors argue that unconstrained test-time optimization of layer-wise merging coefficients is highly susceptible to transductive noise and local stream biases, resulting in high-frequency spatial oscillations in adjacent layer coefficients. They define this failure mode as the **"Overfitting-Optimizer Paradox"**, where unsupervised local optimization fits transductive noise, leading to catastrophic representation collapse and degraded multi-task generalization.

To resolve this, the authors propose **Riemannian Curvature-Regularized Test-Time Model Merging (RCR-Merge)**. RCR-Merge regularizes the spatial Total Variation (TV) of the merging coefficients across network depth, dynamically scaling adjacent-layer relative penalties by the geometric mean of their pre-trained base curvatures (estimated using the diagonal trace of the pre-trained base model's Fisher Information Matrix, or FIM). The framework also incorporates an absolute coordinate anchoring penalty to prevent global coefficient joint-drift and a scale-invariant heuristic called Gradient Norm Balancing (GNB) to dynamically initialize the regularization weights.

---

## 2. Main Strengths of the Submission
1. **Outstanding Writing and Presentation Quality:** The paper is exceptionally well-written, clearly structured, and easy to follow. The introduction of the problem, description of the pipeline, and development of the methodology flow logically. The TikZ schematic (Figure 1) is highly illustrative and professionally designed.
2. **Exemplary Scientific Transparency:** Unlike many deep learning submissions, the authors are highly transparent and proactive about potential criticisms. Section 4.4 ("Limitations and Circularity Analysis") explicitly discusses the risks of simulation-only studies and potential circularity in their evaluation metrics. They should be commended for designing a "Decoupled Isotropic Euclidean" metric specifically to address and break potential mathematical circularity.
3. **Valuable Conceptual Insights:** The identification of representation collapse due to unconstrained test-time optimization of merging coefficients (the "Overfitting-Optimizer Paradox") is a valuable practical insight. Demonstrating that local, relative spatial smoothing (TV) can perform better than rigid global polynomial constraints (PolyMerge) on modular landscapes is a strong conceptual contribution.
4. **Reproducibility Aids:** The authors provide a detailed PyTorch code recipe in the Appendix (Listing 1) and extensive derivations of the graph Laplacian and Taylor error bounds, making the proposed algorithm easy to replicate.

---

## 3. Major Weaknesses and Detailed Critiques

Despite its strong presentation, the submission suffers from critical weaknesses in its empirical validation, theoretical framing, and mathematical assumptions that make it unready for publication in a top-tier machine learning conference.

### Weakness 1: Severe Reliance on Synthetic Emulators & Lack of Standard Real-World Benchmarks
The primary empirical evidence (Tables 1, 2, 3, and 4) is derived entirely from **custom-designed synthetic simulators** (the "Coupled Model II Landscape" and "Stage-wise Modular Transition Landscape") rather than real-world deep neural networks evaluated on standard machine learning benchmarks.
- The emulators model a 12-layer, 4-task optimization space using hardcoded sensitivity parameters ($A_k^{(l)}$) and spatial coupling factors ($\rho=0.5$). 
- Using a custom-built simulator introduces an enormous risk of **confirmation bias**. Because the authors built the simulator, they explicitly designed its loss functions, spatial couplings, and target trajectories to mathematically reflect the exact assumptions of their proposed regularizer. This creates a highly circular experimental environment.
- Although the authors introduce a "Decoupled Isotropic Euclidean" metric to break mathematical circularity in the evaluation, the underlying optimization landscape itself remains simulated. A 1D simulated toy coordinate space cannot capture the high-dimensional, highly non-convex, extremely complex, and noisy optimization dynamics of actual deep neural networks (such as a 7B LLM or 86M ViT).
- Evaluating a new optimization algorithm primarily on synthetic 1D emulators is considered highly sub-standard and insufficient for top-tier venues.

### Weakness 2: Toy Scale and Methodological Concerns in BERT/ViT Pilot Studies
To counter potential criticisms regarding the lack of real-world validation, the authors include two "real-world pilot studies" in Section 4.5 using `bert-base-uncased` (110M parameters) and `vit-base-patch16-224` (86M parameters). A close inspection of these pilot studies reveals that they are **highly toy-like, statistically insignificant, and raise serious methodological concerns**:
1. **Suspiciously Homogeneous Curvatures in BERT:** The authors state that prior to TTA, they pre-computed the diagonal FIM trace across the 12 transformer blocks of BERT-Base and obtained **"normalized base curvatures of $c_l = 1.0000$ across all 12 encoder layers"**. This is mathematically highly improbable for any real, fine-tuned transformer. In modern architectures, different layers (e.g., early self-attention vs. late task-specific heads) have wildly different parameter sensitivities and gradient norms. If the normalized curvatures are all exactly 1.0000, then the curvature weights $\sqrt{c_l c_{l-1}}$ are also exactly 1.0000 across all transitions. Under this condition, RCR-Merge mathematically collapses to standard flat Total Variation! This means the BERT-Base pilot did not actually test the core "curvature-weighted" contribution of the paper; it simply evaluated standard flat TV.
2. **Suspiciously Clean "Toy" Accuracy Numbers:** The authors report that unconstrained AdaMerging collapses BERT Task 2 accuracy to **50.00%** (average **75.00%**), while RCR-Merge maintains **100.00%** accuracy. These clean, perfect percentages (100.00%, 75.00%, 50.00%) are classic indicators of an extremely tiny, toy test set (consisting of exactly **4 samples** or even **2 samples**). Similarly, for ViT-B/16, the reported accuracies are **65.00%**, **35.00%**, **55.00%**, and **57.50%** (which are all multiples of 2.5% and 5.0%, suggesting a test set of exactly **20 samples** or **40 samples**). Evaluating test-time adaptation on as few as 4 to 20 samples is statistically meaningless and cannot be presented as serious scientific evidence of "real-world prevention of representation collapse."
3. **Suspicious CPU Runtime:** The authors state that "The entire ViT-B/16 pilot study executes in less than 15 seconds on a standard CPU." Executing a multi-step backpropagation adaptation loop over an 86M parameter model using PyTorch functional calls on a standard CPU in 15 seconds is only possible if the dataset is virtually non-existent (e.g., 1 or 2 forward-backward steps).

The extreme toy scale of these pilot studies confirms that the paper **completely lacks a rigorous, standard empirical evaluation on real-world datasets**.

### Weakness 3: Tautological and Vacuous Theoretical Guarantees
The paper presents Lemma 3.1 and Theorem 3.2 as formal theoretical guarantees, but their mathematical and scientific utility is highly questionable:
1. **Triviality of Lemma 3.1:** Lemma 3.1 is presented as a formal coordinate-level theoretical guarantee proving that RCR-TV acts as an "analytical barrier" blocking wild jumps in sensitive layers. However, the lemma is **conceptually trivial and almost tautological**. The proof relies on the fact that the optimized loss is bounded by the initial loss: $\mathcal{L}_{\text{joint}}(\boldsymbol{\lambda}^*) \le \mathcal{L}_{\text{joint}}(\boldsymbol{\lambda}_0)$. Since all terms in the objective are non-negative, any single quadratic penalty term in the regularizer must be smaller than the total loss: $\beta \sqrt{c_l c_{l-1}} (\lambda^*_{k, l} - \lambda^*_{k, l-1})^2 \le \mathcal{L}_{\text{joint}}(\boldsymbol{\lambda}^*) \le \mathcal{L}_{\text{joint}}(\boldsymbol{\lambda}_0)$. Rearranging this inequality yields the bound in Equation 24. This bound is an **algebraic identity** that holds for *any* penalty function added to an objective. Framing this basic optimization property as a "deep" coordinate barrier or a "Riemannian geometric" guarantee is highly misleading.
2. **Vacuousness of Theorem 3.2:** Theorem 3.2's global output representation drift bound in Equation 17 features the term $\Lambda^{L-l}$, representing the cumulative product of layer-wise Lipschitz constants. As the authors explicitly acknowledge, for any realistic network depth $L$ (e.g., $L=12$ for BERT-Base, let alone larger models), this bound grows exponentially and becomes **practically vacuous** (representing an astronomical, loose bound that provides zero practical quantitative constraint). Moreover, the proof relies on the convenient assumption that the layer activation's coordinate Lipschitz constant is bounded by the square root of the local FIM trace ($K_l \le S \sqrt{c_l}$), which is not a mathematically proven property of deep neural networks.

The theoretical contributions feel like mathematical embellishments designed to inflate the paper's academic appearance rather than provide meaningful quantitative bounds.

### Weakness 4: Rhetorical Over-engineering of Geometric Framing
The paper employs heavy Riemannian geometry and spectral graph theory terminology ("conformal flat metric space," "local charts," "Riemannian manifold," "Laplacian smoothing filter") to describe what is ultimately a standard layer-wise weighted spatial Total Variation penalty. Because they assume a block-diagonal scalar approximation, the "metric tensor" $G(\theta)$ reduces to a static, diagonal matrix. This means the parameter space is treated as a flat Euclidean space with simple coordinate-wise scaling. There are no actual Riemannian manifold operations (such as calculating Christoffel symbols, geodesics, exponential maps, or parallel transport). The "Riemannian geometry" is essentially a rhetorical framework for a *layer-wise weighted Euclidean norm*. The authors should tone down this elaborate framing and present their method more groundedly as a Fisher-weighted spatial TV regularizer.

---

## 4. Ratings on Key Dimensions

### Soundness: Fair
The proposed pipeline is mathematically structured, and the PyTorch recipe is clear. However, the soundness is limited by the severe block-diagonal scalar approximation of the Fisher Information, the static metric assumption, the triviality of Lemma 3.1, the vacuousness of Theorem 3.2, and the extreme toy scale of the BERT and ViT "pilot studies" (which lack statistical validity).

### Presentation: Excellent
The paper is exceptionally well-written, clearly structured, and highly polished. The notation is clean, and the visual figures and tables are exemplary. The authors are also highly transparent about evaluation circularity and explicitly design uncoupled metrics to break it.

### Significance: Fair
The conceptual focus on spatial smoothing in model merging and the identification of the Overfitting-Optimizer Paradox are interesting. However, because the empirical validation is confined to custom synthetic emulators and statistically insignificant pilot studies, the actual significance and practical utility of the proposed method for real-world machine learning tasks are currently unverified.

### Originality: Good
The paper provides an original and creative combination of existing techniques (unsupervised TTA entropy minimization, diagonal Fisher-based sensitivity estimation, 1D Total Variation regularization, and gradient norm balancing). While the individual components are not new, their integration into a unified pipeline for test-time model merging is novel.

---

## 5. Overall Recommendation

**Score: 2 (Reject)**

**Justification:**
While the paper is exceptionally well-written and conceptually interesting, it fails to meet the empirical and theoretical standards of a top-tier machine learning conference. The primary quantitative results are confined to custom-designed 1D synthetic emulators, which introduces a high risk of confirmation bias. The "real-world" pilot studies are conducted on statistically insignificant toy datasets (likely 4 to 20 samples), and the reported homogeneous curvatures ($c_l = 1.0000$) mathematically collapse the proposed method to standard flat Total Variation. Furthermore, the theoretical guarantees are either algebraic tautologies of standard penalty methods (Lemma 3.1) or practically vacuous (Theorem 3.2). 

To be ready for publication, the paper requires a thorough revision that tones down the over-engineered geometric framing and replaces the synthetic/toy evaluations with a standard, rigorous empirical validation on standard benchmarks (e.g., Source-Free Domain Adaptation, out-of-distribution streaming on ImageNet-C/R, or multi-task model merging on GLUE) using full-scale models and datasets.

---

## 6. Constructive Questions and Suggestions for Revision

1. **Conduct Standard Real-World Empirical Evaluations:** The absolute priority for the authors is to evaluate RCR-Merge on standard, large-scale benchmarks. For vision, they should evaluate on ImageNet-C corruptions and ImageNet-R distribution shifts using Vision Transformers (ViT) and ResNets. For language, they should evaluate on multi-task model merging benchmarks (such as merging GLUE or MMLU task-specific experts) under non-stationary online data streams.
2. **Re-evaluate BERT/ViT Curvatures and Report Real Statistics:** In Section 4.5, please explain why the normalized base curvatures of BERT-Base were exactly $c_l = 1.0000$ across all 12 encoder layers. If this was a placeholder, please compute and report the actual estimated FIM traces. If they were indeed calculated, please check the implementation, as real transformer layers should exhibit wildly heterogeneous sensitivities. Also, please report the exact number of test samples used in these pilot studies and conduct evaluations on standard-sized test sets (e.g., full validation sets of SST-2, MNLI, etc.) to report statistically significant results.
3. **Tone Down the Geometric and Spectral Jargon:** The paper would be significantly stronger if the authors toned down the high-level Riemannian geometry and spectral graph theory framing. Presenting the method as a "Fisher-weighted spatial Total Variation regularizer" is scientifically accurate, highly grounded, and easier for readers to digest without creating a gap between the mathematical theory and the actual implementation.
4. **Revise the Theoretical Guarantees:** 
   - Acknowledge that Lemma 3.1 is a standard algebraic consequence of penalty methods and re-frame it accordingly rather than presenting it as a deep, unique coordinate barrier.
   - Either remove the global representation drift bound in Theorem 3.2 (since it is practically vacuous) or replace it with a tight local bound that does not grow exponentially with network depth.
5. **Clarify GNB Gradient Computations:** In Listing 1, please clarify whether the FIM trace estimation uses the true Fisher (by sampling targets from the model's predictive distribution) or the empirical Fisher (using the argmax or predicted pseudo-labels), as this distinction is vital for practitioners attempting to reproduce your results.
