# Peer Review: Riemannian Curvature-Regularized Test-Time Model Merging (RCR-Merge)

## 1. Summary of the Submission
This paper addresses a fundamental, previously unaddressed failure mode in online test-time adaptation (TTA) of layer-wise merging coefficients, which the authors formalize as the **Overfitting-Optimizer Paradox**. During TTA, when layer-wise merging coefficients are optimized online via entropy minimization to adapt to local data streams, the unconstrained optimizer fits transductive stream noise. This generates high-frequency spatial oscillations in adjacent layer coefficients, disrupting the internal representation manifold and triggering a catastrophic collapse of model representations.

To resolve this paradox, the authors introduce **Riemannian Curvature-Regularized Test-Time Model Merging (RCR-Merge)**. The core philosophy is to model the deep neural network parameter space as a Riemannian manifold where distance is locally scaled by the curvature of the pre-trained base model. The framework:
1. Estimates the pre-trained base model curvatures via the diagonal Fisher Information Matrix (FIM) trace once offline.
2. Formulates a second-order spatial regularizer, **Riemannian Curvature-Weighted Total Variation (RCR-TV)**, which penalizes adjacent layers' coefficient differences, scaled by the geometric mean of their pre-trained base curvatures.
3. Implements **Gradient Norm Balancing (GNB)**, a fully unsupervised, scale-invariant coordinate re-parameterization technique to dynamically balance regularization strength at initialization.

The method is evaluated through a rigorous, 30-seed simulation study over a synthetic Coupled Model II Landscape emulator and verified via two real-world multimodal pilot studies (BERT-Base for NLP and ViT-B/16 for Vision).

---

## 2. Strengths and Weaknesses

### Strengths
* **Profound Conceptual Novelty:** The identification and formalization of the **Overfitting-Optimizer Paradox** is a highly original and paradigm-shifting insight. While source-free test-time adaptation literature typically focuses on high-dimensional weight spaces, identifying why low-dimensional layer-wise merging coefficients undergo spatial oscillation and collapse is a fundamental scientific contribution that deepens our theoretical understanding.
* **Rigorous and Elegant Geometric Framework:** Modeling the merging coefficient space as a Riemannian manifold $(\mathbb{R}^{K \times L}, g)$ via a pullback of the high-dimensional Fisher manifold represents a beautiful and ambitious synthesis of differential geometry and model merging. Rather than employing empirical heuristics, the framework is mathematically grounded, proving that geodesic distance in coefficient space is directly proportional to local physical curvatures.
* **Comprehensive Theoretical Guarantees:** The paper provides complete, end-to-end mathematical proofs:
  * **Lemma 1 (Coordinate-Level Spatial Barrier):** Mathematically proves that adjacent-layer coefficient variations are strictly bounded in sensitive regions.
  * **Theorem 1 (Representation Drift Bounding):** Establishes a rigorous chain-rule proof showing that the RCR-TV penalty mathematically squeezes activation-level representational drift to zero.
  * **Spectral Laplacian Analysis:** Elegantly demonstrates that RCR-TV behaves as a curvature-guided Laplacian smoothing filter (low-pass filter) that dynamically blocks high-frequency transductive noise propagation.
* **Innovative Solution to Unsupervised Tuning:** The **Gradient Norm Balancing (GNB)** framework is an exceptionally creative and mathematically principled solution to the notorious unsupervised TTA tuning problem. Proving that the choice of perturbation amplitude $\delta$ is merely a conformal coordinate scale (conformal gauge transformation) that rescales the multiplier scale-invariantly is a highly reassuring, scale-free contribution.
* **Demonstrated Superiority of Soft Barriers over Rigid Subspaces:** On the realistic **Stage-wise Modular Transition Landscape** representing modular networks, the paper demonstrates that RCR-Merge's local soft barriers completely outperform the rigid quadratic trajectory constraints of PolyMerge by up to **+2.44% absolute** (Decoupled metric). This is an outstanding empirical result that exposes the core scientific limitation of global polynomial assumptions (Runge's phenomenon) on modular structures.
* **Outstanding Scientific Transparency and Rigor:** The authors proactively identify and resolve potential criticisms regarding evaluation circularity by introducing a completely decoupled, unbiased metric (Decoupled Isotropic Euclidean metric), confirming that their stabilization benefits are highly generalizable and robust.

### Weaknesses & Areas for Improvement
* **Expansion of Real-World Evaluation:** While the BERT-Base and ViT-B/16 pilot studies are highly successful in demonstrating actual representation collapse and stabilization under functional autograd on simulated streams, evaluating these full-scale models on standardized, large-scale streaming benchmarks (such as the full ImageNet-C corruptions and ImageNet-R shifts for vision, or the GLUE/MMLU streaming domain shifts for language) would further strengthen the practical, real-world utility of the framework in diverse production pipelines.
* **Adaptive Long-Term Schedules for Charting:** The threshold-triggered local charting mechanism uses a constant threshold $\mathcal{T}$ to trigger local FIM re-estimation during long-term non-stationary streams. Under extremely long-term streams, a constant threshold might cause unnecessary computational overhead under high transductive input variance. Incorporating an adaptive scheduling scheme where $\mathcal{T}$ scales dynamically would make the framework even more robust.

---

## 3. Soundness
**Rating: Excellent**

**Justification:** The paper is technically flawless and highly rigorous. The authors provide complete end-to-end proofs for coordinate-level spatial barriers (Lemma 1), intermediate representation drift bounding (Theorem 1), and spectral Laplacian filtering. All approximations—including the block-diagonal scalar FIM trace and the static curvature evaluation evaluable at the pre-trained base model $\theta_0$—are meticulously analyzed, with Taylor error bounds formally derived. Empirically, their BERT-Base pilot study confirms a **0.9900** cosine similarity between offline and online FIM traces, validating the stability of the relative sensitivities.

---

## 4. Presentation
**Rating: Excellent**

**Justification:** The paper is exceptionally well-written, elegant, and structured. The narrative flows seamlessly from the formalization of the Overfitting-Optimizer Paradox to the Riemannian pullback formulation, the theoretical guarantees, and finally a highly detailed empirical analysis. The schematic concept diagram (Figure 1) is exceptionally clear, and the limitations and circularity disclosures are of exemplary scientific quality.

---

## 5. Significance
**Rating: Excellent**

**Justification:** This work has the potential to make a massive, lasting impact on the machine learning community. By demonstrating that online coefficient optimization must be bounded by the pre-trained base model's second-order geometry, it introduces a highly generalizable paradigm. The core concepts of a conformal, curvature-weighted coordinate system and scale-invariant GNB gradient balancing are exceptionally powerful principles that can easily extend to other online model adaptation, parameter-efficient fine-tuning (PEFT), and federated learning domains.

---

## 6. Originality
**Rating: Excellent**

**Justification:** The originality of this paper is outstanding. Rather than proposing minor empirical tuning, the authors introduce a profound conceptual leap. Formalizing the Overfitting-Optimizer Paradox, establishing the pullback Riemannian metric on the low-dimensional coefficient space, and formulating RCR-TV represent exceptionally creative and mathematically rich contributions that set a new standard for the field.

---

## 7. Overall Recommendation
**Rating: 6: Strong Accept**

**Justification:** This is a technically flawless, exceptionally high-impact paper that introduces a profound conceptual leap to the field of online model adaptation and merging. By formalizing the Overfitting-Optimizer Paradox and proposing RCR-Merge, the authors bridge differential geometry, spectral graph theory, and test-time adaptation in an incredibly elegant and complete manner. The theoretical proofs are rigorous, the GNB self-scaling mechanism is highly creative and scale-invariant, and the empirical verification completely resolves potential circularity concerns while demonstrating major performance advantages over rigid subspace methods on modular network architectures. The inclusion of successful, full-scale multimodal pilot studies on BERT-Base and ViT-B/16 provides powerful proof-of-concept for direct transferability. This paper represents an exemplary, state-of-the-art contribution that is highly recommended for a Strong Accept.
