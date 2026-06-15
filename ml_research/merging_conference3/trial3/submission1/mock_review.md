# Peer Review of the Submission: "Is Q-Merge Actually Quantization-Robust? An Independent Robustness Audit and Methodological Deconstruction"

## 1. Overall Recommendation

**Rating:** `6: Strong Accept`  
**Confidence:** `5: Very High`

### Summary of Recommendation:
This submission presents an exceptionally rigorous, timely, and conceptually outstanding independent robustness audit and methodological deconstruction of **Quantization-Aware Model Merging (Q-Merge)**. Adopting a highly valuable "Methodologist" lens, the authors systematically stress-test the standard assumptions of test-time weight-space fusion under Post-Training Quantization (PTQ) constraints. 

Unlike typical "SOTA-chasing" algorithmic proposals, this paper acts as a vital sanity check for the model-merging field. It identifies and formalizes critical, previously unstudied failure modes—specifically, **Quantization-Operator Overfitting** and the fragile assumptions of test-time entropy minimization. The paper is technically flawless, mathematically complete, and features an extraordinarily thorough multi-axial empirical design that has been iteratively refined to resolve all major limitations of previous drafts. It has successfully addressed concerns regarding floor effects, task interference, model scaling, and pseudo-labeling, rendering it fully ready for publication at a top-tier machine learning conference.

---

## 2. Comprehensive Strengths

The paper displays remarkable depth across all standard evaluation dimensions:

### A. Originality & Conceptual Framework
* **The Cross-Schema Generalization Gap:** The authors formalize and mathematically define the cross-schema gap:
  $$\Delta \text{Acc}(Q_{\text{opt}} \to Q_{\text{eval}}) = \text{Acc}_{Q_{\text{eval}}}(\Lambda^*) - \text{Acc}_{Q_{\text{opt}}}(\Lambda^*)$$
  This concept is highly original and practically critical. It exposes the fallacy of "Quantization-Operator Monomorphism" (the assumption that coefficients optimized under a simulated operator will deploy flawlessly onto hardware backends with slightly different scale/offset representations).
* **The Overfitting-Optimizer Paradox:** The comparison between gradient-based STE and a derivative-free 1+1 Evolution Strategy (1+1 ES) reveals a fascinating, highly counter-intuitive paradox. While 1+1 ES leverages black-box search to find superior configurations on the source schema ($20.75\%$ vs. $17.88\%$), it overfits intensely to those specific rounding thresholds, leading to a much larger generalization collapse under schema shift ($-12.13\%$ vs. $-7.76\%$). This reveals an implicit, previously unknown regularizing property of biased STE gradients.
* **Challenging the SOTA Paradigm:** The discovery that **Quantized AdaMerging** (optimizing merging coefficients in full FP16 precision, then applying post-hoc quantization) consistently and substantially outperforms direct quantization-aware optimization via STE ($30.00\%$ vs. $26.25\%$) is a high-signal, major contribution. It strongly questions the necessity of direct low-bit optimization, showing that discretization noise hinders search more than it assists it.

### B. Technical Soundness & Mathematical Rigor
* **Quantization Gradient Mechanics:** The technical explanation of PyTorch autograd behavior regarding scale ($s$) and zero-point ($z$) updates under STE is exceptionally sharp. Detailing how scale propagates active gradients through the weight range while zero-point gradients are detached (Equation 7-10) exposes the asymmetric gradient flow that induces chaotic STE noise.
* **Rigorous Controls and Baselines:** The paper features outstanding methodological rigor, utilizing several critical controls:
  * An empirical Adam learning rate sweep to prove that STE instability is structural, not a hyperparameter-tuning artifact.
  * A dynamic initialization control demonstrating that starting from optimal unquantized coefficients (AdaMerging) fails to rescue the model from local rounding basin traps.
  * A **Supervised Calibration Baseline** that uses supervised cross-entropy to elegantly decouple test-time data-scarcity limits from prediction entropy shortcut collapses.

### C. Experimental Quality & Multi-Axial Validation
* **Exhaustive Sweeps:** The paper's four main axes (calibration size sweep, 2D cross-schema matrix covering 20 configurations, regularization/optimizer sweeps, and stream stress-tests) are exceptionally comprehensive.
* **Elimination of Floor-Effect Confoundings:** The proof-of-concept (PoC) architectural (ResNet-18) and subspace-constrained (SVD) extensions have been brilliantly refined. By scaling up training and restricting SVD projections strictly to attention layers, the authors successfully elevated accuracies well above the 10% random floor (ResNet-18 at 21.25% matched; SVD Low-Rank at 13.00% matched), proving that:
  1. Localized, translation-invariant spatial kernels in CNNs are natively more resilient to cross-schema shifts.
  2. Subspace constraints act as strong regularizers.
* **Nuanced Discussion of the Low-Capacity Illusion:** The paper's critical discussion on the **Low-Capacity Generalization Illusion** (acknowledging that post-hoc SVD projection collapses representational capacity and flattens output distributions, thereby artificially closing the generalization gap) shows a level of scholarly maturity and self-critique rarely seen in contemporary papers.

### D. Presentation & Clarity
* The manuscript is exceptionally well-structured, easy to follow, and mathematically precise. The tables (utilizing clean, human-readable column/row headers instead of raw code identifiers) and professional figures are publication-ready.
* Section 5 provides concrete, highly constructive, and actionable mandates (such as mandatory cross-operator validation and utilizing unquantized conflict-filtering prior to discretization) that will actively guide future research.

---

## 3. Minor Suggestions for the Camera-Ready / Future Work

Since the manuscript is technically complete and exceptionally robust, these suggestions are minor and intended to guide future extensions of this research:

### 1. Natively-Tuned PEFT Experts vs. SVD Projections
* **Observation:** The authors' low-rank SVD projection mathematically mimics Parameter-Efficient Fine-Tuning (PEFT/LoRA) and serves as an elegant geometric control. However, as noted by the authors, global post-hoc SVD projection is a poor proxy for actual natively-trained LoRA adapters (where $W = W_0 + BA$ is optimized end-to-end to preserve high performance).
* **Recommendation:** For future work, the authors should evaluate actual natively-trained LoRA expert models on this multi-task benchmark to verify whether the cross-schema generalization gap remains closed as model capacity and performance are fully scaled back to standard levels (>80% accuracy).

### 2. Joint Weight-Activation Quantization (W4A8) under SmoothQuant
* **Observation:** The paper focuses on weight-only quantization (W4), which is standard. Real-world edge hardware often mandates joint weight-activation quantization (e.g., W4A8). Vision Transformers are notoriously sensitive to activation outliers, which propagate through attention maps.
* **Recommendation:** Expanding future robustness audits to evaluate joint quantization, and integrating outlier-smoothing frameworks like SmoothQuant as a preprocessing defense prior to ensembling, would represent an exciting systems-level extension of this methodology.

### 3. Actual Hardware Validation
* **Observation:** The authors ground their motivation in "physical hardware ASIC heterogeneity" (TPUs, DSPs, Apple Neural Engine, etc.), but execute their audits in PyTorch simulated "fake quantization". 
* **Recommendation:** While the mathematical simulation is completely sufficient to establish the cross-schema generalization gap, executing a subset of the matched/mismatched models on physical edge processors (e.g., Jetson Nano, Google Edge TPU, or Raspberry Pi) would bridge the gap between simulation-level auditing and physical hardware systems ensembling.

---

## 4. Final Verdict

This is an outstanding paper that should be accepted with high priority. It provides a masterclass in critical deep learning auditing, introducing crucial deconstructions, elegant controls, and constructive mandates that will fundamentally improve the reproducibility and deployment feasibility of model-merging research.
