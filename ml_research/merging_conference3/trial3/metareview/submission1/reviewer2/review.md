# Peer Review Report

## Summary of the Paper
This paper presents a rigorous, independent critical audit and methodological deconstruction of the emerging paradigm of **Quantization-Aware Model Merging** (primarily exemplified by Q-Merge). The authors challenge the over-optimistic "near-lossless" claims in the literature, which typically evaluate optimized low-bit merging configurations under highly idealized assumptions. 

Using a standardized pre-trained Vision Transformer backbone (`timm ViT-Tiny`) fine-tuned on four diverse classification tasks (MNIST, FashionMNIST, CIFAR-10, SVHN), the paper systematically analyzes the behavior of quantization-aware model merging along four distinct axes:
1. **Calibration Stream Size Sweep:** Disclosing that direct low-bit optimization via Straight-Through Estimators (STE) is consistently and substantially outperformed by a full-precision search baseline (*Quantized AdaMerging*) by a $3.75\%$ absolute average margin ($30.00\%$ vs $26.25\%$).
2. **Cross-Schema Generalization Matrix:** Revealing that learned merging coefficients overfit intensely to the exact mathematical operator used during optimization, causing performance to collapse to random-guess levels (approx. 10%) under target hardware-schema shifts (e.g., channel-wise to tensor-wise).
3. **Spatial Regularization vs. Black-Box Search:** Evaluating Elastic Spatial Regularization (Total Variation smoothing) and a derivative-free optimizer (1+1 Evolution Strategy). While 1+1 ES achieves better performance on the source schema ($20.75\%$ vs $17.88\%$ for STE), it suffers from severe generalization collapse ($8.62\%$ accuracy) on mismatched targets due to boundary overfitting.
4. **Stream Distortion and Skew Robustness:** Stress-testing optimization under input noise and class imbalance, revealing extreme vulnerability to label skew under unsupervised entropy minimization objectives, and noting an accidental regularization effect of input Gaussian noise.

The paper also presents a supervised calibration baseline to decouple data scarcity from unsupervised entropy collapse, and provides proof-of-concept empirical extensions on convolutional neural networks (ResNet-18) and subspace-constrained (low-rank global SVD projection) merging.

---

## Strengths and Weaknesses

### 1. Soundness
* **Strengths (Empirical Soundness):** The empirical methodology is exceptionally sound, well-controlled, and transparent. The decoupling of optimization search from quantization constraints via the *Quantized AdaMerging* baseline is a major strength. The use of multiple seeds to report mean and standard deviation ensures statistical significance. The supervised calibration baseline is an outstanding methodological addition to isolate the cause of performance collapse under class skew and data scarcity.
* **Weaknesses (Theoretical Soundness):** From a theoretical perspective, the soundness is limited. Although the paper uses mathematical notation extensively to formalize quantization operators and Straight-Through Estimator (STE) optimization, it does not provide any formal mathematical proofs, theorems, or convergence guarantees:
  - **No Operator Generalization Bounds:** The core claim of "Quantization-Operator Overfitting" is explained qualitatively and supported empirically. However, there is no formal theoretical framework or generalization bounds (e.g., using Lipschitz continuity, PAC-Bayesian theory, or Rademacher complexity) explaining mathematically why or under what conditions continuous merging coefficients fail to generalize across discrete operators.
  - **Heuristic Explanations for Gradient and Noise Effects:** The explanation for the "expectation-based gradient as a smooth surrogate" ($\mathbb{E}_{\eta} [\nabla \mathcal{L}]$) under Gaussian noise is a qualitative citation of randomized smoothing concepts, lacking a formal derivation showing its specific smoothing effect on a discontinuous, multi-task quantized merging landscape. Similarly, the "circular feedback loop" of scale recalculation is discussed heuristically without bounded error analysis.
  - **Unverified Hybrid Optimization Pipeline:** The proposed Hybrid Optimization Pipeline (Appendix B, Algorithm 1) is a heuristic engineering recipe. The authors do not provide any convergence proofs, theoretical validation, or empirical testing to support its viability.

### 2. Presentation
* **Strengths:** The paper is beautifully written, exceptionally structured, and easy to follow. The mathematical notation is clean and rigorous. The tables are highly detailed, and the references to visual aids are well-integrated. The authors are commendably transparent about the limitations of their own evaluations (e.g., calling out SVD projection as a poor PEFT proxy).
* **Weaknesses:** None of note. The presentation is outstanding.

### 3. Significance
* **Strengths:** The paper has high significance for the sub-area of weight-space fusion and edge-hardware deployment. The "Cross-Schema Generalization Gap" is a critical warning for practitioners who assume that simulated PyTorch fake-quantization translates seamlessly to physical ASICs or TPU chips with different rounding/scaling characteristics. The methodological recommendations in Section 5 provide useful directives for future research.
* **Weaknesses:** For a theoretically-oriented researcher, the significance is somewhat restricted. Because the paper does not develop new theoretical tools, generalizable bounds, or proven optimization guarantees, its contributions remain primarily empirical and domain-specific to model merging.

### 4. Originality
* **Strengths:** The conceptual framing of the audit is highly original. Defining the "Cross-Schema Generalization Gap," analyzing the "Low-Capacity Generalization Illusion" in SVD-constrained merging, and systematically exposing the over-optimism of monomorphic operator evaluations are highly creative and novel conceptual contributions.
* **Weaknesses:** The mathematical originality is incremental. The quantization operators, Straight-Through Estimator, 1+1 Evolution Strategy, and Total Variation spatial regularization are standard formulations compiled from existing post-training quantization and optimization literature.

---

## Detailed Ratings

### Soundness: Good
* **Justification:** The empirical soundness is excellent due to a controlled, multi-axial benchmark, exhaustive baselines, multiple random seeds, and brilliant decoupling of variables (e.g., supervised calibration and unquantized AdaMerging baselines). However, the theoretical soundness is limited by a complete lack of formal proofs, convergence bounds, or theoretical generalization guarantees for the observed failure modes or proposed hybrid optimizer.

### Presentation: Excellent
* **Justification:** The writing is incredibly clear, precise, and transparent. The paper is logically structured, transitions smoothly between sections, utilizes clean mathematical notation, and presents highly detailed tables and figures that support the arguments perfectly.

### Significance: Good
* **Justification:** The paper provides high-value deployment warnings and methodological mandates that can significantly elevate evaluation standards in weight-space fusion. However, its impact is largely domain-specific and empirical, rather than contributing new foundational mathematical machinery to the broader machine learning optimization literature.

### Originality: Good
* **Justification:** The work is conceptually highly novel, introducing original paradigms like Quantization-Operator Overfitting and the Low-Capacity Generalization Illusion. However, the mathematical components and tools utilized are standard and drawn directly from prior PTQ and evolutionary search work.

---

## Overall Recommendation

**Rating: 4 (Weak Accept)**

* **Justification:** This is a technically solid, exceptionally well-written, and thorough independent robustness audit of Quantization-Aware Model Merging. It successfully exposes critical, unstudied vulnerabilities—most notably catastrophic cross-operator collapse—that have significant real-world deployment implications. The empirical execution, baseline design, and methodological honesty are outstanding. However, the paper is strictly empirical. It explains complex non-smooth phenomena (like boundary overfitting, gradient noise, and randomized smoothing) using intuitive, qualitative heuristics rather than proving formal mathematical bounds, theorems, or convergence guarantees. It also proposes a "Hybrid Optimizer" in the appendix without any theoretical analysis or empirical validation. Resolving these theoretical gaps and executing actual natively-trained PEFT (LoRA) evaluations would significantly strengthen the paper, making it a valuable addition to the literature.

---

## Constructive Comments and Questions for the Authors

1. **Developing a Formal Theoretical Framework:** Can you formalize "Quantization-Operator Overfitting" mathematically? For example, can you model the quantization operator shift $Q_{\text{opt}} \to Q_{\text{eval}}$ as a boundable perturbation of the continuous loss landscape, and prove a theoretical limit on the performance drop $\Delta \text{Acc}(Q_{\text{opt}} \to Q_{\text{eval}})$ under specific parameter scaling conditions?
2. **STE Gradient Bias Analysis:** Rather than qualitatively attributing optimization failure to "gradient noise," can you mathematically analyze or bound the expectation of the difference between the true gradient and the Straight-Through Estimator approximation, $\mathbb{E} [ \nabla_{\Lambda} \mathcal{L} - \tilde{\nabla}_{\Lambda} \mathcal{L} ]$, on the continuous layer-coefficient search space?
3. **Natively-Trained PEFT/LoRA Evaluation:** The global post-hoc SVD task-vector projection used as a low-rank subspace proxy is highly destructive, degrading model capacity to $13.00\%$. This introduces a major confounding factor, as the apparent robustness could simply be a "Low-Capacity Generalization Illusion." Evaluating actual natively-trained LoRA experts (where high performance is preserved) would resolve this confounder and confirm whether low-intrinsic-dimension spaces actively stabilize cross-operator generalization.
4. **Empirical Validation of the Proposed Hybrid Optimizer:** You formalize an elegant Hybrid Optimization Pipeline in Appendix B (Algorithm 1) and advocate for it in Section 5. Why did you not implement or empirically test this algorithm? Providing empirical results showing that this pipeline successfully bridges the Cross-Schema Generalization Gap would drastically elevate the impact and significance of your work.
5. **Direct Medium-Scale Verification:** While your analytical scaling arguments in the conclusion are logically sound, do you have any empirical results on a medium-scale backbone (e.g., `vit-base-patch16-224` with 86M parameters or Pythia-70M) to verify if the Cross-Schema Generalization Gap indeed expands with model scale?
