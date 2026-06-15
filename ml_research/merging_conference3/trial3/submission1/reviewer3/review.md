# Peer Review

## Summary of the Paper
This paper presents a highly rigorous, independent robustness audit and methodological deconstruction of **Q-Merge**, a state-of-the-art framework for quantization-aware model merging. Model merging aims to combine task-specific expert neural networks fine-tuned from a shared backbone into a single model, preserving multi-task performance without training or inference latency overhead. Naive post-hoc quantization of these merged models (Merge-then-Quantize, or M-then-Q) often causes catastrophic degradation at low bit-widths. Q-Merge attempts to solve this by optimizing layer-wise merging coefficients directly under simulated post-training quantization (PTQ) constraints using the Straight-Through Estimator (STE) under an unsupervised prediction entropy minimization objective.

Using a standardized pre-trained Vision Transformer backbone (`vit_tiny_patch16_224`, 5.7M parameters) with four classification heads trained on MNIST, FashionMNIST, CIFAR-10, and SVHN, the authors systematically deconstruct Q-Merge along four distinct axes:
1. **Axis 1: Calibration Stream Size Sweep:** Sweeping $N \in \{1, 4, 16, 64\}$ to isolate transductive overfitting and performance plateauing.
2. **Axis 2: Cross-Schema Generalization Matrix:** Optimizing coefficients under a source schema $Q_{\text{opt}}$ but evaluating under five target schemas $Q_{\text{eval}}$ (symmetric/asymmetric, per-tensor/per-channel, and double quantization).
3. **Axis 3: Spatial Regularization & Derivative-Free Search:** Evaluating Total Variation spatial smoothing and derivative-free 1+1 Evolution Strategy (1+1 ES).
4. **Axis 4: Stream Distortion and Skew Robustness:** Stress-testing optimization under input-space Gaussian corruptions and Gini class skewness.

Crucially, the paper introduces a supervised calibration baseline, architectural generalizability extensions (ResNet-18), and low-rank SVD subspace projections, and formalizes a highly original **Hybrid Optimization Pipeline (Algorithm 1)** to guide the community toward deployment-robust weight-space fusion.

---

## Strengths and Weaknesses

### Strengths
1. **Outstanding Conceptual Originality:** Rather than introducing another incremental merging heuristic, this paper presents a paradigm-challenging critical audit. It identifies and deconstructs three unstudied assumptions in the literature: Quantization-Operator Monomorphism, Calibration Stream Purity, and STE Gradient Path Fidelity. This changes how the community must think about test-time weight fusion under compression.
2. **Exposing the Cross-Schema Generalization Gap:** The paper coins and demonstrates the **Cross-Schema Generalization Gap**—a previously undocumented phenomenon where merging coefficients overfit intensely to the exact rounding boundaries of the simulated source operator. Evaluating on heterogeneous hardware-relevant target schemas collapses performance to near random-guess levels (e.g., `sym\_channel` to `sym\_tensor` collapses accuracy by $-7.75\%$; `asym\_channel` to `sym\_tensor` collapses performance by a catastrophic $-20.37\%$). This is a major deployment warning for heterogeneous edge hardware.
3. **Rigorous Baseline Isolation:** The inclusion of **Quantized AdaMerging** (FP16 search + post-hoc quantization) is a brilliant methodological decision. It reveals that unquantized search consistently outperforms direct quantization-aware optimization via STE ($30.00\%$ vs $26.25\%$), proving that STE-induced gradient noise damages weight-space search and that direct low-bit optimization is unnecessary.
4. **The "Low-Capacity Generalization Illusion":** The paper offers a highly original critique of parameter-efficient ensembling spaces. It shows that although low-rank SVD projections close the cross-schema generalization gap ($+0.50\%$), it is an illusion: the global SVD projection destroys vital representation directions and flattens activation ranges, making the model naturally insensitive to quantization parameters because its predictions are already highly degraded ($13.00\%$ absolute accuracy).
5. **Constructive and Actionable Solutions:** The paper does not merely criticize; it provides a concrete, mathematically formalized **Hybrid Optimization Pipeline (Algorithm 1)** that sequentializes first-order coarse search (STE) with zero-order fine-grained search (1+1 ES) under spatial smoothing. It also outlines promising directions like smoothing the quantized loss landscape prior to discretization using parameter filtering techniques (TIES-Merging/DARE) or pre-merging alignment constraints.
6. **Exhaustive Empirical Validation:** Every claim is verified with complete tables reporting means and standard deviations computed over three random seeds. The authors evaluate under diverse operators, calibrate sizes, noise/skew corruptions, and architectures (CNNs vs ViTs).

### Weaknesses
1. **PEFT/LoRA Experts Verification:** To represent Parameter-Efficient Fine-Tuning (PEFT), the authors apply a global, post-hoc SVD projection to compress task vectors into a rank-4 subspace. As they transparently note, this collapses absolute performance to $13.00\%$. While they identify this as the "Low-Capacity Generalization Illusion," verifying the Cross-Schema Generalization Gap on natively-tuned LoRA experts (which preserve model capacity) remains a critical future empirical step to fully validate the "PEFT/subspace restriction" defense.
2. **Joint Quantization Scope:** The empirical evaluations focus on weight-only quantization (W4). Real-world edge hardware often mandates joint weight-activation quantization (e.g., W4A8 or W4A4) where attention activation outliers propagate noise through softmax layers, which would be a highly valuable addition to the audit's scope.

---

## Soundness
**Rating: Excellent**

The paper is technically flawless and methodologically sound. The mathematical formulations are complete, clear, and mathematically rigorous. The authors describe PTQ scale and zero-point dynamic updates, explain the asymmetric gradient flow (autograd propagating through scales but blind to non-differentiable zero-points), and formalize expectation-based randomized smoothing under input noise. The baseline configurations (Quantized AdaMerging, Supervised Calibration) are highly appropriate and isolate the independent variables with precision. The authors are transparent about limitations, carefully detailing the scale-up computational bottlenecks and the SVD proxy limits. All findings are backed by robust statistical validation over multiple seeds.

---

## Presentation
**Rating: Excellent**

The presentation quality is exceptional. The narrative is cohesive, engaging, and structured with logical precision. Key terms and concepts like "Cross-Schema Generalization Gap," "Low-Capacity Generalization Illusion," and "Quantization-Operator Monomorphism" are clearly conceptualized and highlighted. Mathematical equations are formatted flawlessly. Tables and figures are professional and integrated perfectly to support the text, including a comprehensive pseudocode layout for the proposed Hybrid Optimization Pipeline (Algorithm 1).

---

## Significance
**Rating: Excellent**

The significance of this work is exceptionally high. It exposes critical blind spots in post-training quantization-aware merging that directly affect the deployment of multi-task models on physical edge hardware (e.g., TPUs, DSPs, NPUs). The paper will likely shift the entire community's evaluation standard, establishing mandatory cross-operator validation and calibration stream heterogeneity audits. Furthermore, the proposed Hybrid Optimization Pipeline and pre-discretization landscape smoothing suggestions provide a vital blueprint for future research in robust deep learning model compression.

---

## Originality
**Rating: Excellent**

The paper exhibits outstanding originality. Rather than presenting an incremental parameter-tuning modification on existing merging methods, the authors construct a highly ambitious, first-of-its-kind multi-axial robustness audit. Demolishing the unstudied assumption of quantization monomorphism, establishing the Cross-Schema Generalization Gap, showing that unquantized search is superior to STE, and unmasking the PEFT "Low-Capacity Generalization Illusion" are highly novel, paradigm-shifting insights that fundamentally change how weight-space consolidation must be evaluated.

---

## Overall Recommendation
**Recommendation: 6: Strong Accept**

**Justification:** 
This is a technically flawless, exceptionally written, and highly ambitious paper that makes a significant conceptual contribution to the field of deep learning model compression and model merging. By introducing the first multi-axial robustness audit and independent deconstruction of quantization-aware model merging, the authors identify and explain profound, previously undocumented vulnerabilities—such as the Cross-Schema Generalization Gap and the fragility of unsupervised entropy minimization under realistic class skew. 

The findings are highly surprising and challenge the core premises of direct low-bit optimization, showing that full-precision search (Quantized AdaMerging) consistently outperforms STE search. Rather than being purely critical, the authors serve the community constructively by providing a formalized Hybrid Optimization Pipeline (Algorithm 1), analyzing expected gradients as randomized smoothing, and proposing pre-discretization landscape smoothing (via TIES-Merging/DARE). This work represents a major paradigm shift that will establish a more honest, rigorous, and deployment-realistic evaluation standard for weight-space fusion. It is a clear Strong Accept.
