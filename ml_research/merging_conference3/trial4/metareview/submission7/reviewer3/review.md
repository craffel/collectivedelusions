# Final Peer Review

## Summary of the Paper
This paper presents **SuiteMerge**, a systematic and independent methodological audit of the adaptive model-merging literature (specifically critiquing methods like AdaMerging and PolyMerge). Adaptive model merging has been proposed as a training-free approach to multi-task learning by interpolating the weights of task-specific expert models at test time (Test-Time Adaptation, or TTA) over incoming streams using unsupervised prediction entropy minimization. 

The authors uncover a severe, previously unreported confounding variable in this literature: **Task Suite Bias**. Most existing adaptive merging publications validate their algorithms on a single, arbitrary combination of four datasets (MNIST, FashionMNIST, CIFAR-10, SVHN) which masks fundamental failures. By systematically partitioning these tasks into five distinct suites designed along axes of domain distance and representational conflict, the authors demonstrate that unconstrained online TTA overfits to local transductive stream noise in high-conflict regimes.

As a highly practical and regularized alternative, the authors propose **Offline Few-Shot Validation Tuning (OFS-Tune)**. OFS-Tune restricts layer-wise coefficients to continuous low-degree polynomial trajectories (linear $d=1$ or quadratic $d=2$) and optimizes them offline using a tiny, stratified labeled validation set ($M=10$ samples per task) via Nelder-Mead search. This completely shifts optimization offline, requiring exactly **zero test-time compute, zero backpropagation latency, and zero edge-device energy consumption** at deployment, while naturally bypassing the privileged task-routing assumptions required by online TTA on mixed streams.

Through a calibrated mathematical simulation study over 30 independent seeds and independent physical weight-space deep network validation (on a 5-layer CNN on CPU), the authors prove that OFS-Tune consistently matches or exceeds the performance of online methods across all task relationships. It successfully avoids transductive stream-level noise overfitting, and behaves as a "safe-by-default" algorithm in high-conflict regimes where unconstrained online adaptation collapses.

---

## Strengths and Weaknesses

### Strengths
1. **Outstanding Practical Utility and Engineering Relevance:** Shifting the parameter optimization from a latency-intensive, power-hungry online TTA phase to a brief, pre-deployment offline calibration phase (OFS-Tune) is of massive value to practitioners. It completely eliminates deployment-time computational overhead, VRAM bottlenecks, backpropagation latency, and energy costs, making model merging highly viable for edge-device and commercial pipelines.
2. **Methodological Rigor and Exposing "Task Suite Bias":** Exposing how a monolithic single-suite benchmark masks localized failures is a crucial, refreshing "reality-check" for the community. The systematic partitioning into 5 distinct suites based on domain distance and representational conflict is an elegant, highly effective diagnostic framework.
3. **High Statistical Standards:** Evaluating all simulated results over **30 independent random seeds** and reporting standard deviations is an exceptional standard of rigor that far exceeds typical machine learning publications.
4. **Robust Isolation of Confounding Variables:**
   - The **OFS-Unconstrained** baseline successfully separates the role of validation-data access from structural trajectory constraints, proving that polynomial trajectory regularization is mathematically necessary to act as a low-pass noise filter.
   - The construction of symmetrical optimization budgets (**OFS-Tune via Adam** and **AdaMerging via L-BFGS-B**) proves that online TTA's failures are driven by objective misalignment and over-parameterization rather than solver limitations.
5. **Physical Weight-Space Validation:** Going beyond synthetic simulation to validate weight-space linear mode connectivity on CPU weights is highly commendable. Comparing scratch-trained (independent) and pre-trained (mode-connected) initializations validates the underlying axioms of linear mode connectivity, and the evaluation of temporal Parameter Exponential Moving Average (EMA) smoothing provides an exhaustive analysis of test-time mitigation.
6. **Robustness to Non-Smooth Trajectories:** The evaluation of Piecewise Linear Splines and Block-wise Parameter Sharing under non-smooth "zig-zag" landscapes proves that the continuous trajectory framework can be natively scaled to handle block-specific attention-MLP sensitivity shifts in Transformer architectures, completely neutralizing circularity concerns.
7. **Exceptional Clarity and Transparency:** The authors are highly transparent, explicitly detailing all simulator assumptions and limitations, and honestly discussing the "surrogate loss mismatch" and simulated-to-physical gaps in Section 3.9.

### Weaknesses
1. **Scale of Physical Weight-Space Validation:** While the simulator is calibrated against ViT-B/32, the physical weight-space validation is conducted on a relatively small-scale architecture: a **5-layer CNN on CPU using MNIST/FashionMNIST**. Model merging is primarily used on massive, billion-parameter models (such as LLMs or VLMs) where representation clashing, VRAM constraints, and deep layer interactions are far more complex. While the authors outline a highly actionable PEFT/TRL/Mergekit scaling roadmap in Section 5, immediate physical weight-space results on a larger architecture would have further strengthened the paper's empirical weight.
2. **Inference-Time Prediction Routing:** While OFS-Tune completely eliminates routing requirements for parameter adaptation, any multi-head merged model deployed on interleaved mixed streams still requires some routing mechanism at inference time to select the correct task-specific output head. Although the authors discuss practical, low-latency options (CLIP or simple logistic regression trained on the 10 samples), implementing this classifier adds a minor layer of engineering complexity shared by all multi-task architectures.

---

## Soundness
**Rating: Excellent**

The technical claims are supported by an exceptionally rigorous and sound methodology. The mathematical simulator is physically grounded and calibrated against empirical ViT classification statistics. The validation on actual deep neural network weights directly confirms linear mode connectivity axioms. The use of 30 independent random seeds, symmetrical solver budgets, and stratified sampling (which mathematically resolves a $99.96\%$ probability of validation class omission) represents an outstanding level of scientific rigor.

---

## Presentation
**Rating: Excellent**

The paper is exceptionally clearly written, well-structured, and easy to follow. The introduction does a beautiful job of contrasting online and offline data-access and deployment trade-offs. The mathematical notations are precise, and the tables are comprehensive. The visual quality is outstanding; Figure 2 clearly illustrates suite comparison with confidence intervals, and Figure 3 beautifully visualizes optimized coefficient trajectories across depth.

---

## Significance
**Rating: Excellent**

This work has highly substantial significance for both practitioners and researchers. For practitioners, it provides a safe-by-default, zero-test-time-compute calibration protocol (OFS-Tune) that is easily deployable on edge-devices. For researchers, it serves as a vital reality-check, calling for a transition toward multi-axial, multi-suite evaluation standards that analyze representational conflicts across diverse task combinations, while providing a concrete, highly actionable roadmap to scale offline trajectory tuning to frontier LLMs and VLMs.

---

## Originality
**Rating: Excellent**

The paper displays outstanding originality in its critical audit. Exposing "Task Suite Bias," modeling transductive stream-level selection bias, isolating trajectory regularization through the "OFS-Unconstrained" baseline, and deconstructing optimization asymmetry represent highly original contributions. Furthermore, the formulation and evaluation of piecewise linear splines and block-wise parameter sharing under non-smooth trajectories is a highly creative extension that makes the continuous trajectory framework fully compatible with Transformer architectures.

---

## Overall Recommendation
**Rating: 5: Accept**

This is an exceptionally strong, methodologically sound, and highly practical paper. It exposes major benchmark biases and fragile assumptions in the adaptive model-merging literature, while proposing a simple, robust, and zero-test-time-compute alternative (OFS-Tune) that is highly compatible with commercial edge-device deployment. The paper's claims are thoroughly backed by extensive multi-seed simulation, detailed solver deconstructions, and independent physical weight-space validation. While scaling the physical validation to larger architectures remains an important next step, the paper provides a clear, actionable scaling roadmap and sets a new benchmark for statistical and methodological rigor in model merging. It is a highly valuable contribution that should be accepted.
