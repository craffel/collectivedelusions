# Synthesized Peer Review

**Paper Title:** Is Q-Merge Actually Quantization-Robust? A Methodological Deconstruction and Robustness Audit of Quantization-Aware Model Merging  
**Overall Recommendation:** **5: Accept** (A highly rigorous, independent, and methodologically robust audit that delivers counter-intuitive findings with significant practical and scientific impact. Highly recommended for publication, contingent on minor presentation corrections).  
**Soundness:** Excellent  
**Presentation:** Good  
**Significance:** Excellent  
**Originality:** Excellent  

---

## 1. Summary of the Paper
The paper presents a rigorous, independent robustness audit and methodological deconstruction of **Q-Merge**, a prominent quantization-aware model merging framework. The authors systematically deconstruct three unstudied foundational assumptions in quantization-aware weight-space fusion:
1. **Quantization-Operator Monomorphism:** The assumption that merging coefficients optimized under a simulated operator can be deployed under mismatched, hardware-specific target schemas.
2. **Calibration Stream Purity:** The assumption that calibration data is always clean, static, and class-balanced.
3. **STE Gradient Path Fidelity:** The assumption that straight-through estimation (STE) is a reliable guide for navigating highly non-smooth, quantized loss landscapes.

Using a standardized Vision Transformer backbone (`timm ViT-Tiny`) across four diverse datasets (MNIST, FashionMNIST, CIFAR-10, and SVHN), the authors reveal critical vulnerabilities in Q-Merge:
- **Quantization-Operator Overfitting:** Continuous merging coefficients overfit intensely to the exact optimization operator (e.g., a per-channel to per-tensor shift collapses performance back to random guess levels of ~10% accuracy).
- **The Superiority of Full-Precision Search (Quantized AdaMerging):** Direct low-bit optimization via STE is consistently outperformed by **Quantized AdaMerging** (FP16 search followed by post-hoc target quantization), proving that STE-based search introduces harmful gradient noise.
- **Biased Gradient Regularization:** Derivative-free search (1+1 ES) fits rounding thresholds even better but collapses worse under schema shift, revealing that STE's biased gradients exert an implicit regularizing effect.
- **Calibration Vulnerability:** Unsupervised prediction entropy minimization collapses under realistic, class-skewed calibration streams, though input noise acts as an accidental regularizer by smoothing discrete rounding thresholds.

The authors conclude by proposing four concrete methodological mandates to improve evaluation standards in weight-space consolidation.

---

## 2. Strengths of the Paper
1. **Exceptional Methodological Rigor:** Dismantling Q-Merge along four distinct axes (calibration size, cross-schema generalization, spatial smoothing/optimizers, and data corruptions/skew) represents an exceptionally comprehensive evaluation that sets a high bar for rigor in weight-space fusion research.
2. **Counter-Intuitive Discovery of "Quantized AdaMerging":** Showing that full-precision coefficient search followed by post-hoc quantization consistently outperforms direct low-bit optimization under quantization constraints ($30.00\%$ vs $26.25\%$ average accuracy) is a high-signal, high-impact finding that challenges the standard "quantization-aware search" trend and could save researchers significant GPU search overhead.
3. **First-of-its-Kind Cross-Schema Evaluation:** Uncovering **"Quantization-Operator Overfitting"** is a major practical contribution. This highlights a severe deployment warning for practitioners compiling simulated PyTorch models onto hardware ASICs with heterogeneous scale/zero-point representations.
4. **Elegant Theoretical Connections:** The authors provide strong mathematical grounding for their findings, particularly the randomized smoothing interpretation of input-space noise injection (Section 4.5) and the 1/5-th success rule of the 1+1 Evolution Strategy (Section 3.4).
5. **Constructive and Actionable Mandates:** The paper does not simply criticize Q-Merge; it provides four highly constructive recommendations (e.g., using TIES-Merging/DARE to smooth the unquantized landscape first) to guide future research.

---

## 3. Weaknesses of the Paper
1. **Model and Task Scale Limitations:** The audit is conducted entirely on a lightweight model (`ViT-Tiny`, 5.7M parameters) and standard, low-resolution classification datasets (MNIST, FashionMNIST, CIFAR-10, SVHN). While this is justified by the authors as a means of ensuring complete scientific control, modern model-merging is primarily deployed on **billion-parameter Large Language Models (LLMs)** and large Vision-Language Models (VLMs). Verifying these findings on a larger architecture (such as CLIP or Llama-3 merging) would significantly strengthen the paper's claims, although the authors provide a thoughtful discussion on LLM/VLM generalization in Section 5.
2. **High-Interference Regime Only:** The paper operates under an extreme weight conflict/task interference scenario (where the unquantized FP16 Task Arithmetic baseline scores only 35.12%, compared to $>80\%$ for individual experts). It remains unclear if the Cross-Schema Generalization Gap and the failure of STE optimization are similarly severe in a **low-interference merging regime** (e.g., merging models that share close, highly related task boundaries), where weight-space alignments are smoother.
3. **No Active Therapeutic Implementation:** While the authors propose four mandates and discuss starting from TIES-Merging/DARE to smooth the landscape, they do not implement or evaluate a concrete "robust" optimization pipeline to solve the Cross-Schema Generalization Gap. This makes the paper primarily diagnostic rather than offering an active, optimized solution.

---

## 4. Minor Suggestions and Presentation Corrections

The paper is exceptionally solid and ready for publication. However, addressing the following minor presentation and bookkeeping issues will ensure that the final camera-ready version is technically flawless:

### [Presentation 1] LaTeX Running Header Compilation Issue (Major Defect)
On pages 2 through 12, the running header at the top of each page compiles to:
$$\text{"Title Suppressed Due to Excessive Size"}$$
- **Diagnosis:** In the preamble, the authors set a very reasonable short running title using:
  `\icmltitlerunning{Is Q-Merge Actually Quantization-Robust?}` (Line 60 in `example_paper.tex`).
  However, the ICML style file (`icml2026.sty`) measures the running title's height by wrapping it in a vertical box (`\vbox`):
  `\global\setbox\titrun=\vbox{\small\bf\@icmltitlerunning}`
  and then checks if the height of this vertical box exceeds 6.25pt:
  `\ifdim\ht\titrun>6.25pt \gdef\@runningtitleerror{2}`
  Because any single line of text wrapped in `\small\bf` has an intrinsic line-height of 7–8pt, the height `\ht\titrun` will **always** exceed 6.25pt, even for extremely short, single-line running titles. This triggers an error flag and automatically overrides the running title with "Title Suppressed Due to Excessive Size".
- **Actionable Correction:** Since this is a bug in the template's line-height checking logic (measuring `\vbox` height instead of using `\hbox` or checking for line breaks), the authors should either patch the measuring command in their document class or report it to the venue's technical chair, as the template's line-height rule is fundamentally broken.

### [Bookkeeping 2] Workspace Housekeeping and Slurm Log Clutter
There are numerous transient Slurm log files (e.g., `check_cuda_*.err`, `evaluate_merging_*.out`, `train_experts_*.err`) and wrapped slurm scripts (`*.slurm.wrapped.slurm`) cluttering the root directory of the repository.
- **Actionable Correction:** The authors should organize their repository by moving these transient job files, execution logs, and auto-generated slurm wrappers to a dedicated directory (e.g., `logs/` or `slurm_job_scripts/`) or adding them to `.gitignore` to keep the codebase clean and professional.

### [Notation 3] Loose Notation for the Optimization Step (Equation 12)
Equation 12 represents the coefficient gradient update as:
$$\Lambda^{(t+1)} = \Lambda^{(t)} - \eta \cdot \text{Adam}\left( \nabla_{\Lambda} \mathcal{L}_{\text{entropy}}(\Lambda^{(t)}) \right)$$
- **Actionable Correction:** The Adam optimizer is treated as a simple static function of the current gradient. However, Adam maintains dynamic internal states representing the first and second moments of the gradients ($m_t$ and $v_t$). To be mathematically rigorous, the authors should either write Adam's complete state-update equations or clarify that $\text{Adam}(\cdot)$ represents the bias-corrected step vector.

### [Presentation 4] Heatmap Figure Labels
In Figure 2 and Table 2, the labels on the x-axis and y-axis in the matrix and heatmap contain raw underscores (e.g., `sym_tensor`, `sym_channel`). Replacing these with human-readable text (e.g., "Symmetric Per-Tensor" and "Symmetric Per-Channel") would improve the professional aesthetic.
