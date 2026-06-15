# Revision Plan: Phase 4 Iterative Refinement (Third Round - Completed and Verified)

We have successfully addressed the critical feedback from the third round of Mock Review through a series of rigorous, highly focused textual edits, mathematical formulations, and scoping extensions:

## 1. Refolding "The Dual-Optimizer Paradox" to "The Overfitting-Optimizer Paradox" (Addressing Critical Flaw 1)
- **Critique:** Framing Adam GD's layer-specificity as a "physical reality" is overly generous. Since it achieves virtually no test accuracy improvement over unoptimized Task Arithmetic ($84.52 \pm 1.57\%$ vs $84.44 \pm 0.37\%$) while introducing massive seed instability (4x variance increase) and lower CIFAR-10 accuracy, this delicate layer-specific coordination is highly likely an aggressive form of transductive overfitting on the 256-image calibration set.
- **Action Taken:**
  - We refolded the "Dual-Optimizer Paradox" into the **"Overfitting-Optimizer Paradox"**.
  - We explicitly explained that both 1+1 ES and Adam GD suffer from transductive overfitting. Under 1+1 ES, it takes the form of high-frequency optimization noise easily smoothed out by Spatial Averaging (coarse regularizer). Under Adam GD, first-order autograd finds a highly precise, delicate configuration that overfits the transductive calibration statistics, making it extremely sensitive to perturbation (shuffling/averaging) without delivering generalizable test-set performance gains.
  - **Files Updated:** `submission/sections/00_abstract.tex`, `submission/sections/01_intro.tex`, `submission/sections/04_experiments.tex`, and `submission/sections/05_conclusion.tex`.

## 2. Formulating the CIFAR-10 Collapse vs. SVHN Rescue Trade-off (Addressing Critical Flaw 2)
- **Critique:** The Spatial Mean is not flatly "superior" or "sufficient" under zero-order search; it is a coarse regularizer that rescues the sacrificial SVHN task by breaking the joint entropy minimization task-bias, but does so at the cost of collapsing performance on the more complex CIFAR-10 task.
- **Action Taken:**
  - We revised the third bullet point of Section 4.2 to explicitly frame this as **"The SVHN Rescue vs. CIFAR-10 Collapse Trade-off and the Spatial Mean Illusion"**.
  - We detailed individual task dynamics, proving that the simple average accuracy is misleading and that layer-specific hierarchies are indeed critical to maintain representations on complex, non-linear domains like CIFAR-10.
  - **File Updated:** `submission/sections/04_experiments.tex`.

## 3. Resolving the CKA representational-accuracy discrepancy (Addressing Critical Flaw 3)
- **Critique:** The CKA activation differences are statistically tiny (well within measurement standard deviations) and directly decouple from downstream accuracy (where CIFAR-10 classification collapses by 10.35% despite maintaining >0.95 CKA).
- **Action Taken:**
  - We moderated all CKA claims across captions (Table 2, Figure 3) and text to emphasize that differences are statistically marginal.
  - We added a dedicated discussion explaining the decoupling of activation similarity (CKA) and downstream classification accuracy. We caution future researchers that high-level activation subspaces can remain highly aligned ($>0.95$ CKA) even when minor weight-space shifts corrupt fine-grained decision boundaries, leading to catastrophic classification failures.
  - **Files Updated:** `submission/sections/00_abstract.tex`, `submission/sections/01_intro.tex`, `submission/sections/04_experiments.tex`, and `submission/sections/05_conclusion.tex`.

## 4. Proposing Coefficient Regularization as a Solution (Required Revision 2)
- **Critique:** Suggest incorporating explicit coefficient regularization during test-time adaptation to stabilize optimization and reduce seed variance.
- **Action Taken:**
  - We mathematically formulated and proposed **Explicit Coefficient Regularization** (proximity penalty $||\Lambda - \lambda_{\text{init}}||^2_2$ scaled by hyperparameter $\beta$) as a solution to prevent transductive overfitting, reduce seed-to-seed variance, and preserve the representational hierarchies of complex tasks.
  - **Files Updated:** `submission/sections/04_experiments.tex` (added Section 4.5.4) and `submission/sections/05_conclusion.tex`.

## 5. Expanding Limitations and Scope (Required Revision 4)
- **Critique:** Clarify that findings are evaluated on saturated classification tasks and acknowledge that in larger-scale networks (like 7B+ modern LLMs) or highly complex downstream tasks, layer-by-layer optimization might remain critical.
- **Action Taken:**
  - We updated our Limitations paragraph in the Conclusion to explicitly clarify that our findings are demonstrated on saturated, low-resolution vision tasks. We acknowledge that in modern 7B+ parameter decoder-only language models or complex domains (e.g., instruction-tuning, cross-modal tasks), representational hierarchies are highly distinct, and localized layer-by-layer optimization may remain critical.
  - **File Updated:** `submission/sections/05_conclusion.tex`.

---

## LaTeX Source Files Updated, Compiled, and Verified:
1. `submission/sections/00_abstract.tex` - Overfitting-Optimizer Paradox framing, CKA decoupling.
2. `submission/sections/01_intro.tex` - Figure 1 caption, results bullet points, concluding paragraph.
3. `submission/sections/04_experiments.tex` - Result treatments discussion, CKA decoupling analysis, Section 4.5.1 on overfitting paradox, Section 4.5.4 proposing Explicit Coefficient Regularization.
4. `submission/sections/05_conclusion.tex` - Concluding paragraph, Limitations and Future Work.
5. All PDFs compiled and synchronized successfully via `tectonic`.
