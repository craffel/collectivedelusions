# Intermediate Review: 5. Impact and Presentation

## Major Strengths
1. **The Elegance and Practicality of PolyMerge:**
   The parameterization of layer-wise blending coefficients using a low-degree polynomial of network depth is a stellar, high-quality contribution. It is extremely simple to understand, trivial to implement, adds zero hyperparameter tuning, and dramatically reduces optimization parameters from 56 to 12. Crucially, it provides a massive and highly practical performance boost (+8% to +9% over unconstrained methods) across all formats. This represents the kind of elegant and robust engineering that the machine learning community should champion.
2. **Exhaustive and Comprehensive Evaluation:**
   The paper evaluates performance across an exceptionally thorough set of conditions: six distinct quantization formats (from FP32 to aggressive INT4 channel-wise), four diverse image classification datasets (MNIST, FashionMNIST, CIFAR-10, SVHN), and compares against six distinct, relevant baselines (Uniform TA, AdaMerging, RegCalMerge, Q-Merge, PolyMerge, HessMerge).
3. **Rigorous and Transparent Limitations:**
   The authors display remarkable scientific integrity by explicitly pointing out and discussing two critical, fundamental limitations in Section 4.4: the "Expert-to-Merge Drop" (the -31.27% performance gap between individual experts and the best merged model due to disjoint domain shifts) and the "Unusable INT4 absolute performance" (noting that the 19.07% accuracy, while state-of-the-art, is practically broken and unusable). This honest and self-critical analysis is a breath of fresh air and increases the paper's scientific credibility.
4. **Writing and Clarity:**
   The paper is exceptionally well-written, clearly structured, and mathematically sound. The equations are clean, and the explanation of the second-order Taylor expansion and task-vector scale pathology is highly accessible.

---

## Areas for Improvement
1. **Unjustified Complexity of CR-SACM:**
   The proposed CR-SACM framework adds major algorithmic complexity (measuring layer-wise norms, clipping norms with $\beta$, scaling perturbations, clamping, and extra forward-backward passes). However, this complexity is completely unjustified: in all practical settings (FP32 and INT8), it actively **degrades** the model's accuracy compared to standard PolyMerge. It only provides a marginal improvement in the unusable INT4 format. The authors should critically evaluate whether the high complexity of CR-SACM is actually worth its negative-to-negligible returns.
2. **Missing Simpler Baseline (Direct Parameter Perturbation):**
   The authors choose to apply the sharpness-aware perturbation in the 56-dimensional intermediate coefficient space, which artificially creates the "norm scale pathology." A much simpler and more elegant design would apply standard SAM directly in the 12-parameter polynomial coefficient space $\mathbf{p}$. The paper completely lacks this critical and simpler baseline, which would likely bypass the scale pathology altogether without requiring any of the complex norm-clipping equations in CR-SACM.
3. **Practical Viability of Post-Hoc Multi-Domain Merging:**
   The massive performance drop from individual experts (88.67%) to the best merged model (57.40%) raises a fundamental question about the practical viability of merging models fine-tuned on completely disjoint, disparate domains. While the authors address this honestly, they should discuss alternative routing or mixture-of-experts (MoE) baselines to contextualize why weight-space merging should be preferred over simple multi-model ensembles or routing when the performance drop is so severe.

---

## Overall Presentation Quality
The presentation quality is **excellent**. The writing is extremely clean, and the layout of the figures and tables is professional and highly readable. The transition from the mathematical analysis of the second-order curvature to the empirical results is logical and easy to follow.

---

## Potential Impact and Significance
- **Significance of the Polynomial Subspace Constraint (PolyMerge):** **High**. This represents a highly effective, elegant, and generalizable regularization paradigm for test-time adaptive model merging. It is likely to influence future work and be adopted by practitioners who want a fast, simple, and highly effective way to stabilize test-time model composition.
- **Significance of the Proposed CR-SACM/CR-PolySACM Method:** **Low to Moderate**. Due to its high complexity and negative return on investment in all practical precision formats, practitioners are highly unlikely to adopt CR-PolySACM. Its primary impact is theoretical: it provides an interesting scientific analysis of task-vector norm scale discrepancies in weight-space flatness optimization under extreme (and practically unusable) discretization noise.
