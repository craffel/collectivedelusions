# 5. Impact & Presentation

## Presentation, Clarity, and Structure
The writing and structure of this paper are **outstanding**. The overall narrative is highly cohesive, clear, and easy to follow, making it accessible to both researchers and practitioners in model merging.

### Presentation Strengths:
* **Exemplary Narrative Flow:** The paper reads like a masterclass in applying Occam's razor. It identifies an unnecessary trend of complexity, introduces a simple closed-form alternative (PFSR), analyzes an elegant advanced extension (OTSP), and uses rigorous mathematical and empirical analysis to deconstruct both, showing that the simpler method is actually the superior one.
* **Rigorous Mathematical Notation:** The mathematical formulations in Section 3 are clean, unambiguous, and systematically laid out. Steps 1 to 5 are extremely easy to follow, with each variable clearly defined.
* **Self-Critical and Honest Tone:** The author is exceptionally transparent about the limitations of the work, including the "Orthogonal Masking Effect," the "Hard Gating Penalty" under active task overlap, and the "Evaluation Gap" of their simulation sandbox. This intellectual honesty significantly strengthens the paper's scientific credibility.
* **Proper Contextualization:** The related work section is well-structured, successfully placing the paper in the context of static merging (Task Arithmetic, TIES-Merging), dynamic routing (MoE, QWS-Merge), and Löwdin orthogonalization.

## Significance & Potential Impact
* **Theoretical Significance:** High. The paper provides elegant, closed-form proofs of ensembling dynamics under isotropic representation noise. The SNR Equivalence proof and the derivation of the Noise Amplification Penalty are major theoretical contributions that will influence future research on representation-space projections and orthogonalization limits.
* **Practical Significance:** High. The post-hoc ensembling paradigm has massive systems-level benefits—loading and executing only selected specialists at runtime rather than executing all experts simultaneously. The addition of **Top-$k$ Sparse Gating** (Section 4.5) successfully bridges the gap between ensembling accuracy (which requires soft gating) and VRAM savings (which requires hard sparse gating), showing how we can load at most $k=2$ experts while retaining cooperative performance.

## Constructive Suggestions for Presentation & Impact Improvement

To further elevate the presentation and maximize the impact of the paper, the author should address the following actionable points:

### 1. Detailing Top-$k$ Interaction with Self-Calibrated Temperature Scheduling
In Section 4.5, the author introduces Top-$k$ sparse gating to preserve ensembling benefits while maintaining sparse execution, and in Section 3.5, they propose self-calibrated temperature scheduling ($\tau_b = \gamma \cdot \text{std}_k(u_{k, b})$).
* *Suggestion:* The author should provide a brief discussion on how these two features interact. **Does Top-$k$ sparse gating require a different scaling multiplier $\gamma$ compared to standard Softmax, or does the self-calibrated temperature automatically adjust to the top-$k$ coordinate distribution?** Clarifying this integration would help practitioners deploy the complete parameter-free routing pipeline.

### 2. Expanding on the Class Prototype Imbalance Problem
In Section 5 (Future Outlook / Discussion), the author notes that expert registries with heterogeneous class cardinalities ($C_k$) may face coordinate imbalances: the entropy of task projection scores scales logarithmically with class counts: $O(\log C_k)$.
* *Suggestion:* The author should provide a more concrete guideline or formula on how to normalize or balance the raw SVD centroid projection coordinates across experts with unbalanced vocabulary sizes (e.g., one expert with 2 classes and another with 1000 classes) to prevent the router from systematically biasing toward experts with larger vocabulary scales.

### 3. Improving Figure 1 Labels
In Figure 1, the legend and labels should be checked to ensure they are self-contained and easy to read.
* *Suggestion:* Ensure that the x-axis (e.g., Homogeneous $B=256$, Heterogeneous $B=256$, Heterogeneous $B=1$) is clearly labeled and that the captions explain the standard deviations and error bars comprehensively.

## Ratings
* **Presentation Rating: Excellent**
* **Significance Rating: Excellent (with the real-world ResNet-18 POC and Top-$k$ gating)**
