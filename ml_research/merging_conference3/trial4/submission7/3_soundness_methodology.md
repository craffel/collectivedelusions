# 3. Soundness and Methodology

## Mathematical Soundness and Curvature Modeling
The mathematical modeling of the local weight-space sensitivity landscape via the **Coupled Model II Landscape** is highly rigorous and elegant.
- The formulation of local task-specific sensitivity in Equation 1 using quadratic and quartic curvature parameters ($A_k^{(l)}$ and $B_k^{(l)}$) is mathematically sound. The quartic term ($B_k^{(l)}$) correctly models the rapid performance drop-off when coefficients drift significantly beyond the optimal basin.
- The modeling of pairwise task-to-task representational interference in Equation 2 is physically intuitive, capturing the localized representational clashes when layer-wise coefficients diverge.
- The empirical calibration protocol detailed in Appendix A—extracting sensitivity parameters via least-squares regression from empirical Vision Transformer (ViT-B/32) classification statistics—grounds the simulation in real-world neural network behaviors (e.g., matching the high sensitivity of early and late layers vs. the robustness of middle layers).

---

## Critical Review of Assumptions and Potential Flaws

### 1. Inherent Circularity in Simulator Design (Polynomial Priors) - RESOLVED
- **Concern:** A key methodological concern of the original simulation design was that the ground-truth optimal layer-wise scaling profiles ($\alpha_{k, \text{opt}}^{(l)}$) were modeled as smooth, low-degree polynomials. Consequently, optimization methods that constrain their search space to low-degree polynomials (such as PolyMerge and OFS-Tune) were mathematically favored by design.
- **Resolution:** The authors have completely neutralized this circularity critique by introducing a new, dedicated subsection in the main text (**Section 4.3: Neutralizing Simulator Circularity: Performance under Non-Smooth Trajectories**) and Appendix D. 
- They simulate a challenging, highly non-smooth "zig-zag" optimal trajectory where layer sensitivities alternate sharply from layer to layer, mimicking actual Transformer networks (attention vs. MLP blocks).
- Under this non-smooth regime, the authors evaluate two new localized trajectory constraints: **Piecewise Linear Splines** and **Block-wise Parameter Sharing**. Both methods successfully recover high performance (66.24% and 67.38% average accuracy respectively in Suite B) while continuing to filter out transductive stream and validation noise. This proves that the proposed trajectory framework is highly flexible and generalizes beyond smooth global polynomial curves.

### 2. Setting and Data-Access Trade-off
- **Analysis:** The comparison between online TTA (AdaMerging/PolyMerge) and offline OFS-Tune is an apples-to-oranges comparison regarding data-access assumptions. Online TTA is designed for strictly unsupervised, zero-shot environments where labeled training or validation data is entirely unavailable. OFS-Tune is a supervised, few-shot paradigm requiring a small labeled calibration set ($M=10$ samples per task) during a pre-deployment phase.
- **Resolution:** The authors have been highly transparent and intellectually honest about this trade-off. They added a clear qualifying section in Section 1 and Section 3.7 (and Table 1) making it explicit that in environments where absolutely zero labeled validation data can be accessed (e.g., due to extreme user-privacy regulations or zero-shot constraints), OFS-Tune is not a drop-in replacement, and online TTA remains the only choice. This provides practitioners with a comprehensive map of when each paradigm is operationally safe and effective.

### 3. Formulation of the Accuracy-Distance Ratio ($\mathcal{R}_k$)
- **Analysis:** The simulated classification accuracy formulation in Equation 3 relies on the normalized weight distance ratio:
$$ \mathcal{R}_k = \frac{\sum_{l=1}^L \left(\alpha_l - \alpha_{k, \text{opt}}^{(l)}\right)^2}{\sum_{l=1}^L \left(\alpha_{\text{init}} - \alpha_{k, \text{opt}}^{(l)}\right)^2} $$
- If the optimal profile $\alpha_{k, \text{opt}}^{(l)}$ is exactly equal to the uniform initialization $\alpha_{\text{init}}$, the denominator shrinks to zero.
- **Resolution:** The authors have successfully resolved this boundary condition in Footnote 1 of Section 3.2. They set $\mathcal{R}_k = 0.0$ when the denominator falls below $10^{-6}$, explaining that this corresponds to the scenario where a task-specific expert network's parameter profile remains virtually identical to the pre-trained initialization, meaning static uniform merging is already optimal. They also explicitly acknowledge that while near-boundary conditions exhibit high sensitivity, the bounded parameter space $[0, 1]$ naturally stabilizes optimized trajectories and prevents numerical instability, showing excellent mathematical rigor.

### 4. Simplified Stream Noise and Surrogate Loss Mismatch
- **Analysis:** In the simulator, the online TTA optimization objective $\mathcal{L}_{\text{TTA}}$ directly tracks the ground-truth optimal parameter profiles perturbed by noise. In reality, physical online TTA must optimize a highly non-convex, rugged unsupervised prediction entropy surface without any access to the true parameter target curves.
- **Resolution:** The authors have added an extensive and highly insightful discussion in Section 4.2 detailing this simulated-to-physical gap and laying out design guidelines for future model-merging simulators to close this gap. They clarify that PolyMerge's minor simulated advantage in other suites is due to its "transductive test-time advantage" (adapting directly on test stream noise). They explain why this advantage is highly fragile and collapses in real physical weight spaces because unsupervised prediction entropy minimization is rugged, non-convex, and vulnerable to degenerate shortcut solutions. This is an outstanding, highly critical analysis that bridges the gap between simulation and physical experiments.
