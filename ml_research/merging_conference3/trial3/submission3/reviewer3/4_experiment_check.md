# 4. Experiment Check

## Critical Evaluation of the Experimental Setup
An evaluation of the experimental design reveals a major disconnect between the simulated results and the physical validation results, raising critical empirical concerns:
1. **Skeptical Reliance on Numerical Simulation:**
   The primary findings of the paper (Tables 1 & 2, Figures 2, 3, 4, 5, 7) are evaluated inside a "highly calibrated numerical simulation" rather than actual Vision Transformer weights. While continuous mathematical simulations can be helpful for initial prototyping, they often fail to capture the high-dimensional saddle points, variable local curvature, and non-linear layer interactions (e.g., ReLU, LayerNorm transitions) of real neural networks. This makes the primary evaluation less convincing than one executed on actual Vision Transformer models.
2. **Artificial Noisy Evaluation in Simulation:**
   In Table 2 (Model II Robustness sweep), the joint average accuracy of the Task Arithmetic baseline remains exactly **$84.44\% \pm 0.00$** across all noise scales ($\gamma \in \{1.0, 1.5, 2.0, 2.5, 3.0\}$). In real-world physical settings, test-time input noise degrades the representations and lowers the classification accuracy of *all* models, including Task Arithmetic (as demonstrated in the physical experiments where Task Arithmetic's accuracy drops from $70.75\%$ to $37.17\%$). 
   The fact that Task Arithmetic remains perfectly unaffected by noise in Table 2 strongly implies that the simulation only corrupts the *unlabeled adaptation batch* (unsupervised TTA stream) to disrupt the optimizer, but evaluates the final classification accuracy on *clean* data. This is an artificial evaluation setting. In actual edge deployments, persistent sensor noise or weather artifacts affect both the TTA phase and the final inference phase. Evaluating the adapted coefficients on clean data rather than noisy data is a significant shortcut that fails to represent real-world robustness.

## Evaluation of Baselines and the "Task Arithmetic" Paradox
In the physical experiments (Tables 3 & 4), which run on actual MLP and CNN weights, a glaring paradox emerges: **the proposed FlatMerge method is consistently outperformed by the simple, static, zero-overhead baseline of Task Arithmetic** across almost all scenarios:
- **MLP Merging (Table 3):**
  - Under Clean data ($\gamma = 0.0$), Task Arithmetic achieves **$70.75\%$** joint average accuracy, while ZO-FlatMerge only reaches **$54.71\%$** (a huge **$16.04\%$ absolute deficit**).
  - Under Moderate noise ($\gamma = 1.0$), Task Arithmetic achieves **$63.74\%$**, while ZO-FlatMerge achieves **$49.11\%$** (a **$14.63\%$ absolute deficit**).
  - Under Heavy noise ($\gamma = 2.0$), Task Arithmetic achieves **$48.91\%$**, while ZO-FlatMerge achieves **$48.88\%$**.
  - Under Extreme noise ($\gamma = 3.0$), ZO-FlatMerge finally beats Task Arithmetic by a moderate margin (**$41.35\%$** vs. **$37.17\%$**).
- **5-Layer CNN Merging (Table 4):**
  - Under Clean data ($\gamma = 0.0$), Task Arithmetic achieves **$58.20\%$** joint average accuracy, while ZO-FlatMerge only reaches **$48.57\%$** (a **$9.63\%$ absolute deficit**).
  - Under Moderate noise ($\gamma = 1.0$), Task Arithmetic achieves **$40.67\%$**, while ZO-FlatMerge achieves **$29.20\%$** (an **$11.47\%$ absolute deficit**).
  - Under Heavy noise ($\gamma = 2.0$), Task Arithmetic achieves **$24.60\%$**, while ZO-FlatMerge achieves **$19.77\%$** (a **$4.83\%$ absolute deficit**).
  - Under Extreme noise ($\gamma = 3.0$), Task Arithmetic achieves **$17.77\%$**, while ZO-FlatMerge achieves **$16.07\%$** (a **$1.70\%$ absolute deficit**).

This is a highly concerning empirical finding. Task Arithmetic requires **zero adaptation steps, zero dynamic computation, zero activation memory, and zero weight-reconstruction overhead**, yet it beats the complex, dual-regularized ZO-FlatMerge by up to $15\%$ in realistic clean and moderate-noise settings on physical models. While FlatMerge successfully prevents the catastrophic constant-prediction collapse of standard AdaMerging and PolyMerge (which collapse CNN joint accuracy to a near-random $\approx 14\% - 17\%$), its inability to beat simple static blending under moderate-to-high noise levels severely undermines its practical utility for real neural networks.

## Missing Baselines and Simulated vs. Real Gaps
- **Missing Baselines in Physical Validation:** 
  The physical validations are missing key comparative baselines. Why is RegCalMerge missing from Tables 3 & 4? Why is PolyMerge missing from Table 3? For a thorough empirical evaluation, these methods should be included in the physical runs to establish a complete comparison on actual weights.
- **Simulated vs. Real Gaps:**
  In Table 1 (Model II Simulation), PolyMerge ($d=2$) achieves a robust joint average of **$85.54\% \pm 1.35\%$**, which is better than Task Arithmetic's $84.44\%$. However, in Table 4 (Real CNN), PolyMerge ($d=2$) catastrophically collapses to **$14.27\%$** under clean conditions. Why is there such an extreme gap between simulated behavior (where PolyMerge is excellent) and real behavior (where PolyMerge completely collapses to random guessing)? This discrepancy further highlights the lack of fidelity in the simulated loss landscapes and underscores the danger of relying on simulation-based main results.

## Statistical Rigor in Physical Validation
While the simulated experiments are repeated across 15 independent random seeds with detailed standard deviations reported, the physical validations in Tables 3 & 4 appear to be single-run, deterministic results. There are no standard deviations, confidence intervals, or multi-seed trials reported for the physical MLP and CNN models. For an empirically rigorous paper, the authors must report standard deviations across multiple runs/seeds for the physical models to prove that the observed physical differences are statistically significant.
