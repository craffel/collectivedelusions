# Experiment Check

This file evaluates the experimental design, completeness of baselines, fairness of comparison, and validity of empirical claims in the paper.

---

## 1. Experimental Design and Fairness
- **Strengths of Design**:
  - The evaluation is divided into two distinct parts: **Part A: Frozen Classification Heads** (strictly frozen classifier probes to measure pure representation alignment) and **Part B: Active Classification Head Adaptation** (co-adapting linear classifiers to measure joint boundary alignment).
  - This is an exceptionally fair and rigorous experimental design. Many papers in the adaptive model merging literature obscure their results by reporting numbers under joint adaptation without decoupling whether the gains are coming from weight alignment or classifier probe tuning. This division is a major scientific strength.
  - The inclusion of **Unconstrained Scaling** as an ablation is highly constructive. It isolates the impact of the convex barycentric simplex constraint, proving that the constraint functions as a stable safeguard at a slight performance cost.

---

## 2. Weaknesses and Areas of Improvement in the Experiments

### A. Omission of Individual Task Results for Core Baselines (Table 1, Part A)
- **Problem**: In Table 1, Part A (Frozen Classification Heads), the individual task-wise accuracies (for SUN397, Cars, RESISC45, EuroSAT, SVHN, GTSRB, MNIST, DTD) are completely omitted (marked as `--`) for the two fundamental static baselines: **Task Arithmetic** and **TIES-Merging**. Only their average accuracies are reported (69.10% and 72.90% respectively).
- **Flaw**: Task Arithmetic and TIES-Merging are the standard non-adaptive baselines in this field. Their task-wise accuracies on CLIP ViT-B/32 are well-known and widely documented. Omitting them in Part A (while including them for all other models in Part A, and for all models in Part B) represents a significant gap in the completeness of the results table. The author must report these task-wise results to provide a complete, side-by-side comparison.

### B. Lack of Statistical Significance and Error Bars (Critical Flaw)
- **Problem**: Test-time adaptation is performed on small local calibration splits (such as the 20% sample split in Table 3, or the extreme 5 samples per class split in Table 4). 
- **Flaw**: Point estimates are reported for all results without any error bars, standard deviations, or statistical significance indicators (e.g., across 3 or 5 random seeds).
- **Impact**: 
  - For the 5-sample extreme low-data regime, results are highly sensitive to the specific samples selected in the calibration batch. 
  - Furthermore, several performance margins are extremely slim. For instance, **BPAM-Static** (69.21%) outperforms **Task Arithmetic** (69.10%) by only **+0.11%** absolute. In Part B, **BPAM-Full** (75.22%) outperforms **Task Arithmetic + Head Tuning** (74.80%) by only **+0.42%** absolute.
  - Without running multiple seeds and reporting standard deviations (e.g., $75.22 \pm 0.35\%$), it is impossible to determine whether these marginal improvements are statistically significant or merely random noise. Providing error bars is a strict requirement for a high-quality ICML submission.

### C. Evaluation of Hyperparameter Sensitivity under Default Settings
- **Problem**: The sensitivity study of the proximity penalty hyperparameter $\beta$ (Table 4) is conducted exclusively under the extreme 5-sample-per-class calibration regime.
- **Flaw**: While this demonstrates the utility of the penalty under extreme data scarcity, the paper does not show a corresponding sensitivity curve for $\beta$ under the **standard data regime** (unlimited or standard 20% calibration splits). 
- **Recommendation**: To be thoroughly complete, the paper should show the sensitivity of $\beta \in [0, 1.0]$ under both the default/standard data regime and the extreme low-data regime in parallel. This would illustrate the exact transition boundary of where the proximity penalty shifts from being "empirically redundant" to "empirically vital".

---

## 3. Overall Experiment Rating: Good
The experimental design is methodologically sound, transparent, and highly disciplined in separating frozen-head and active-head results. However, the completeness of the primary table (missing task-wise baseline accuracies) and the scientific rigor of reporting (absence of error bars and statistical significance testing over multiple seeds) fall short of the standard expected for an accepted publication at a tier-1 conference.
