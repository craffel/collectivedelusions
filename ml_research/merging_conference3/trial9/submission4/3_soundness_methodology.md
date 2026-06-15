# Soundness and Methodology Review

This file details several critical methodological vulnerabilities, hidden accuracy trade-offs, internal inconsistencies, and parameter sensitivities in the paper's formulation and evaluation of **Momentum-Merge**.

---

## 1. Internal Inconsistencies & Text-Table Discrepancies (Major Flaw)
There is an internal inconsistency between the claims made in the main text of the paper (Abstract, Introduction, Section 4.3, and Conclusion) and the actual empirical values reported in Table 1, as well as the uncalibrated sweeps in the appendices:
* **The Baseline Text Discrepancy:** In `submission/sections/04_experiments.tex` L23, the baseline text claims that **Momentum-Merge (Base)** matches ChemMerge's joint accuracy within 0.05% (**76.15% vs. 76.20%**). However, in Table 1, Momentum-Merge (Base) is evaluated at **74.85%** and ChemMerge is at **74.71%**. The 76.15% and 76.20% values originate from an uncalibrated parameter sweep across a different seed setup, creating significant numerical friction.
* **Appendix Discrepancies:** Appendix C (Table 3) and Appendix D (Section text) continue to reference old or uncalibrated results like **76.15%** (at $\beta = 0.60, \tau = 0.100$) and **76.10%** (at $\beta = 0.60, \tau = 0.005$) as peak performance. This conflicts with the synchronized results in Table 1 of the main text, where Momentum-Merge (Base) achieves **74.85%**.
* **Impact:** While the authors successfully updated most of the main text to reflect the synchronized 10-seed results, they missed several verbal references and baseline descriptions in the text and appendices. This minor scientific hygiene issue undermines the manuscript's empirical coherence.

---

## 2. The Hidden Cost of Stateful Smoothing (Accuracy-Stability Trade-off)
The paper presents stateful momentum smoothing as a pure benefit that improves stability. However, a close examination of Table 1 exposes a major hidden trade-off that is not sufficiently emphasized:
* **SABLE + Layer Centroids (Stateless Calibrated):** achieves **77.24%** classification accuracy.
* **Momentum-Merge (Advanced, Statefully Smoothed Calibrated):** achieves only **74.98%** classification accuracy.
* **The Degradation:** Adding stateful momentum smoothing actually **DEGRADES classification accuracy by 2.26% absolute** (77.24% vs. 74.98%)! 

### Technical Explanation
Stateless similarity routing is highly plastic; it adapts ensembling weights rapidly at each layer to match local activation representations. While representation noise introduces high routing jitter (0.0285), this routing plasticity is crucial for classification accuracy. Momentum smoothing acts as a heavy low-pass filter (especially under a very sharp temperature like $\tau = 0.005$ and $\beta = 0.60$). While it successfully dampens layer-to-layer routing jitter to near-zero (0.000374), it introduces **extreme routing over-smoothing**. The routing weights become overly sluggish and cannot adapt fast enough to representational shifts across depth, dragging accuracy down by 2.26% absolute. The accuracy boost of the "Advanced" variant is entirely driven by the Layer Centroid calibration, and adding momentum actually *harms* performance. This represents a fundamental **Accuracy-Stability trade-off** that the authors should be more explicit about.

---

## 3. Ignored Baselines in Text Discussions
While the authors did include the control baselines **SABLE + Layer Centroids** (77.24%) and **ChemMerge + Layer Centroids** (76.60%) in Table 1, their discussions in the Abstract and Intro somewhat glide over them:
* Both SABLE + Layer Centroids (77.24%) and ChemMerge + Layer Centroids (76.60%) significantly outperform **Momentum-Merge (Advanced)** (74.98%) in accuracy.
* While Momentum-Merge (Advanced) has outstanding routing stability (routing jitter is 38$\times$ lower than ChemMerge and 76$\times$ lower than SABLE), it does so by trading off a substantial amount of accuracy. The authors should more transparently frame Momentum-Merge as a stability-optimized choice, rather than claiming it "outperforms SOTA across both stability and parsimony" without highlighting the accuracy cost relative to calibrated baselines.

---

## 4. Severe Vulnerability to Calibration Data Scarcity (Recurrence Trapping)
Our sensitivity sweep over the offline calibration dataset size $|\mathcal{C}_k|$ (samples per task) exposes a major architectural vulnerability in Momentum-Merge (Advanced):

| Calibration Size $|\mathcal{C}_k|$ | SABLE + Layer Centroids Acc (%) | MM (Advanced) Acc (%) | Performance Gap |
| :---: | :---: | :---: | :---: |
| **8** | **76.00%** | **71.20%** | **-4.80% (Collapse)** |
| **16** | **77.25%** | **73.15%** | **-4.10%** |
| **32** | **76.65%** | **74.00%** | **-2.65%** |
| **64** | **78.00%** | **76.25%** | **-1.75%** |
| **128** | **76.65%** | **75.95%** | **-0.70%** |

### Technical Analysis of the Collapse
Under data-scarce calibration (e.g., $|\mathcal{C}_k| = 8$), the layer centroids are highly noisy because they are estimated from extremely few samples. Stateless `SABLE + Layer Centroids` is highly robust, only dropping 2.0% in accuracy (from 78.00% to 76.00%) because its independent routing decisions allow it to recover from noisy centroids at any given layer. 

In contrast, `Momentum-Merge (Advanced)` collapses catastrophically by **5.05%** absolute (down to **71.20%**). This occurs because of **Recurrence Trapping**. Momentum-Merge (Advanced) initializes its boundary condition using the raw similarity weight of the first adapted layer (Eq. 10). When centroids are noisy, this initial boundary weight is highly inaccurate. Because Momentum-Merge relies on momentum memory, this initial routing error is propagated across network depth, trapping the ensembling coefficients in highly sub-optimal states throughout the entire forward pass. This is a severe architectural vulnerability that makes Momentum-Merge highly unsuited for data-scarce settings.

---

## 5. Hyperparameter Sensitivity and Interaction
We analyze the joint interaction of the momentum coefficient $\beta$ and the Softmax temperature $\tau$:
* Stateless SABLE ($\beta = 0.0$) is highly robust to temperature changes, maintaining stable accuracy between **75.30%** and **75.85%** across temperatures from $0.005$ to $0.200$.
* In contrast, when Momentum-Merge is run with high inertia ($\beta = 0.8$), its accuracy collapses to **71.70%** under a larger temperature ($\tau = 0.300$).

This indicates that high-inertia stateful smoothing introduces high sensitivity to the temperature parameter $\tau$, requiring delicate joint tuning and reducing the model's robustness to hyperparameter choices.
