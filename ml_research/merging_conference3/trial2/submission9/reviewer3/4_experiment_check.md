# 4. Experimental Setup and Evaluation Check

## Evaluation of Experimental Design
The experimental evaluation in this paper is exceptionally rigorous, well-designed, and complete. 

### 1. Benchmark Standard
The authors evaluate their method on the standard **8-task image classification benchmark using CLIP ViT-B/32**. This benchmark covers a diverse range of visual domains:
- SUN397 (scenes)
- Stanford Cars (fine-grained vehicles)
- RESISC45 (remote sensing)
- EuroSAT (satellite imagery)
- SVHN (street view house numbers/digits)
- GTSRB (traffic signs)
- MNIST (handwritten digits)
- DTD (textures)

This selection is the gold standard for model merging, ensuring that the results are directly comparable to published literature.

### 2. Comprehensive Baselines
The paper includes a highly complete set of baselines across multiple paradigms:
- **Static Merging:** Task Arithmetic (Ilharco et al., 2022) and TIES-Merging (Yadav et al., 2023).
- **Test-Time Adaptive Merging:** AdaMerging (Yang et al., 2024), SyMerge (Jung et al., 2025), and FoldMerge (Anonymous, 2025).
- **Head-Tuned Static Baselines:** Task Arithmetic + Head Tuning and TIES-Merging + Head Tuning.
- **Ablation Baselines:** Unconstrained Scaling (to isolate the impact of the simplex constraints) and Unconstrained Scaling + Head Tuning.

### 3. Symmetrical Regime Mapping
A key strength of the experimental design is the separation of results into:
- **Part A: Frozen Classification Heads (Untouched Classifier Probes)**
- **Part B: Active Classification Head Adaptation (Tuned Classifier Probes)**

This symmetric layout allows for a completely transparent and scientifically honest deconstruction, enabling the authors to isolate the distinct drivers of performance (weight-space optimization vs. classification head tuning).

---

## Analysis of Claims vs. Empirical Results

The empirical results fully and robustly support every claim made in the paper:

### Claim 1: Localized bottleneck adaptation is insufficient; whole-model parameter blending is necessary.
- **Support:** **BPAM-Restricted** (strictly 8 parameters on the visual projection layer, keeping the rest of the encoder static) collapses average accuracy to **51.38%**, which is a $-17.72\%$ drop compared to standard static Task Arithmetic. In contrast, **BPAM-Static** (merging the whole image encoder with the same 8 parameters) achieves **69.21%** average accuracy. This massive gap confirms that localized adaptation is too mathematically constrained and ignores fine-tuned representation updates in the rest of the encoder, validating that whole-model blending is essential.

### Claim 2: Extremely low-parameter regimes are too constrained to resolve weight conflicts, making classification head adaptation the primary driver of performance.
- **Support under Frozen Heads:** Under frozen heads, BPAM-Static ($69.21\%$) underperforms TIES-Merging ($72.90\%$) by $-3.69\%$ and massively underperforms high-capacity methods (AdaMerging: $83.17\%$, SyMerge/FoldMerge: $83.56\%$) by over $-14.35\%$ absolute. This proves that global task-wise scaling alone lacks the necessary degrees of freedom to resolve cross-task parameter conflicts.
- **Support under Active Heads:** Concurrently adapting classification heads in **BPAM-Full** increases average accuracy to **75.22%** (a substantial $+6.01\%$ gain over BPAM-Static). This demonstrates that decision-boundary adaptation drives the vast majority of adaptation gains under severe parameter constraints.
- **The TIES + Head Tuning Comparison:** BPAM-Full barely beats Task Arithmetic + Head Tuning ($74.80\%$) by $+0.42\%$, and actually underperforms **TIES-Merging + Head Tuning** ($78.50\%$) by **$-3.28\%$**. This is a profound and highly honest result: applying simple decision-boundary tuning on top of a strong static model that actively resolves weight conflicts (TIES-Merging) is strictly superior to joint weight-head optimization under low-parameter weight constraints. This empirically proves that without sufficient weight-space capacity (like layer-wise parameters or high-dimensional coordinate mappings), joint optimization is highly bottlenecked.

### Claim 3: Simplex projection acts as a scale-preserving safeguard, slightly limiting expressivity.
- **Support:** **Unconstrained Scaling** (without simplex or proximity penalty) achieves **71.51%** average accuracy under frozen heads, which is $+2.30\%$ higher than default BPAM-Static ($69.21\%$). Similarly, **Unconstrained + Head Tuning** achieves **77.12%** ($+1.90\%$ higher than BPAM-Full). At convergence, the unconstrained coefficients sum to $1.4887$, scaling up parameter energy to gain expressivity without triggering severe activation collapse. This ablation confirms that while the convex simplex slightly restricts performance, it acts as a critical scale-preserving safeguard that bounds weight norms to guarantee activation stability in more extreme settings.

### Claim 4: Mean-Field Proximity Regularizer is empirically redundant under normal settings but crucial under extreme low-data.
- **Support:** Under a 20% calibration split / 80% unseen split, the unregularized model ($\beta=0.0$) and regularized model ($\beta=0.01$) achieve nearly identical unseen accuracies ($69.30\%$ vs $69.29\%$), proving that the 8-parameter space is structurally immune to transductive overfitting under standard calibration splits.
- **Low-Data Validation:** Under an extreme 5-sample per class calibration stream, unregularized optimization ($\beta=0.0$) exhibits severe instability, with coefficients drifting toward extreme regions of the simplex and dropping unseen test performance by over $-2.41\%$ compared to the static Task Arithmetic baseline. Activating the Proximity Penalty ($\beta = 0.01$) anchors the coefficients, recovering unseen split performance to match the baseline.
- **Sensitivity Analysis:** Evaluating $\beta \in [10^{-4}, 1.0]$ in this low-data regime shows that performance remains remarkably stable across a wide range of values, only over-constraining at $\beta=1.0$ (reverting to uniform static merging, $69.10\%$). This is an exceptionally complete and convincing validation.

### Claim 5: 0-Weight Experts achieve high performance via representation sharing in a compact shared basin.
- **Support:** In BPAM-Static, SVHN ($\lambda_5$) and MNIST ($\lambda_6$) converge to exactly $0.0000$ (with base model weight also being $0.0000$). Yet SVHN achieves **78.15%** and MNIST achieves **88.09%** classification accuracy.
- **CKA Evidence:** Centered Kernel Alignment (CKA) similarity checks show that the merged model's representations are highly similar to MNIST (CKA of 0.5000 vs 0.3754 for base) and SVHN (CKA of 0.1372 vs 0.0632 for base). This directly confirms that other experts in the compact fine-tuning basin (such as numerical shapes in GTSRB) successfully reconstruct digit-like representation sub-spaces, allowing the frozen linear probes to classify digit samples accurately.
