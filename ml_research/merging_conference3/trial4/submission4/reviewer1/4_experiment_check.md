# 4. Experimental Evaluation and Claims Check

A rigorous review of the experiments and results reveals several critical weaknesses, unexamined assumptions, and gaps between the empirical evidence and the authors' high-level claims.

## Evaluation of Datasets and Benchmarks
1. **Predominance of the Synthetic Simulation Landscape (Model II):**
   The vast majority of the paper's multi-seed experiments (Tables 1, 2, 3 and Figures 1, 2) are conducted on a *calibrated mathematical simulation landscape (Model II)*. While the authors claim this is a "highly rigorous standard methodology proxy," it is ultimately a simplified, low-dimensional quadratic approximation of the true neural network loss landscape. Real neural loss landscapes are highly non-convex, containing complex saddle points and local minima. Testing primarily in a simulation weakens the empirical foundations of the paper.
2. **Artificial and Weak Physical Benchmarks:**
   To address the simulation gap, the authors implement physical neural network experiments. However, these setups are extremely toy:
   - The MLP experiment uses a simple 12-layer model on a synthetic multi-task classification dataset.
   - The ResNet-18 experiment uses a binary split of CIFAR-10 (Task 0: Vehicles vs Task 1: Animals) with only 120 fine-tuning samples. The task expert accuracies are very low (86.00% and 65.00% respectively), making the experts extremely weak. 
   - A standard, large-scale model merging evaluation on full datasets (e.g., merging fully-trained ResNet-50s or Vision Transformers on ImageNet or full CIFAR-10, or LLMs on GLUE) is entirely missing.

## Critical Analysis of Baselines
1. **The Global Task-Wise (DC-Only) Superiority Under TTA:**
   In Table 1 (Standard Clean Stream), **Online Global Task-Wise (DC-Only)** achieves an average accuracy of **85.91%**, which outpeforms **Online SpectralMerge-LP ($F=3$)** (85.32%) and **Online SpectralMerge-Reg** (85.17%). 
   - Under Online TTA, a simple global scalar per task is superior to the proposed spectral methods. The authors fail to explain why a practitioner should adopt the mathematical and algorithmic complexity of SpectralMerge when a simple 1D global baseline is both more effective and easier to implement.
2. **Missing Baselines on Physical Experiments:**
   While the authors compare against PolyMerge, Layer-wise, and Uniform on the physical PyTorch models, they **do not** include Online AdaMerging, RegCalMerge, or Online PolyMerge on these physical models. They only evaluate them on the simulation benchmark. This raises concerns about whether these online baselines would fail or perform poorly on physical models.

## Do the Results Support the Claims?

### Claim 1: "SpectralMerge completely resolves the Overfitting-Optimizer Paradox."
* **Reality Check:** This claim is only partially supported and fails for the hard-cutoff variant on real models.
  - In Table 3 (Section 4.6), **SpectralMerge-LP ($F=3$)** and **LP-Adaptive** completely collapse to **29.00%** accuracy (the majority-class guessing baseline). 
  - The authors explain this collapse via the "PEFT-Induced Step-Function Discontinuity." Since only a few layers were fine-tuned, the optimal trajectory has infinite frequency support, and a low-pass hard cutoff cannot reconstruct it.
  - This means that one of the two core proposed variants (SpectralMerge-LP) is **completely non-functional** on physical architectures fine-tuned with standard localized parameter-efficient protocols. It only works on the continuous, smooth simulation benchmark. Only the regularized variant (SpectralMerge-Reg) survives.

### Claim 2: "SpectralMerge is highly resilient/immune to validation target selection bias and adversarial stream noise."
* **Reality Check:** This "immunity" is a trivial property of any offline frozen method, not a unique feature of SpectralMerge.
  - In Table 2, **OFS-Tune SpectralMerge-LP (M=10)** achieves 86.46% accuracy across all adversarial streams.
  - This occurs because OFS-Tune is optimized offline on a validation set and then **frozen** during test-time. It does not adapt to the test-time stream, so it is naturally unaffected by test-time non-stationarity.
  - Crucially, the **OFS-Tune Global Task-Wise (DC-Only, M=10)** baseline is also completely frozen and achieves 85.42% accuracy across all streams.
  - Framing this as a "robustness breakthrough" unique to SpectralMerge is highly misleading. It is simply the standard, expected property of freezing a model's weights before testing.

## Statistical Significance and Missing Error Bars
* **The Discrepancy:** The authors report standard deviations over 30 random seeds for the simulation benchmark (Tables 1 and 2), which is commendable.
* **The Flaw:** However, for the physical MLP and pre-trained ResNet-18 CIFAR-10 experiments (Table 3, Figures 3 and 4), **there are no standard deviations, error bars, or multiple seeds reported.**
* **The Concern:** The offline tuning is performed on extremely tiny validation datasets ($M=10$ or $M=15$). With such small sample sizes, the optimization is highly susceptible to high variance and selection bias in the validation samples. Without reporting average results and standard deviations over multiple random validation splits for the physical models, the reported single numbers (such as 54.00% for SpectralMerge-Reg vs 29.00% for spatial/polynomial) could be highly unstable or cherry-picked.
