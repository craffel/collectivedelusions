# 4. Experimental Evaluation

## Critical Evaluation of the Experimental Setup

### 1. Toy Scale of the Primary Benchmark
* **SimpleCNN on MNIST/FashionMNIST/KMNIST:** The primary end-to-end evaluation is conducted on a custom SimpleCNN model containing only ~500,000 parameters, trained on MNIST, FashionMNIST, and KMNIST (28x28 grayscale images).
* **Toy-like Scope:** While a toy setting is a useful sanity check for initial proof-of-concept, it is highly outdated for a modern (2026) paper on model merging. Model merging is a paradigm designed specifically to alleviate the storage, serving, and fine-tuning costs of **large foundation models** (e.g., LLMs, deep ViTs). 
* **Lack of Generalizability Proof:** Success on a 500k parameter CNN on MNIST does not provide strong evidence that the method's advantages (such as mitigating scale mismatches) will translate to deep, multi-billion parameter architectures with complex attention and feed-forward layers.

---

## Statistical Significance and Rigor

### 1. Statistical Insignificance on the Main Benchmark
Looking closely at the results in Table 1, the standard deviation of the accuracy across 3 independent seeds is exceptionally high (~1.1% to 2.5%), which obscures any clear benefit of the proposed methods:
* **Overlapping Confidence Intervals (Task Arithmetic vs. RMS-Scale):** 
  * Tuned Task Arithmetic: $72.50 \pm 1.17\%$ (Interval: $[71.33\%, 73.67\%]$)
  * Tuned RMS-Scale: $73.22 \pm 2.15\%$ (Interval: $[71.07\%, 75.37\%]$)
  * The difference of $0.72\%$ is substantially smaller than the standard deviations. The confidence intervals heavily overlap, meaning we cannot statistically conclude that RMS-Scale outperforms standard linear merging on this benchmark.
* **Overlapping Intervals (SVD Isotropic vs. RMS-Scale):**
  * SVD Isotropic: $73.13 \pm 2.49\%$ (Interval: $[70.64\%, 75.62\%]$)
  * Tuned RMS-Scale: $73.22 \pm 2.15\%$ (Interval: $[71.07\%, 75.37\%]$)
  * The difference of $0.09\%$ is completely negligible and statistically insignificant.
* **Overlapping Intervals (Un-tuned Baselines vs. PF-RMS):**
  * Task Arithmetic (Default): $71.68 \pm 1.36\%$ (Interval: $[70.32\%, 73.04\%]$)
  * Ties-Merging (Default): $71.81 \pm 1.73\%$ (Interval: $[70.08\%, 73.54\%]$)
  * PF-RMS (Ours): $72.23 \pm 2.25\%$ (Interval: $[69.98\%, 74.48\%]$)
  * The difference of $0.55\%$ is well within the margin of noise.

The statistical evidence from this toy benchmark is extremely weak. The claim that RMS-Scale and SD-Scale "exceed standard Task Arithmetic... and Ties-Merging" is not robustly supported by the data due to the high seed-to-seed variance on this small CNN.

---

## Fairness of Baseline Comparisons

### 1. Severe Performance Collapse of AdaMerging
* **AdaMerging Collapse:** AdaMerging yields only $62.79 \pm 6.64\%$, which is a massive drop from standard linear averaging (72.50%). In established model-merging literature, AdaMerging is known to be a strong baseline that typically outperforms Task Arithmetic on standard vision benchmarks.
* **Suspected Tuning Deficits:** Why does AdaMerging collapse so severely on this custom SimpleCNN? It is highly likely that AdaMerging was under-tuned or its optimization parameters (learning rate, entropy minimization step count, batch size) were poorly calibrated for this specific toy network. Without a sensitivity study or detailed explanation of AdaMerging's tuning pipeline, this comparison appears unfair.

### 2. Ties-Merging Anomalies
* **Tuned Worse Than Default:** Table 1 reports that Ties-Merging (Validation-Tuned) gets $71.77 \pm 2.06\%$, which is actually *worse* than Ties-Merging (Default, $\lambda=1.0$) at $71.81 \pm 1.73\%$. This is mathematically counter-intuitive since the search space for validation tuning includes $\lambda=1.0$. This anomaly points to a potential issue in the validation-set tuning pipeline (e.g., a noisy validation set, severe overfitting, or a bug in the evaluation script).
* **Underperformed Task Arithmetic:** In literature, Ties-Merging almost always outperforms Task Arithmetic due to its pruning of redundant updates and resolution of sign conflicts. Here, Ties-Merging (71.77%) underperforms Task Arithmetic (72.50%). The authors should explain why pruning 60% of parameters in a tiny 500k parameter model was appropriate, as such heavy pruning might have severely damaged the capacity of the small layers.

---

## Support for Key Claims
* **Claim:** Minimalist scale calibration is highly sufficient and robust for resolving task interference. (Support: **Weak**. The statistical benefits on the main benchmark are negligible and within the margin of noise).
* **Claim:** The method scales seamlessly to deep, multi-billion parameter architectures. (Support: **Weak**. This is supported only by a *simulated* activation-alignment experiment on CLIP, without any downstream task accuracy results. The simulation used potentially isotropic updates, which artificially forced the convergence to the orthogonal limit).
* **Claim:** PF-RMS provides a highly robust, training-free, and parameter-free baseline. (Support: **Moderate**. While it does perform comparably to the tuned variants without needing a grid-search, its absolute improvement over default un-tuned baselines is still within the margin of statistical noise).
