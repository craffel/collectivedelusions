# 4. Experimental Evaluation Critique

## Experimental Setup Quality
The experimental setup has two major, limiting weaknesses:
1. **Toy-Scale Evaluation:** The physical validation is restricted strictly to **ViT-Tiny** (5.7M parameters) and four extremely small, simple classification datasets (**MNIST, FashionMNIST, CIFAR-10, SVHN**). Modern model merging research is focused on large language models (LLMs) with billions of parameters or large vision-language models on highly complex tasks. Evaluating solely on an extremely small vision backbone on toy, low-resolution datasets severely limits the generalizability of the findings. The "Scaling Analysis" in Section 4.5 is purely theoretical and hypothetical, lacking any empirical validation on larger backbones (e.g., ViT-Base, ResNet-50) or other modalities (e.g., NLP).
2. **Artificial Low-Data Regime:** The authors sample only 2048 training images per task and fine-tune for just 10 epochs. This creates an artificial "low-data, partially converged" regime. This setup appears designed to keep the expert task vectors small so that the local quadratic Taylor approximation doesn't completely break down. In realistic deployment scenarios, merged experts are fully converged models with large task vectors, where ACM's local Taylor expansion is mathematically shown to exhibit high cubic approximation error.

## Baseline Selections and Fairness
The authors compare ACM against appropriate baselines: Task Arithmetic, diagonal Fisher Merging, and three Test-Time Adaptation (TTA) baselines (AdaMerging, PolyMerge, RegCalMerge). However, the fairness of the comparison is highly questionable:
* **Strawman TTA Setup:** The TTA baselines collapse to extremely low accuracies on physical ViT-Tiny (e.g., PolyMerge at 38.96%, AdaMerging at 55.42%). The authors attribute this to "transductive overfitting" on the tiny 32-sample calibration batch over 15 optimization steps. However, TTA methods are designed to adapt over larger, continuous streaming target distributions. Forcing them to optimize on an extremely restricted, low-data 32-sample batch represents a strawman experimental setup that guarantees their failure, creating an artificial advantage for ACM.
* **Tuning Inconsistency:** Task Arithmetic is labeled as "Best Tuned 0.4," meaning its static scale factor was tuned directly on the test set. While this makes Task Arithmetic a highly competitive Oracle baseline, it exposes a major flaw: a simple, zero-calibration baseline tuned with a single scalar outperforms or matches the highly complex, mathematically heavy ACM variants across the board.

## Support for Central Claims
The empirical results **do not** support the central claims of the paper regarding the practical superiority of curvature-aware analytical merging.
* **Overstated Abstract Claims:** The abstract claims that ACM-GlobalNorm achieves a Joint Average accuracy of 57.76%, "significantly outperforming both diagonal Fisher Merging (56.03%) and polynomial test-time adaptation (38.96%)."
* **The Hidden Truth in Table 2:** The authors bury the comparison with the most critical and standard baseline: **Task Arithmetic (Best Tuned 0.4)**, which achieves **60.72%** average accuracy.
* **The Breakdown of Proposed ACM Variants:**
  * **Vanilla ACM** ($60.89\%$) barely beats Task Arithmetic ($60.72\%$) by a statistically insignificant **$0.17\%$** absolute accuracy.
  * **ACM-Norm** ($58.89\%$) is **$-1.83\%$** worse than Task Arithmetic.
  * **ACM-GlobalNorm** ($57.76\%$) is **$-2.96\%$** worse than Task Arithmetic.
  * **Lasso ACM-GlobalNorm** ($57.52\%$) is **$-3.20\%$** worse than Task Arithmetic.
  * **Gauss-Seidel Coordinated ACM-GlobalNorm** ($36.65\%$) represents a complete catastrophic collapse.
* **The Contradiction of Scale Normalization:** In Section 3.4 and 3.5, the authors pitch scale-normalization (ACM-Norm and ACM-GlobalNorm) as essential contributions to prevent "sacrificial task bias" and achieve balanced multi-task merging. Yet, Table 2 reveals that these proposed normalized systems perform significantly *worse* than the unnormalized Vanilla ACM and the standard Task Arithmetic baseline.

Therefore, the experimental results empirically disprove the core premise of the paper: modeling high-dimensional non-diagonal curvature via local quadratic surrogates does not yield a superior merged model in physical settings. Instead, a simple, uniform, curvature-blind interpolation (Task Arithmetic) is highly robust and superior to almost all proposed ACM variants.
