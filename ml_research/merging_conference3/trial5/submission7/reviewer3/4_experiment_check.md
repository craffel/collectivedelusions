# Intermediate Evaluation: Experiment Check

## Critical Evaluation of the Experimental Setup

### 1. Low-Scale and Outdated Backbone and Datasets
The experimental evaluation is restricted to a **very low-scale and outdated** setting:
- **Backbone:** The authors use `vit_tiny_patch16_224` (approximately 5.7M parameters). Modern model merging research is conducted on models with billions of parameters (e.g., LLaMA-2 7B/13B, Mistral-7B) or large-scale vision models (e.g., ViT-Base/Large, CLIP-L). It is unclear if the findings on a toy 5.7M parameter ViT scale to larger, more complex models where parameter conflicts and optimization landscapes are drastically different.
- **Datasets:** The evaluation relies on classic toy datasets: MNIST ($28 \times 28$ grayscale), FashionMNIST ($28 \times 28$ grayscale), CIFAR-10 ($32 \times 32$ color), and SVHN ($32 \times 32$ color). These datasets are heavily saturated, extremely small in resolution, and do not represent the complexity of real-world multi-task learning or model merging applications (e.g., merging specialized instruction-following LLMs or high-resolution domain adapters).

### 2. Complete Absence of Statistical Significance / Error Bars
The empirical findings are presented as single-point values in Table 1 and Table 2. There are **no standard deviations, error bars, or confidence intervals**, and no mention of testing over multiple random seeds. 
This is a critical flaw because of the extremely small sample sizes used:
- **Calibration Set:** 64 images total (16 per task).
- **Test Set:** 512 images per task.

Let's calculate the margin of error for a sample size of 512. For an accuracy of approximately $62\%$ ($p = 0.62$), the standard error of a proportion is:
$$\text{SE} = \sqrt{\frac{p(1-p)}{n}} = \sqrt{\frac{0.62 \times 0.38}{512}} \approx 2.15\%$$
A standard $95\%$ confidence interval has a margin of error of $1.96 \times 2.15\% \approx 4.2\%$. 

**The Flaw:**
- The difference between PG-Merge ($p=0.05$) and the static Uniform Merging baseline is only **$0.54\%$** ($62.70\%$ vs. $62.16\%$).
- The difference between PG-Merge ($p=0.05$) and RegCalMerge is only **$0.35\%$** ($62.70\%$ vs. $62.35\%$).
Both of these differences are **significantly smaller than the statistical margin of error ($\pm 4.2\%$)** for the 512-image test set. In fact, a $0.54\%$ change on a 512-image test set corresponds to only **2.7 images** being classified differently. Without running multiple trials with different random splits of the calibration and test sets, it is highly likely that the claimed "state-of-the-art" joint mean of PG-Merge is an artifact of random noise rather than a systematic algorithmic advantage.

### 3. Inconsistent Task Performance and Failure to Outperform Static Baseline Consistently
Looking at the individual task results in Table 1:
- On **MNIST**, the static Uniform Merging baseline achieves **$65.04\%$**, which is **$1.76\%$ higher** than PG-Merge ($63.28\%$).
- On **SVHN**, Uniform Merging achieves **$33.20\%$**, which is **$1.17\%$ higher** than PG-Merge ($32.03\%$).
- PG-Merge only outperforms Uniform Merging on **Fashion** ($75.59\%$ vs $72.07\%$) and **CIFAR-10** ($79.88\%$ vs $78.32\%$).

**The Flaw:** PG-Merge does not consistently outperform Uniform Merging across the four tasks. In fact, on 2 out of 4 tasks, the static, completely training-free Uniform Merging is superior. 
The joint average is pulled up slightly by Fashion and CIFAR-10, but the overall gain is marginal. Given that Uniform Merging requires **zero optimization steps, zero calibration data, zero hyperparameter tuning, and zero computational overhead**, the extremely small, statistically insignificant $0.54\%$ joint accuracy gain of PG-Merge is not practically justified. The paper fails to demonstrate that active test-time adaptation is actually beneficial compared to a simple, robust static average.

### 4. Poor Baseline Representation and Trajectory Collapse
- **PolyMerge Collapse:** PolyMerge's joint performance is reported as a catastrophic **$46.97\%$**, with MNIST collapsing to near-random ($13.48\%$). This suggests a major bug in the implementation of the PolyMerge baseline. PolyMerge is designed to restrict coefficients to a low-degree polynomial subspace over depth. If properly optimized, PolyMerge should easily match or exceed Uniform Merging (since Uniform Merging is a special case of PolyMerge where the polynomial is a constant $0$-degree polynomial $\alpha_{k, l} = 0.3$). The fact that PolyMerge collapses to near-random suggests that the baseline was poorly optimized, had an inappropriate learning rate, or was severely misconfigured, making the comparison unfair and highly biased.
