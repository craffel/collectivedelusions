# Experimental Evaluation

## Evaluation of the Experimental Setup
The experimental setup designed by the authors is highly controlled, clean, and comprehensive:
- **Benchmark:** The use of `timm ViT-Tiny` (5.7M parameters) with four classification heads on distinct datasets (MNIST, FashionMNIST, CIFAR-10, SVHN) is a standard, robust testbed for multi-task model merging.
- **Dimensionality:** The coefficient search space consists of 56 parameters ($\Lambda \in [0, 1]^{4 \times 14}$). This is a reasonable and tractable search space that is consistent with test-time merging adaptation literature (e.g., AdaMerging).
- **Statistical Significance:** The authors report results as "mean $\pm$ standard deviation over three random seeds" across all major tables. This is an excellent practice that ensures the reported performance gaps are statistically significant rather than random fluctuations.

---

## Analysis of Baselines
The paper includes an exceptionally strong set of baselines that are crucial for a rigorous methodological deconstruction:
1. **FP16 Task Arithmetic (Unquantized Baseline):** Scoring $35.12\%$ average accuracy, this baseline reveals a high level of representation conflict and task interference among the experts (which individually achieve $>90\%$ unmerged accuracy). This establishes a highly challenging "stress test" environment for low-bit merging.
2. **Naive Merge-then-Quantize (M-then-Q):** Scoring $21.50\%$ average accuracy under 4-bit quantization, this establishes the starting point for post-hoc compression.
3. **Quantized AdaMerging:** Optimizing coefficients in FP16 to minimize prediction entropy, and then applying post-hoc quantization to INT4. This baseline is the most critical contribution of the experimental setup, as it isolates the effect of direct low-bit optimization (STE) from continuous-coefficient search.

---

## Consistency and Support of Results for Claims

The empirical results in the paper provide exceptionally strong, clear, and consistent support for all of the authors' primary claims:

| Claim | Empirical Evidence | Verdict |
| :--- | :--- | :--- |
| **STE is consistently outperformed by unquantized search** | Table 1 shows Quantized AdaMerging achieves $30.00\%$ average accuracy, while Q-Merge (STE) at matched $N=16$ peaks at only $26.25\%$. | **Fully Supported** |
| **Direct STE fails to consistently beat naive post-hoc baseline** | Table 1 shows Q-Merge ($N=1$) yields $17.00\%$ accuracy, which is worse than the naive M-then-Q baseline ($21.50\%$). | **Fully Supported** |
| **Continuous coefficients overfit to the source quantization operator** | Table 2 (Cross-Schema Matrix) shows catastrophic drops under schema shift. Moving from `asym_channel` (source) to `sym_tensor` (target) drops accuracy from $33.00\%$ to $12.63\%$ (a drop of $-20.37\%$). | **Fully Supported** |
| **Stochastic black-box search (1+1 ES) overfits boundaries even more intensely** | Table 3 shows 1+1 ES achieves $20.75\%$ on the source schema (beating STE's $17.88\%$) but collapses to $8.62\%$ on mismatched target (worse than STE's $10.12\%$), yielding a larger generalization gap ($-12.13\%$ vs $-7.76\%$). | **Fully Supported** |
| **Unsupervised entropy minimization collapses under class skew** | Table 4 shows a drop in average accuracy from $26.25\%$ to $15.50\%$ under severe class skew (Gini skew), with CIFAR-10 dropping to $8.50\%$ and SVHN to $6.50\%$. | **Fully Supported** |
| **Input Gaussian noise acts as a stochastic regularizer** | Table 4 shows average accuracy is $25.38\%$ under corrupted streams, which is very close to clean stream performance ($26.25\%$). | **Fully Supported** |
| **Supervised calibration prevents entropy collapse** | Table 5 shows supervised Q-Merge achieves $35.00\%$ (vs $26.25\%$ unsupervised) on standard streams, and $23.75\%$ (vs $15.50\%$ unsupervised) on highly skewed streams. | **Fully Supported** |
| **Subspace robustness is a low-capacity illusion** | Table 6 shows Low-Rank Subspace projection eliminates the cross-schema gap ($+0.50\%$), but its absolute performance collapses to $13.00\%$, indicating a degenerate state of information loss. | **Fully Supported** |

---

## Experimental Limitations

While the empirical findings are robustly supported, a critical reviewer should note the following limitations in the experimental scope:

1. **Limited Architectural Diversity:** The main evaluations are restricted to `ViT-Tiny` (5.7M parameters) and `ResNet-18` (11.7M parameters). Although convolutional networks and Vision Transformers are compared, both are lightweight models. Modern deep learning operates on much larger architectures (e.g., ViT-Base/Large, Llama-3, CLIP). 
2. **Post-Hoc SVD Projection as PEFT Proxy:** The authors use global SVD projection to mathematically model PEFT/low-rank subspace constraints. However, as noted in Section 4.7, this is a poor proxy for natively-trained PEFT (such as LoRA), which is optimized end-to-end. The "Low-Capacity Generalization Illusion" observed here is a consequence of SVD capacity destruction, and the paper fails to experimentally verify whether natively-trained LoRA experts maintain high performance while closing the Cross-Schema Generalization Gap.
3. **No Empirical Verification of Proposed "Hybrid Optimizer":** In Appendix B and Section 5, the authors formalize and advocate for a "Hybrid Optimization Pipeline" combining standard STE with 1+1 ES and Total Variation spatial regularization. However, they do not present any empirical results or testing for this proposed algorithm. It remains a purely speculative recommendation.
4. **Limited Seed Sweep:** Sweeping over 3 random seeds is a standard practice in resource-constrained environments, but for low-bit landscapes (which are notoriously noisy and seed-sensitive), a wider sweep (e.g., 5 or 10 seeds) would provide higher statistical confidence.
