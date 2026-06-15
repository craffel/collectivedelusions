# 4. Experiment Check

## Rigor of Experimental Setup and Datasets
The experimental setup is exceptionally rigorous, well-designed, and controlled:
* **Controlled Benchmark:** The authors use `vit_tiny_patch16_224` (5.7M parameters) with 4 classification tasks (MNIST, FashionMNIST, CIFAR-10, SVHN). The individual expert models are highly optimized, establishing an unmerged average upper bound of $90.88\%$.
* **Challenge Level:** The unquantized FP16 Task Arithmetic baseline scores only $35.12\%$. This indicates that the fine-tuned experts have diverged significantly, creating a high-interference, highly non-convex loss landscape. Evaluating quantization-aware merging under this high-conflict scenario is an outstanding stress-test that exposes optimization vulnerabilities that would be masked in cooperative, low-interference setups.
* **Statistical Rigor:** All reported averages include standard deviations computed over three random seeds (e.g., Q-Merge $N=16$ scores $26.25 \pm 0.58\%$).

## Evaluation of Baselines
The paper includes a highly comprehensive and complete suite of baselines that successfully isolate and evaluate the independent variables:
1. **FP16 Task Arithmetic (35.12%):** Establishes the unquantized merging task-interference ceiling.
2. **Naive Merge-then-Quantize (21.50%):** Represents the default post-hoc quantization baseline without adaptation.
3. **Quantized AdaMerging ($N=16$, 30.00%):** A crucial baseline that optimizes continuous coefficients in FP16 to minimize entropy and applies post-hoc target quantization. This isolates whether direct low-bit optimization under quantization constraints is actually necessary.
4. **Supervised Calibration Baseline (Table 5):** Evaluates Q-Merge using supervised cross-entropy directly on the $N$-sample calibration set (standard: $35.00\%$, skewed: $23.75\%$), decoupling dataset scarcity from the unsupervised entropy objective's failures.

## Support for Core Claims
The empirical results provide overwhelming, statistically robust support for all major claims in the paper:

### Claim 1: Direct low-bit optimization via STE is not necessary and underperforms full-precision search.
* **Support:** Quantized AdaMerging ($30.00\%$) substantially outperforms matched Q-Merge ($26.25\%$) by an absolute $3.75\%$. Furthermore, lowering the learning rate of Q-Merge ($10^{-3}$ and $10^{-4}$) fails to close this gap (yielding $22.50\%$ and $21.62\%$), proving that STE gradient noise is a structural bottleneck rather than a hyperparameter tuning issue.

### Claim 2: Learned coefficients overfit to the simulated operator (Cross-Schema Generalization Gap).
* **Support:** Table 2 clearly shows that moving from `sym_channel` ($17.88\%$ matched) to `sym_tensor` collapses accuracy to $10.13\%$ (a $-7.75\%$ drop, near the 10.00% random-guess floor). Moving from `asym_channel` ($33.00\%$ matched) to `sym_tensor` triggers a catastrophic $20.37\%$ performance collapse (down to $12.63\%$). Double quantization (`double\_quant`) is highly resilient, dropping only $2.00\%$ (down to $31.00\%$), which is logically supported by its minimal scale discretization error. Optimal unquantized initialization fails to close this gap.

### Claim 3: 1+1 ES improves local search but worsens cross-operator overfitting.
* **Support:** Table 3 shows that 1+1 ES achieves higher matched source accuracy than STE ($20.75\%$ vs $17.88\%$). However, it suffers a worse collapse on mismatched target schemas, dropping to $8.62\%$ (generalization gap of $-12.13\%$), compared to STE's drop to $10.12\%$ (gap of $-7.76\%$). This supports the claim that black-box search finds hyper-customized local minima on rounding boundaries, while STE gradients exert an implicit regularizing effect.

### Claim 4: Continuous spatial smoothers are inadequate.
* **Support:** Table 3 and Appendix A verify that TV regularization fails to rescue target performance under shift ($9.50\%$). Weak regularization ($\alpha = 0.05$ or $0.10$) has no smoothing impact, while strong regularization ($\alpha \ge 1.0$) over-regularizes coefficients toward linear ensembling, damaging source accuracy ($11.25\%$) without helping target accuracy ($9.00\%$).

### Claim 5: Unsupervised entropy minimization is vulnerable to class skew.
* **Support:** Gini skew drops unsupervised Q-Merge accuracy to $15.50\%$. The supervised calibration baseline (Table 5) rescues this performance to $23.75\%$ ($+8.25\%$), confirming that prediction entropy minimization is blind to labels and collapses to degenerate shortcut states under skew.

### Claim 6: CNN and subspace generalizability.
* **Support:** Table 6 shows ResNet-18 has a smaller gap ($-4.25\%$) than ViT-Tiny ($-7.76\%$), validating that localized CNN spatial kernels are more resilient. Low-rank SVD projection closes the gap ($+0.50\%$) but collapses absolute performance to $13.00\%$, validating the "Low-Capacity Generalization Illusion" claim.

The empirical evidence is exhaustive, consistent, and provides a masterclass in critical research auditing.
