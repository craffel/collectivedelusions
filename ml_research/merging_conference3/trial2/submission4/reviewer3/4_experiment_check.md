# 4. Experimental Evaluation and Claims Check

## Evaluation of Experimental Setup and Datasets
The paper employs a highly appropriate and standard multi-task model merging benchmark:
- **Model Backbone:** Pre-trained Vision-Language CLIP ViT-B/32.
- **Benchmark Tasks:** 8 diverse classification datasets (SUN397, Stanford Cars, RESISC45, EuroSAT, SVHN, GTSRB, MNIST, DTD).
- **Evaluation Size:** Up to 1024 images per task, which the authors statistically justify as highly representative. The calculated standard error of the multi-task average is extremely narrow ($SE_{\text{avg}} \approx 0.51\%$), ensuring that the reported differences are statistically meaningful.

This setup is clean, standard in the model merging literature, and sufficient to evaluate the claim of multi-task capability preservation.

## Evaluation of Baselines
The authors do not set up "strawman" baselines. Instead, they evaluate against:
1. **Oracle Experts:** Evaluates the unmerged fine-tuned models on their respective tasks, establishing the absolute upper-bound performance (91.02% average accuracy).
2. **Thoroughly Optimized Task Arithmetic (TA):** To avoid comparing against an unoptimized baseline, the authors perform a detailed grid sweep over the global weight-scaling factor $\lambda \in [0.10, 0.80]$ to identify the absolute peak performance of standard Task Arithmetic (68.74% at $\lambda = 0.20$).
3. **Advanced Static Merging Methods:** Includes Git Re-Basin (41.50%), ZipIt! (49.30%), and TIES-Merging (61.20%), highlighting that linear weight permutations or redundant task-vector pruning fail to resolve representations across heterogeneous CLIP tasks.
4. **Server-Grade Adaptive Merging (SyMerge):** Serves as the representative for state-of-the-art gradient-based test-time adaptation methods (89.74% average accuracy, but requiring a 10-minute optimization window and high memory footprint).
5. **Decoupled Task Arithmetic (DTA, control baseline):** To isolate the contribution of Decoupled Scale Routing (DSR), they evaluate standard Task Arithmetic under the same decoupled scaling ($\lambda_{static}=0.25, \lambda_{proj}=0.20$), which achieves 69.45% accuracy.

This baseline selection is highly rigorous, transparent, and comprehensive.

## Alignment Between Results and Claims
The empirical results strongly support the paper's core claims:
1. **Extreme Efficiency:** The reported calibration preparation time of **11.95 seconds** for EdgeMerge represents a massive **$50\times$ speedup** over SyMerge (600 seconds) while utilizing only ~100 MB of training-free, forward-only GPU memory. This strongly supports the claim of being highly deployable and practical for edge environments.
2. **Performance Preservation:** EdgeMerge achieves 68.69% accuracy under coupled scaling, matching the peak optimized Task Arithmetic baseline (68.74%). Under Decoupled Scale Routing (DSR), it reaches 69.58% accuracy, which is a +0.84% absolute improvement over Task Arithmetic's peak.
3. **Plateau Preservation / Hyperparameter Stability:** This is perhaps the most robustly supported claim. Figure 3 and Table 4 show that standard Task Arithmetic is extremely fragile (rapidly collapsing to $<30\%$ accuracy as $\lambda$ diverges from 0.20). EdgeMerge, however, exhibits a broad performance plateau across $\lambda \in [0.20, 0.35]$, providing a crucial engineering "safety guardrail" for real-world deployment.
4. **Data-Free, Seed-Invariant Calibration:** Table 2 shows that the standard deviation of EdgeMerge's accuracy across multiple seeds is exactly $0.000\%$. Table 3 confirms that using synthetic Gaussian noise or pure zero tensors for calibration achieves the exact same multi-task accuracy (68.69%). The high Cosine Similarity ($\sim$0.91) and Spearman Correlation ($\sim$0.52) in Table 5 provide robust quantitative proof for the manifold-projection hypothesis.
5. **Representational Invariance:** Table 6 compares Mismatched Calibration (using $X_k^{base}$) vs. Correct Calibration (using $X_k^{expert}$). The resulting multi-task average accuracies are virtually identical (matching to three decimal places), verifying that the base feature reuse shortcut is mathematically sound and functionally flawless.

## Critical Comments and Limitations
1. **The Accuracy Gap:** While EdgeMerge is highly efficient, there is an undeniable **21.05% performance gap** compared to server-grade SyMerge (68.69% vs. 89.74%). The authors are highly transparent about this trade-off, framing EdgeMerge as an extreme-efficiency exploration for resource-constrained edge systems rather than a raw accuracy competitor under unconstrained conditions.
2. **Ablation Transparency:** The ablation study (Section 5.3.4) reveals that the dynamic channel gating weights ($\alpha_k[j]$) do not actually outperform uniform gating once Decoupled Scale Routing (DSR) is applied. This means the core performance benefit is driven by the decoupled scaling factors (DSR) rather than the localized activation-salience routing itself.
