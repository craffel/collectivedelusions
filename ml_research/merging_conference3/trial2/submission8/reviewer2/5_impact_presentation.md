# Evaluation of Presentation, Major Strengths, and Potential Impact

## 1. Major Strengths
- **Clear and Engaging Narrative:** The paper is exceptionally well-structured, starting with a pragmatically grounded introduction to edge-deployment challenges, moving logically through the methodology, and concluding with a highly detailed empirical and theoretical appendix.
- **Strong Mathematical Appendix:** The authors provide rigorous analytical derivations of the expected $L_1$ norm ratios and expected $L_2$ reconstruction errors under both Laplace and Gaussian weight-residual distributions. This elevates the work above a purely heuristic empirical paper.
- **Genuinely Practical Insights for Edge AI:** The detailed storage analysis (CSR/COO calculations, immediate 5-20x storage savings) and the synergy experiments with INT8 quantization (yielding a 4.0x additional storage reduction with only a 0.12% drop in accuracy) are highly relevant and immediately useful for edge-deployment practitioners.
- **Rigorous Statistical Verification:** The authors evaluate their methods over three independent random seeds and conduct a two-tailed paired $t$-test to verify that the performance difference between global Uniform Pruning and Adaptive Saliency Pruning is highly statistically insignificant ($p > 0.05$), highlighting that the simpler uniform approach is pragmatically superior.

## 2. Major Areas for Improvement
- **Address the Merging Coefficient Equivalence Flaw:** The core theoretical claim that "norm-preserving rescaling" is a crucial new post-hoc operator is undermined by its mathematical equivalence to scaling up the merging coefficient ($\bar{\lambda}_k = \lambda_k / p$). The authors must acknowledge this equivalence and expand the hyperparameter search space of the unrescaled baseline in their experiments to ensure a scientifically fair comparison.
- **Resolve Naming Inconsistencies:** The term "Norm-Preserving" is a misnomer, as the proposed reciprocal scaling actually over-scales (boosts) the expected $L_1$ norm by 2.58x to 3.30x. The paper should reframe this heuristic as "Signal-Strength Boosting" or "Over-scaled Pruning" to avoid conceptual contradictions.
- **Validate on Larger Datasets and Architectures:** Evaluating on toy datasets (MNIST, Fashion, CIFAR-10, SVHN) in a tiny low-data regime (1024 samples) heavily overparameterizes the CLIP ViT-B/32 backbone. This makes the fine-tuned task vectors extremely small, artificially buffering the models against pruning. The authors must validate their framework on larger datasets (e.g., ImageNet-1K) and modern LLM backbones to prove that their findings hold in standard fine-tuning regimes.
- **Match Baseline Sparsity Levels:** The authors compare NP-BTVP-U at 90% sparsity ($p=0.10$) with DARE-Merging at 80% sparsity ($p_{\text{drop}}=0.80$). To ensure a rigorous comparison, both methods should be evaluated at identical sparsity levels (e.g., both at 90% and both at 80%).

## 3. Presentation Quality
The presentation quality of the paper is **Excellent**.
- The writing style is formal, mathematically precise, and polished.
- The equations are clearly written and typeset.
- The tables and figures (e.g., the pruning resilience curves and merging method comparison) are clean, professional, and directly support the narrative.

## 4. Potential Impact and Significance
The potential impact of this paper is **Moderate**.
- **Practical Impact:** High. The combination of 10% sparse task vectors with INT8 quantization to achieve a 40x compression footprint with negligible accuracy loss is an excellent recipe for deploying multi-task experts on edge/IoT hardware. The insight that layer-wise saliency pruning introduces inter-layer scale instability (the Saliency Double-Bind) provides valuable design guidelines for engineering teams.
- **Theoretical/Scientific Impact:** Low. Because the core "Norm-Preserving Rescaling" mechanism is mathematically equivalent to simply scaling the merging coefficient $\lambda_k$, the paper does not introduce a fundamentally new mathematical paradigm for model merging. The observation that loss landscape flatness (SAM) does not provide a pruning buffer is interesting, but is easily explained by standard optimization theory and the geometry of overparameterized interpolation.
