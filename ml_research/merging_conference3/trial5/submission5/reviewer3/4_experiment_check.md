# Paper Evaluation: 4. Experimental Setup and Empirical Validity

## Experimental Setup & Datasets
The authors design an **Isolating Coordinate Sandbox** to evaluate routing algorithms under controlled conditions. This synthetic environment mimics the representation space of a pre-trained Vision Transformer (ViT-Tiny, $L=14$ layers, $D=192$ feature dimensions) on $K=4$ simulated task domains: MNIST, FashionMNIST, CIFAR-10, and SVHN. 
- **Calibration Split (64 samples):** Matches the real-world constraint of dynamic test-time model merging (TTA), where only a minute unlabeled or unlabeled-calibration split is available to adapt routing weights.
- **Test Split (1000 samples):** Provides a sufficiently large and balanced evaluation split (250 samples per task) to measure multi-task generalization.

## Baseline Quality and Sweep Completeness
The baseline selection is exceptionally comprehensive, including:
1. **Uniform Merging:** A standard static baseline that acts as a fundamental floor.
2. **QWS-Merge SOTA:** The state-of-the-art "quantum-inspired" dynamic router.
3. **L3-Router Variants:** Including L3-Linear, L3-Tanh, and L3-Softmax, each evaluated in both **unregularized** and **regularized ($L_2$ weight decay)** configurations.
4. **Global Linear Router:** The ultimate baseline champion that represents the simplest classical formulation.

The empirical sweeps are highly thorough and designed to proactively address potential confounding variables:
- **Task Correlation Audit (Section 13):** Introduces a task correlation parameter $\rho \in \{0.0, 0.25, 0.50, 0.75\}$ to prove that classical linear routing's superiority is not an artifact of orthogonal task subspaces. It demonstrates that as tasks share more dimensions, classical routing continues to dominate wave-based routing.
- **Learning Rate Audit (Section 9):** Rules out bad optimization choices for QWS-Merge by sweeping learning rates from $10^{-2}$ down to $10^{-4}$ under strict initialization controls, proving that QWS's collapse is a fundamental structural property of its non-monotonic cosine landscape.
- **Multi-Seed Robustness Audit (Section 12):** Evaluates all methods across 5 independent random seeds with complete dataset regeneration, demonstrating that the collapse of QWS and the superiority of regularized classical projections are statistically robust behaviors (QWS Joint Mean: $33.34\% \pm 9.51\%$, Global Linear: $69.68\% \pm 1.11\%$).

## Do the Results Support the Claims?
Yes, the empirical findings strongly support the authors' central claims:
1. **Unstability of wave cosine activations (QWS-Merge):** The SOTA model collapses to $36.10\%$ mean accuracy (underperforming uniform merging) and collapses catastrophically to $2.00\%$ on OOD SVHN (Table 2).
2. **Superiority of Layer-wise Linear Routing:** Our L3-Linear router avoids this collapse, achieving $63.10\%$ Joint Mean accuracy ($+27.00\%$ over QWS-Merge).
3. **The Layer-Wise Over-Engineering Confounder:** The simple global classical Linear Router achieves $67.20\%$ Joint Mean (outperforming all layer-wise models). This proves that introducing layer-wise specialized routing parameter space for classification heads is redundant and introduces unnecessary optimization noise (as proven mathematically by layer-averaging collapse in Section 3.5).
4. **Exposing the Overfitting Confounder on SVHN:** Table 2 and Section 4.3 show that classical layer-wise routers collapse on SVHN to $9.60\%$--$12.80\%$ without regularization, but applying $L_2$ weight decay lifts L3-Linear to $13.20\%$. The authors also rigorously discuss that the severe collapse on SVHN is driven by the low separability and high noise of the SVHN prototypes (expert ceiling of only $32.00\%$), which makes unregularized routing algorithms ignore the SVHN coordinates in favor of the much stronger majority task boundaries.
5. **Stream Heterogeneity and the Robustness-Accuracy Illusion:** Table 3 shows that mixed-task streams cause severe "heterogeneity collapse" in linear ($67.20\% \to 51.10\%$) and quantum ($36.10\% \to 10.80\%$) routers due to batch-averaging. L3-Softmax drops by only $4.10\%$ ($54.40\% \to 50.30\%$). However, L3-Softmax's absolute accuracy is inferior to the Linear Router's in both scenarios. This beautifully supports the "Robustness-Accuracy Illusion" claim, demonstrating that relative stability is merely an artifact of Softmax's simplex constraints forcing routing coefficients toward a mediocre, uniform average.
6. **Real-Scale CLIP-ViT-B/16 Pilot:** The pilot in Section 4.5 replicates all major trends: static uniform is at $72.40\%$, global Linear Router is at $88.60\%$, QWS SOTA collapses to $41.20\%$, and L3-Linear achieves $84.80\%$ Joint Mean accuracy. This proves that the sandbox is highly predictive of real-world parameter manifolds.
