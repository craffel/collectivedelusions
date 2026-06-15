# 3. Soundness and Methodology

An evaluation of the clarity of description, appropriateness of methods, potential technical flaws, and reproducibility.

## Clarity of Description
The mathematical and architectural description in the paper is exceptionally clear and structured:
- Every phase of the two main paradigms (EER and EPL-OCA) is explicitly formulated, with clear and mathematically precise equations (Equations 1 through 14).
- Key variables and parameters (such as tracking rate $\beta$, routing temperature $\tau$, vocabulary size $Y_k$, and ensembling coefficient $\alpha_{k, b}$) are well-defined and consistently referenced.
- The distinction between the synthetic sandbox setup (orthogonal subspaces and class prototypes) and the real-world embeddings (512-dimensional ResNet-18) is clearly articulated.

## Appropriateness of Methods
- **Normalized Shannon Prediction Entropy (EER):** Utilizing Shannon entropy as a zero-shot confidence indicator is mathematically sound, and the proposed scale-invariant Normalized Shannon Entropy correctly resolves vocabulary-size bias in heterogeneous registries, which is a common real-world problem.
- **Chronological Centroid Tracking (EPL-OCA):** Centroid updates are designed to happen *after* routing and prediction for sample $x_b$ to prevent chronological data leakage. This is highly correct and aligns with real-world streaming applications.
- **Amortized Pseudo-Labeling:** Bypassing heavy entropy evaluations via a temporal caching policy ($N_{\text{amortize}} = 10$) is a highly appropriate, systems-level mitigation of the post-activation divergence FLOP bottleneck ($0.25 + 0.75K$ passes).
- **Centroid-Gated Routing (CG-EER):** Using a cosine similarity threshold ($\delta \ge 0.7$) to establish spatial representation validity boundaries is an appropriate and elegant way to neutralize out-of-distribution (OOD) expert overconfidence on real embeddings.

## Evaluation of Potential Technical Flaws and Assumptions
- **Idealized Synthetic Sandbox Assumptions:** The synthetic sandbox assumes orthogonal task subspaces and class orthogonality. While this is an idealized setting, the authors are highly honest about this limitation, explaining in Section 4.1 and Section 4.10 that real-world correlated manifolds have topological overlaps that would actually cause centroid-based methods to decay more gracefully and allow soft blending to act as a stronger spatial interpolator.
- **SVHN Noise Calibration:** Calibrating SVHN to $39.44\%$ accuracy with a severe noise scale of $0.56$ represents a highly degraded edge sensor. This extreme noise represents an aggressive "stress-test" that allows the authors to empirically demonstrate EER's unique ability to shield clean tasks from catastrophic cross-task noise infiltration, which is methodologically sound.
- **Overlapping Class Namespace and Evaluation Bias:** The authors honestly note that because tasks share the same integer namespace $\{0, \dots, 9\}$, incorrect routing results in a background chance probability of $\approx 10\%$, creating a slight optimistic bias. This transparent evaluation is highly commendable and demonstrates excellent scientific integrity.
- **Unsupervised online centroids failure:** The total collapse of EPL-OCA Hard/Soft on real embeddings is not a technical flaw of the paper, but rather a **profound and honest scientific finding**. Instead of hiding or tweaking the real feature results, the authors present a complete and rigorous evaluation of the *Entropy Calibration Discrepancy* and the *Self-Referential Pseudo-Label Corruption Loop*. This is exceptionally valuable for practitioners, as it clarifies exactly why and when online centroid tracking fails in production, and how to resolve it (via CG-EER or EPL-OCA Soft).

## Reproducibility
The paper is highly reproducible. All mathematical operations are explicit, and the key hyperparameters and configuration details are clearly documented in the text:
- **EMA centroid tracking rate:** $\beta = 0.05$
- **Softmax routing temperatures:** $\tau = 0.001$ (Hard) and $\tau = 0.5$ (Soft)
- **Centroid-gating threshold:** $\delta \ge 0.7$
- **Warm-up window:** $T_{\text{warmup}} = 200$ steps (with sensitivity sweep of $10, 50, 100, 200$ provided)
- **Amortization interval:** $N_{\text{amortize}} = 10$ (with sweep of $1, 5, 10, 20$ provided)
- **Task stream block sizes (temporal task locality):** $B_{\text{block}} \in [1, 10, 50, 100]$
- **Backbone models:** frozen ViT-Tiny sandbox and pre-trained ResNet-18
- **Datasets:** MNIST, FashionMNIST, CIFAR-10, SVHN
- **Hardware/Benchmark:** Single core of AMD EPYC 7763 CPU @ 2.45GHz

This comprehensive list of parameters and settings, combined with explicit mathematical formulations, ensures that any researcher or practitioner can readily reproduce the results.
