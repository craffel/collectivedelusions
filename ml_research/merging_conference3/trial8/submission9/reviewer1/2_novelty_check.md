# Evaluation Phase 2: Novelty and Literature Delta

## 1. Delta from Prior Work
The paper positions its work relative to three main bodies of literature:
- **LoRA and Weight-Space Parameter Merging (e.g., Task Arithmetic, TIES-Merging, AdaMerging):** These methods create a static merged model in weight-space. The paper notes they suffer from parameter interference (especially for heterogeneous tasks), assume a homogeneous input distribution, and require expensive offline optimization loops.
- **Dynamic Routing and Activation-Space Ensembling (e.g., Mixture of Experts (MoE), SABLE, SPS-ZCA):** MoE requires expensive training from scratch. SPS-ZCA and SABLE enable dynamic activation-space ensembling but require pre-computing task centroids from an **offline labeled calibration dataset** ($|\mathcal{C}_k|=64$). The proposed EER and EPL-OCA paradigms operate completely calibration-free and zero-shot, removing this labeled calibration bottleneck.
- **Test-Time Adaptation (TTA) (e.g., TENT, CoTTA):** Traditional TTA methods update model parameters during inference using gradient descent (e.g., minimizing prediction entropy). The paper highlights that this is bottlenecked by the memory and latency of backpropagation and is prone to optimization instability. The proposed approaches operate entirely in the forward pass with zero backpropagation, ensuring low and stable latency.

## 2. Characterization of Novelty
The novelty of the paper's proposed methods is best characterized as **incremental but highly practical**.
- **Expert Entropy Routing (EER):** The concept of using prediction entropy as a confidence metric for task selection is well-established in uncertainty estimation and test-time adaptation. The adaptation of this heuristic to zero-shot routing among LoRA experts is a natural extension of prediction confidence heuristics. The scale-invariant *Normalized Shannon Entropy* is also a straightforward algebraic correction for differing vocabulary sizes.
- **Entropy-Pseudo-Labeled Online Centroid Adaptation (EPL-OCA):** The combination of pseudo-labeling via prediction entropy with running-average centroids (exponential moving average) is a standard technique in online clustering and semi-supervised learning. Applying it to activation-space blending (SPS) is an elegant combination of existing building blocks rather than a fundamental mathematical breakthrough.
- **Centroid-Gated Entropy Routing (CG-EER):** This is a heuristic patch designed to fix the *Entropy Calibration Discrepancy* on real embeddings by utilizing spatial gating with pre-computed centroids.

### Theorist Perspective on Novelty
From a theoretical perspective, the novelty is quite limited:
- The methods represent a **creative combination of existing heuristics** (prediction entropy, running averages, cosine similarity, activation blending) rather than a deep, theoretically-grounded mathematical innovation.
- The paper does not provide any new mathematical insights into the structure of representation spaces or the dynamics of online ensembling. Instead, it relies on empirical "sandboxes" and engineering trade-offs.
- Terms like "Representational Sparsity Paradox" and "Entropy Calibration Discrepancy" are coined to describe well-known empirical phenomena (e.g., high intra-class variance and OOD overconfidence) but are not mathematically formalized or theoretically analyzed.
- There are no formal proofs or theoretical bounds on the performance of EER, the convergence of online centroids, or the stability of the self-referential pseudo-label loop in UCG-EER.
