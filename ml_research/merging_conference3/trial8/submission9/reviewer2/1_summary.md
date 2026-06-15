# 1. Summary of the Paper

This paper presents a comprehensive and technically rigorous study of **Zero-Shot Calibration-Free Model Merging** for dynamic test-time model ensembling of Low-Rank Adaptation (LoRA) experts on resource-constrained edge devices. 

## Main Topic and Motivation
State-of-the-art dynamic activation ensembling frameworks (such as SPS-ZCA and SABLE) rely on pre-computing task-routing centroids from offline, labeled calibration splits (typically requiring $|\mathcal{C}_k|=64$ annotated samples per task). This requirement is a critical bottleneck in privacy-restricted, zero-downtime, or streaming edge environments where data cannot be uploaded, manual labels are unavailable, and plug-and-play operation is mandatory. To resolve this bottleneck, the paper explores fully unsupervised, calibration-free test-time routing and adaptation.

## Evaluated Approaches
The authors propose and compare two primary architectural paradigms:
1. **Zero-Shot Expert Entropy Routing (EER) [Accuracy-First]:** This paradigm bypasses representation-space centroids entirely. Instead, it routes incoming test samples on-the-fly using the scale-invariant **Normalized Shannon prediction entropy** of all $K$ expert adapters:
   $$\bar{H}(p_k(x_b)) = \frac{H(p_k(x_b))}{\log(Y_k)} \in [0, 1]$$
   The sample is routed directly to the expert $k^*$ exhibiting the minimum normalized entropy (maximum confidence):
   $$k^* = \arg\min_{k} \bar{H}(p_k(x_b))$$
2. **Entropy-Pseudo-Labeled Online Centroid Adaptation (EPL-OCA) [Efficiency-First]:** This paradigm maintains running task centroids $\{\mu_k\}_{k=1}^K$ on-the-fly to enable single-pass ensembling. It routes inputs by calculating similarity to the previously updated centroids, performs Single-Pass Activation-Space Dynamic Blending (SPS), and updates the centroids after the prediction step using EER-guided pseudo-labels to avoid chronological data leakage.
3. **Centroid-Gated Entropy Routing (CG-EER) [Hybrid Semi-Supervised]:** To resolve out-of-distribution (OOD) overconfidence on real embeddings, this hybrid method applies an unsupervised threshold ($\delta \ge 0.7$) on the representation-space cosine similarity to each task's pre-computed offline centroid, gating out experts that are spatially far from the sample.
4. **Unsupervised Centroid-Gated Entropy Routing (UCG-EER) [Fully Unsupervised]:** A variant of CG-EER where gating centroids are obtained on-the-fly via unsupervised online-accumulated centroids (EPL-OCA) instead of pre-computed offline centroids.
5. **Amortized Pseudo-Labeling:** A systems-level optimization that runs heavy entropy pseudo-labeling only once every $N_{\text{amortize}} = 10$ steps (utilizing temporal task locality/coherent streams) and caches the routing decision for intermediate steps to bring complexity down from $0.25 + 0.75K$ passes to $\approx 1.3\times$ passes.

## Key Findings and Empirical Evidence
- **In the Synthetic Sandbox (192-dimensional representation space):**
  - **EER** achieves **71.38% Joint Mean accuracy**, outperforming the supervised SPS-ZCA baseline (66.76%) by **+4.62%** absolute.
  - Under extreme linear domain drift ($d=0.45$), EER maintains complete robustness, delivering **71.18%** Joint Mean accuracy.
  - **EPL-OCA** achieves only **49.88%** Joint Mean (49.78% under drift) due to the **Representational Sparsity Paradox** (class-specific orthogonality within task subspaces causes centroids to jitter between class prototypes instead of converging on task manifolds). However, softening the ensembling temperature to $\tau = 0.5$ (EPL-OCA Soft) boosts accuracy to **61.62%** (+11.74% absolute improvement) by acting as a spatial regularizer.
  - Streaming K-Means collapses to **30.29%** (Static) and **27.38%** (Refined), highlighting the value of EPL-OCA's entropy-based soft supervision (+19.59% over K-Means).
- **On Real 512-dimensional ResNet-18 Embeddings:**
  - **EER** drops to **35.38%** Joint Mean accuracy due to the **Entropy Calibration Discrepancy** where simpler experts (e.g., MNIST) exhibit severe OOD overconfidence, claiming SVHN/CIFAR-10 samples with extremely low entropy.
  - **CG-EER** resolves this, achieving **61.50% Joint Mean accuracy**, outperforming SPS-ZCA (60.80%) by **+0.70%** absolute.
  - **UCG-EER** collapses to **28.45%** accuracy due to a **self-referential pseudo-label corruption loop** (overconfident MNIST expert claims SVHN/CIFAR-10 early on, polluting the running MNIST centroid with OOD features).
  - **EPL-OCA** experiences a **complete methodological collapse** (**27.45%** Hard, **31.52%** Soft), failing to outperform static Uniform Weight Merging (**31.66%**), driven by the same self-referential loop.
- **Physical Edge serving & Latency:**
  - CPU profiling shows that Amortized EER ($N_{\text{amortize}}=10$) slashes latency from $0.9166$ ms per sample ($6.52\times$ overhead) to **0.2211 ms** per sample (**$1.57\times$ overhead**).
  - Energy profiling shows Amortized EER reduces energy footprint by **$4.14\times$** (down to $\approx 0.11 \mu\text{J}$ per sample on typical edge hardware) while preserving high accuracy (**71.20%** Joint Mean on coherent streams with temporal locality block size $B_{\text{block}} \ge 10$).

## Explicitly Claimed Contributions
1. **First study** of completely calibration-free, zero-shot dynamic model ensembling, eliminating the offline labeled data bottleneck.
2. **EER formulation & evaluation** achieving 71.38% Joint Mean accuracy, outperforming SOTA offline-supervised SPS-ZCA.
3. **Chronological data leakage fix** in online centroid updates, providing an honest, non-leaking evaluation of EPL-OCA.
4. **Real ResNet-18 validation & ECD formulation**, revealing the fragility of pure zero-shot routing and proposing CG-EER to resolve OOD overconfidence.
5. **UCG-EER analysis & self-referential loop discovery**, providing deep physical insights into the limits of pure unsupervised online adaptation.
6. **FLOP complexity and edge-deployment trade-off profiling**, demonstrating CPU runtime speedups and energy savings.
