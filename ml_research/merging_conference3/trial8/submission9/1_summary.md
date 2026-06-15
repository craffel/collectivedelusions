# 1. Summary of the Paper

This paper presents a comprehensive, systems-ML comparative study of **Zero-Shot Calibration-Free Model Merging** for test-time ensembling of Low-Rank Adaptation (LoRA) experts. The central objective of this work is to eliminate the severe operational bottleneck of state-of-the-art dynamic ensembling frameworks (such as SPS-ZCA and SABLE), which rely on pre-computing task-routing centroids using offline, labeled calibration splits (typically $|\mathcal{C}_k|=64$ samples). The paper focuses on privacy-restricted, zero-downtime, and streaming edge environments where labeled calibration data is entirely unavailable and on-device backpropagation is too resource-intensive.

The authors formulate, evaluate, and compare two primary calibration-free serving paradigms:
1. **Zero-Shot Expert Entropy Routing (EER) [Accuracy-First]:** A direct-routing approach that processes incoming samples through all $K$ specialized expert adapters in parallel, computes a proposed scale-invariant *Normalized Shannon Entropy* to measure prediction confidence, and routes 100% of the compute to the expert exhibiting the minimum entropy.
2. **Entropy-Pseudo-Labeled Online Centroid Adaptation (EPL-OCA) [Efficiency-First]:** A centroid-routing approach that maintains running task centroids on-the-fly to enable single-pass serving. It pseudo-labels incoming samples using EER to identify the active expert, updates the corresponding task centroid via a running average, and performs single-pass activation ensembling via Single-Pass Activation-Space Dynamic Blending (SPS) based on cosine similarity to the centroids.

### Core Evaluation and Findings

- **Synthetic Sandbox:** Evaluated in a 192-dimensional multi-task sandbox modeling MNIST, FashionMNIST, CIFAR-10, and SVHN as orthogonal Gaussian manifolds.
  - **EER** achieves an outstanding **71.38%** Joint Mean accuracy under heterogeneous shuffled streams, outperforming the supervised SOTA baseline (SPS-ZCA, 66.76%) by **+4.62%** absolute accuracy. Under extreme continuous representation drift ($d=0.45$), EER maintains complete robustness with **71.18%** Joint Mean accuracy.
  - **EPL-OCA Hard** ($\tau = 0.001$) experiences the **Representational Sparsity Paradox**, where intra-task class orthogonality introduces high spatial sparseness and noise in the centroid representation space, causing accuracy to degrade to **49.88%** (and **49.78%** under drift).
  - **EPL-OCA Soft** ($\tau = 0.5$) mitigates this by acting as a spatial regularizer, achieving **61.62%** accuracy (+11.74% absolute improvement over EPL-OCA Hard).
  - **Streaming K-Means Baseline:** An entirely unsupervised baseline with a Hungarian matching step at $T_{\text{warmup}}$ achieves only **30.29%** (Static) and **27.38%** (Refined) Joint Mean accuracy, highlighting the severe collapse under the Representational Sparsity Paradox without entropy-guided soft supervision.
  - **Zero-Shot Cosine Routing (ZCR):** Averaging classification head weights yields a poor **26.88%** accuracy, demonstrating that weight-space centroids fail due to class orthogonality.

- **Real Representation Embeddings (ResNet-18):** On real ImageNet-pre-trained features:
  - EER experiences the **Entropy Calibration Discrepancy and OOD Overconfidence**, where simpler domains (like MNIST) produce lower entropy on out-of-distribution (OOD) data than complex domains in-distribution, dropping accuracy to **35.38 ± 0.66%**.
  - The authors propose **Centroid-Gated Entropy Routing (CG-EER)**, applying an unsupervised threshold ($\delta \ge 0.7$) on representation-space cosine similarity to task centroids. If similarity is below 0.7, that expert is gated out (its routing entropy is set to 1.0). CG-EER achieves **61.50 ± 0.18%** accuracy, outperforming SPS-ZCA by **+0.70%** absolute accuracy.
  - Attempting to make CG-EER completely calibration-free via unsupervised online centroids (**UCG-EER**) yields only **28.45 ± 1.59%** accuracy, collapsing due to a self-referential pseudo-label corruption loop where early overconfident MNIST selections corrupt the running centroids.

- **Systems-ML Complexity Analysis:** 
  - To compute EER's prediction entropy, the mid-to-late blocks (Layers 4 to 12) must be executed independently $K$ times due to post-Layer-4 activation divergence, resulting in a FLOP cost of $0.25 + 0.75K$ forward passes ($3.25\times$ for $K=4$).
  - To mitigate this, the authors propose **Amortized Pseudo-Labeling**, which caches routing decisions and runs the full entropy pseudo-labeler only once every $N_{\text{amortize}} = 10$ steps. Under temporal task locality (coherent streams of block size $\ge 10$), this reduces CPU latency from $6.52\times$ to **$1.57\times$** ($0.2211$ ms per sample) and slashes edge energy footprint by **$4.14\times$**, while maintaining high accuracy (**71.20%**).
