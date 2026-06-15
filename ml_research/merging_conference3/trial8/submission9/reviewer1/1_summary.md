# Evaluation Phase 1: Summary of the Paper

## 1. Main Topic and Scope
The paper addresses the challenge of serving multiple task-specific Low-Rank Adaptation (LoRA) experts at test-time under resource-constrained on-device settings without relying on offline labeled calibration splits. It explores "Zero-Shot Calibration-Free Model Merging" in non-stationary streaming environments with heterogeneous inputs and continuous covariate shift (domain drift).

## 2. Proposed Approaches
The authors propose and evaluate several paradigms:
- **Zero-Shot Expert Entropy Routing (EER) [Accuracy-First]:** A direct routing approach where each incoming sample is passed through the early blocks of a shared frozen backbone and then through all specialized LoRA experts in parallel. The sample is routed to the expert with the minimum scale-invariant **Normalized Shannon Entropy**, defined as:
  $$\bar{H}(p_k(x_b)) = \frac{H(p_k(x_b))}{\log(Y_k)}$$
  where $Y_k$ is the class vocabulary size of expert $k$. This normalization is designed to eliminate bias toward smaller vocabularies.
- **Entropy-Pseudo-Labeled Online Centroid Adaptation (EPL-OCA) [Efficiency-First]:** A centroid-based ensembling approach that dynamically maintains running task centroids in the representation space. It uses EER to generate pseudo-labels for incoming samples and updates the corresponding running centroids sequentially (avoiding chronological data leakage). Cosine similarity to the running centroids is then used to determine soft ensembling weights ($\alpha_{k,b}$) via a Softmax function with temperature $\tau$, which are used to merge experts in activation space via Single-Pass Activation-Space Dynamic Blending (SPS).
- **Centroid-Gated Entropy Routing (CG-EER):** A semi-supervised hybrid method developed for real representation spaces (e.g., ResNet-18). It utilizes pre-computed offline task centroids to gate out experts when the representation cosine similarity is below a threshold ($\delta \ge 0.7$), neutralizing out-of-distribution (OOD) overconfidence.
- **Amortized Pseudo-Labeling:** A systems optimization designed to reduce the computational overhead of EER (which otherwise scales as $0.25 + 0.75K$ forward passes). It caches routing decisions and only runs the full entropy evaluation every $N_{\text{amortize}} = 10$ steps, which is shown to be effective under temporal task locality.

## 3. Key Findings
- **Synthetic Sandbox Evaluation (192D, Gaussian manifolds):**
  - EER achieves a **71.38%** Joint Mean accuracy under heterogeneous shuffled streams, outperforming the supervised SPS-ZCA baseline (66.76%) by **+4.62%** absolute.
  - Under extreme continuous representation drift ($d=0.45$), EER maintains complete robustness, achieving **71.18%** accuracy.
  - EPL-OCA achieves a lower Joint Mean accuracy of **49.88%** (and **49.78%** under drift) due to the **Representational Sparsity Paradox** (orthogonal class representations within task subspaces introduce high spatial sparseness, causing centroids to jitter between prototypes).
- **Real-World ResNet-18 Embedding Evaluation (512D):**
  - EER's accuracy drops to **35.38%** due to the **Entropy Calibration Discrepancy**—simpler experts (like MNIST) exhibit severe OOD overconfidence, producing very low entropy predictions on complex OOD inputs (e.g., SVHN or CIFAR-10) and biasing routing decisions.
  - EPL-OCA experiences a **complete methodological collapse** on real features, dropping to **27.45%** (Hard) and **31.52%** (Soft), failing to outperform static Uniform Weight Merging (31.66%). This is because the overconfident pseudo-labels corrupt the running centroids in a self-referential feedback loop.
  - CG-EER resolves this OOD overconfidence, achieving **61.50%** accuracy (outperforming supervised SPS-ZCA by **+0.70%**).
  - Making CG-EER completely unsupervised via online-accumulated centroids (**UCG-EER**) results in a catastrophic accuracy collapse to **28.45%** due to the self-referential pseudo-label corruption loop.
- **Systems Profiling:**
  - Amortized EER reduces CPU latency to **0.2211 ms per sample** (a practical $1.57\times$ overhead compared to $6.52\times$ for full EER) and slashes energy footprint by **$4.14\times$** while preserving high accuracy (**71.20%**).

## 4. Explicitly Claimed Contributions
1. **First study of calibration-free, zero-shot dynamic model ensembling**, removing the offline labeled data bottleneck.
2. **Formulation of EER**, which outperforms supervised ZCA in the synthetic sandbox and survives covariate shift.
3. **Identification and mitigation of chronological data leakage** in online centroid updates for an honest evaluation of EPL-OCA.
4. **Validation on real 512D ResNet-18 embeddings**, diagnosing the *Entropy Calibration Discrepancy* and proposing the hybrid *CG-EER* to resolve OOD overconfidence.
5. **Investigation of fully unsupervised centroid-gated routing (UCG-EER)**, uncovering the *self-referential pseudo-label corruption loop*.
6. **Mathematical formulation of systems-level FLOP serving complexity** under activation divergence and demonstrating the practicality of Amortized EER.
