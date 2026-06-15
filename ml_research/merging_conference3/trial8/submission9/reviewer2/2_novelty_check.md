# 2. Novelty Check

An assessment of the key novel aspects, the "delta" from prior work, and the characterization of novelty.

## Delta from Prior Work
The proposed work is situated at the intersection of parameter merging, dynamic activation-space ensembling, and test-time adaptation (TTA). The delta from existing SOTA methods is substantial and highly practical:
1. **Dynamic Activation-Space Ensembling (SPS-ZCA, SABLE, PFSR):** While these training-free methods achieve dynamic ensembling, they are strictly dependent on offline, labeled calibration splits ($|\mathcal{C}_k|=64$) to calculate task-routing centroids before deployment. This work is the first to achieve dynamic activation-space ensembling completely unsupervised and calibration-free, eliminating the offline labeled data bottleneck.
2. **Static Parameter-Space Merging (TIES, DARE, AdaMerging):** These methods suffer from severe parameter interference when merging highly conflicting tasks, assume uniform distribution of inputs, or require expensive offline optimization loops. The proposed methods operate dynamically in the forward activation space, maintaining independent expert representations while avoiding offline optimization.
3. **Test-Time Adaptation (TENT, CoTTA):** Traditional TTA requires online backpropagation to minimize entropy. On low-power edge nodes, this creates massive memory overhead (storing activation graphs), significantly inflates latency ($3\times$ to $5\times$), and risks optimization instability. The proposed methods operate entirely in the forward pass, using training-free, geometrically-sound entropy pseudo-labeling and running centroids, resulting in zero backpropagation, low deterministic latency, and stable on-device serving.

## Key Novel Aspects
- **Normalized Shannon Prediction Entropy:** To scale zero-shot routing to heterogeneous registries with varying class vocabularies $Y_k$, the paper introduces a scale-invariant Normalized Shannon Entropy:
  $$\bar{H}(p_k(x_b)) = \frac{H(p_k(x_b))}{\log(Y_k)}$$
  This mathematically neutralizes the bias of raw entropy toward experts with smaller vocabularies.
- **Chronological Data-Leakage Correction:** The authors identify and correct a hidden data-leakage bottleneck in online centroid updates, ensuring that centroid updates occur strictly *after* routing and prediction steps are completed for the current sample.
- **Amortized Pseudo-Labeling under Temporal Locality:** Recognizing that real-world streaming data exhibits temporal locality (test inputs arrive in blocks of the same task), the authors propose Amortized Pseudo-Labeling ($N_{\text{amortize}} = 10$). This is a highly novel, practitioner-first system optimization that caches routing decisions, bringing the serving complexity down to an extremely practical $1.3\times$ passes.
- **Delineation of the "Entropy Calibration Discrepancy" and "Self-Referential Loop":** On real ResNet-18 embeddings, the authors discover that simpler experts suffer from severe out-of-distribution (OOD) overconfidence (lower entropy on foreign data than on in-distribution data). They show that this discrepancy corrupts the online centroids via a self-referential pseudo-label loop, providing deep theoretical and empirical insights into the limits of fully unsupervised test-time adaptation.
- **Centroid-Gated Entropy Routing (CG-EER):** A hybrid semi-supervised framework that combines spatial gating (using small offline pre-computed centroids) with prediction-entropy routing to break the self-referential loop and achieve SOTA accuracy on real embeddings.

## Characterization of Novelty
The novelty of this paper is **significant and highly pragmatic**. 
Rather than proposing an incremental mathematical tweak that works only in an idealized simulation, the authors have conducted a rigorous "honest" study that exposes the severe limitations of online unsupervised centroid adaptation on real-world uncalibrated embeddings. 
From a practitioner's perspective, this is a major contribution: identifying and mathematically characterizing the *Entropy Calibration Discrepancy* and the *Self-Referential Pseudo-Label Corruption Loop* saves weeks of engineering effort, redirecting practitioners from unstable fully-unsupervised online centroids toward more viable hybrid semi-supervised approaches (CG-EER) or soft regularizers (EPL-OCA Soft). The combination of systems-level latency/energy profiling with deep geometric analyses represents a exemplary, holistic systems-ML contribution.
