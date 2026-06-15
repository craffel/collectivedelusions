# Peer Review Report

## 1. Summary of the Paper
The paper presents an empirical and systems-ML study of "Zero-Shot Calibration-Free Model Merging," which aims to serve multiple task-specific Low-Rank Adaptation (LoRA) experts on resource-constrained edge devices without relying on pre-computed task centroids from offline labeled calibration data. 

The authors propose two main paradigms:
1. **Zero-Shot Expert Entropy Routing (EER) [Accuracy-First]:** A direct-routing method that passes incoming samples through the early shared blocks of a backbone and then through all LoRA experts in parallel, routing each sample to the expert with the minimum scale-invariant **Normalized Shannon prediction entropy**.
2. **Entropy-Pseudo-Labeled Online Centroid Adaptation (EPL-OCA) [Efficiency-First]:** A centroid-based ensembling method that dynamically tracks running task centroids in representation space. It uses EER to generate online pseudo-labels and updates task centroids via a running average. Cosine similarity to these centroids is then used to compute soft ensembling weights for merging expert activations in a single pass via Single-Pass Activation-Space Dynamic Blending (SPS).

The proposed methods are evaluated on a synthetic 192-dimensional representation sandbox and on real 512-dimensional embeddings from a pre-trained ResNet-18 model. The authors also conduct a systems-level serving complexity and CPU latency profiling of their pipelines, proposing **Amortized Pseudo-Labeling** to reduce the forward-pass complexity from $0.25 + 0.75K$ to approximately $1.3\times$ passes.

---

## 2. Strengths and Weaknesses

### Strengths
- **Exemplary Academic Honesty and Transparency:** The authors deserve significant commendation for their rigorous and transparent reporting. Rather than obscuring or downplaying the failures of their proposed zero-shot methods on real features, they clearly document and analyze these failures (the complete collapse of EER, EPL-OCA, and UCG-EER on ResNet-18 features). This provides high scientific value to the community by identifying fundamental boundaries of unsupervised test-time adaptation.
- **Thorough Systems-Level Evaluation:** The paper includes a solid practical profiling of physical serving overhead under post-activation divergence. The authors mathematically formulate the FLOP serving complexity, profile CPU latencies, and conduct a detailed theoretical energy and memory bandwidth analysis on edge nodes.
- **Exhaustive Ablation Studies:** The experimental section contains a wealth of valuable ablations. The authors analyze softmax temperature sensitivity, warm-up window sizes, temporal task locality under amortization, and vocabulary-size normalization.
- **Excellent Presentation and Clarity:** The paper is extremely well-structured, easy to follow, and clearly written. The mathematical formulations and algorithmic steps are presented with precision.

### Weaknesses
- **Lack of Rigorous Theoretical Grounding:** Although the authors describe their work as "comprehensive" and "technically rigorous," the paper **completely lacks formal theoretical foundations**. There are no theorems, lemmas, propositions, or formal mathematical proofs. All proposed methods (EER, EPL-OCA, CG-EER) are empirical heuristics. The paper relies heavily on intuitive, qualitative explanations (e.g., "Representational Sparsity Paradox," "Entropy Calibration Discrepancy") rather than formal mathematical modeling.
- **No Analytical Justification for Prediction Entropy as a Task Routing Metric:** The central assumption of EER—that prediction entropy can act as a reliable zero-shot surrogate for task-routing—is a heuristic. There is no formal theoretical framework establishing why the correct expert's entropy should be bounded below that of incorrect experts under reasonable distribution assumptions. Indeed, the paper's own empirical results on real-world ResNet-18 features *disprove* this heuristic: simpler experts (like MNIST) exhibit severe out-of-distribution (OOD) overconfidence, yielding lower entropy on OOD data than on their own in-distribution data, which collapses the zero-shot routing accuracy to 35.38%.
- **No Convergence or Stability Analysis of the Self-Referential pseudo-label update loop:** In EPL-OCA, centroids are updated on-the-fly using self-generated pseudo-labels, which are then used to compute routing similarities. This creates a self-referential feedback loop. On real features, this loop catastrophically collapses (EPL-OCA Hard drops to 27.45%, and UCG-EER drops to 28.45%). While the authors describe this collapse, they offer no mathematical framework or stability analysis of this dynamical update system. A theoretically sound paper would analyze the convergence properties of these online updates using stochastic approximation or dynamical-systems theory.
- **Extreme Simplifications in the Synthetic Sandbox:** The 192D synthetic sandbox relies on perfect subspace orthogonality, class orthogonality, and isotropic Gaussian noise. These highly idealized assumptions are unrepresentative of real deep-learning representation manifolds, which exhibit high topological overlaps and non-isotropic noise. Since both zero-shot methods fail on real embeddings, the high performance in the synthetic sandbox has limited scientific and practical generalizability.
- **Overlapping Class Namespace Bias:** The evaluation across all tasks uses an overlapping namespace $\{0, \dots, 9\}$. As the authors acknowledge, this introduces an optimistic bias of $\approx 10\%$ absolute accuracy because incorrect routing can result in a "false correct" prediction. A more rigorous evaluation would utilize disjoint class spaces.

---

## 3. Detailed Dimension Ratings

### Soundness
- **Rating:** Fair
- **Justification:** The experimental evaluations, systems profiling, and ablation studies are highly detailed and exceptionally honest. However, from a theoretical standpoint, the soundness is limited. The paper relies entirely on empirical heuristics without formal mathematical guarantees or proofs. Furthermore, the primary "completely calibration-free zero-shot" claims completely collapse when moving from an idealized synthetic sandbox to real-world embeddings, forcing a reliance on a semi-supervised hybrid method (CG-EER) that requires pre-computed offline centroids, which undermines the core thesis of the paper.

### Presentation
- **Rating:** Excellent
- **Justification:** The paper is exceptionally well-written, logically structured, and easy to read. The algorithms are clearly stated with precise algebraic notation, and the tables, figures, and systems analysis are documented with outstanding clarity.

### Significance
- **Rating:** Fair
- **Justification:** On-device serving of multiple LoRA experts under streaming conditions is an important and timely problem. However, because the proposed calibration-free, zero-shot methods (EER and EPL-OCA) completely collapse on real embeddings, their immediate practical significance is limited. The only working real-world method, CG-EER, is semi-supervised and requires offline calibration data, meaning the paper does not fully deliver on its "calibration-free" promise for real deployments. Nonetheless, the practical systems-level amortization strategy and the detailed diagnosis of failure modes provide a highly useful foundation for future researchers.

### Originality
- **Rating:** Good
- **Justification:** The combination of prediction-entropy routing, EMA-based online centroid adaptation, and single-pass activation blending is a creative and original combination of existing techniques. Additionally, the empirical analysis and diagnosis of the Entropy Calibration Discrepancy and the self-referential corruption loop on real features represent highly valuable and original scientific contributions.

---

## 4. Overall Recommendation
- **Rating:** 3: Weak Reject
- **Justification:** 
The paper possesses clear merits: exemplary academic honesty, excellent systems-level complexity and latency profiling, a thorough set of baselines, and highly detailed ablation studies. However, the weaknesses currently outweigh these merits:
1. **Critical Lack of Theoretical Rigor:** Despite using formalistic language, the paper provides no formal mathematical proofs, theorems, or convergence analyses. The proposed methods are purely empirical heuristics.
2. **Failure of Primary Claims on Real Embeddings:** The primary contribution—completely calibration-free, zero-shot model ensembling—is shown to be non-functional on real ResNet-18 embeddings due to uncalibrated OOD overconfidence and self-referential feedback corruption. The only viable real-world pipeline (CG-EER) is semi-supervised, reverting to pre-computed offline centroids and violating the calibration-free thesis.
3. **Synthetic-to-Real Disconnect:** The synthetic sandbox relies on unrealistic assumptions (perfect subspace and class orthogonality) that artificially isolate the proposed methods from real-world representation challenges, rendering the synthetic success of limited scientific generalizability.

To make this paper ready for publication, the authors need to either (a) provide a formal mathematical/dynamical-systems analysis of their online update loops and representational limits to elevate the theoretical value of the work, or (b) tone down their theoretical claims, re-frame the paper to focus on the semi-supervised hybrid paradigm as the primary real-world contribution, and thoroughly address the synthetic-to-real gap.

---

## 5. Questions and Constructive Feedback for the Authors

1. **Analytical Formulation of the Self-Referential Loop:** Can you model the online centroid update (Eq. 10) as a discrete-time dynamical system or a stochastic approximation algorithm? Providing a mathematical analysis of the stability and convergence conditions of this loop under calibration discrepancy would significantly elevate the theoretical rigor of the paper.
2. **Formalizing the Representational Sparsity Paradox:** Rather than treating this qualitatively, can you provide a formal proof or mathematical bound showing how class orthogonality and representation-space dimensionality impact the variance of the running centroid and the subsequent cosine similarity routing error?
3. **Addressing the Overlapping Namespace Bias:** To provide an unbiased evaluation, can you re-evaluate your pipelines using disjoint class namespaces across tasks (e.g., $Y_k = [10k, 10k+9]$) to eliminate the $\approx 10\%$ background chance probability and verify the pure routing capabilities of your models?
4. **Generalization to Modern Architectures:** Since model merging and LoRA ensembling are most commonly applied to large-scale Vision Transformers (ViT) or Large Language Models (LLMs), do you have any preliminary results or theoretical insights into whether the Entropy Calibration Discrepancy and self-referential corruption loops manifest similarly on high-dimensional Transformer embeddings (e.g., 768D or 4096D) compared to the 512D ResNet-18 linear heads?
